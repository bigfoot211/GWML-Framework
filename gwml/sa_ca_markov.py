# -*- coding: utf-8 -*-
"""
gwml.sa_ca_markov
=================
模块 3（情景预测）之机理约束的元胞自动机-马尔可夫模型 SA-CA-Markov
（Scenario-Anchored / Self-Adaptive Cellular Automata-Markov）。

对应论文：
    英文版 3.4 节 "Module 3: Scenario-Anchored CA-Markov (SA-CA-Markov)"；
    中文版 2.3 节 "机理约束 SA-CA-Markov 模拟模型构建"。

核心公式（论文式 (7)）
---------------------
    P'(i→j | u) ∝ P(i→j) × S_j(u) × C_j(u)

其中：
    P      ：由历史两期土地利用（t1→t2）统计得到的原始转移概率矩阵；
    S_j(u) ：GWRF 输出的地类 j 在位置 u 的空间适宜性系数（归一化 [0,1]）；
    C_j(u) ：GNID 输出的地类 j 因子空间约束系数（归一化 [0,1]）；
    P'     ：逐像元校正后的转移概率（每行归一化）。

模块能力
--------
1. 由两期土地利用栅格估计马尔可夫转移概率矩阵（并给出多年预测需求）；
2. 以"适宜性 × 约束 × 邻域支持"打分，执行 CA 式空间分配，逐期迭代模拟；
3. 情景化约束：基准发展（BAU，无约束）/ 生态保护 / 均衡发展；
4. 回溯检验：用 t1、t2 训练并模拟 t2（或下一期），对比实测栅格，
   输出总体精度、Kappa 与面积误差（论文表 5 的回溯验证）。
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def transition_matrix(landuse_t1: np.ndarray, landuse_t2: np.ndarray,
                      classes) -> np.ndarray:
    """由两期土地利用栅格估计马尔可夫转移概率矩阵。

    P[i, j] = count(类 i 在 t1 且类 j 在 t2) / count(类 i 在 t1)
    （行归一化；行和不足 1 时补齐到自身，保证每行和为 1。）

    Returns
    -------
    P : (K, K) ndarray
    """
    a = np.asarray(landuse_t1)
    b = np.asarray(landuse_t2)
    if a.shape != b.shape:
        raise ValueError("两期栅格形状必须一致")
    classes = list(classes)
    idx = {c: k for k, c in enumerate(classes)}
    K = len(classes)
    P = np.zeros((K, K))
    flat_a, flat_b = a.ravel(), b.ravel()
    for k1, c1 in enumerate(classes):
        m1 = flat_a == c1
        total = int(m1.sum())
        if total == 0:
            continue
        for k2, c2 in enumerate(classes):
            P[k1, k2] = int((flat_b[m1] == c2).sum()) / total
    rowsum = P.sum(axis=1)
    for k in range(K):
        if rowsum[k] > 0:
            P[k] /= max(rowsum[k], _EPS)
        else:
            P[k, k] = 1.0
    return P


def markov_demand(area_t1: np.ndarray, P: np.ndarray, steps: int) -> np.ndarray:
    """按马尔可夫链投影 steps 期后的各类面积需求。

    area_t1 : (K,) ndarray 各类在 t1 的像元数/面积
    P       : (K, K) 转移概率矩阵
    """
    a = np.asarray(area_t1, dtype=float).ravel()
    M = np.linalg.matrix_power(P, int(steps))
    return M.T @ a


class SACAMarkov:
    """机理约束的 CA-Markov 模型。

    Parameters
    ----------
    classes : list
        地类编码列表（如 [0, 1] 或 [0, 1, 2]）。第一个元素通常为背景/非目标。
    neighborhood : int
        CA 邻域窗口边长（奇数，默认 5；对应论文所用 5×5 邻近滤波器）。
    alpha : float
        邻域支持系数：打分 = 适宜性 × 约束 × (1 + alpha × 邻域同类占比)。
    random_state : int
        随机种子（用于并列打分时的稳定次序，默认确定性分配）。
    """

    def __init__(self, classes, neighborhood: int = 5, alpha: float = 1.0,
                 random_state: int = 42):
        self.classes = list(classes)
        self.K = len(self.classes)
        self.neighborhood = neighborhood
        self.alpha = alpha
        self.random_state = random_state
        self.transition_ = None
        self.area_t1_ = None

    # ------------------------------------------------------------------ #
    def fit(self, landuse_t1: np.ndarray, landuse_t2: np.ndarray):
        """以历史两期数据估计转移概率矩阵。

        Returns
        -------
        self（transition_ 属性为 (K, K) 转移矩阵）
        """
        self.transition_ = transition_matrix(landuse_t1, landuse_t2,
                                             self.classes)
        a = np.asarray(landuse_t1)
        self.area_t1_ = np.array([int((a == c).sum()) for c in self.classes],
                                 dtype=float)
        return self

    # ------------------------------------------------------------------ #
    def _neighborhood_support(self, current: np.ndarray, cls: int) -> np.ndarray:
        """计算每个像元邻域窗口内目标地类的占比（0~1）。"""
        K = self.neighborhood
        if K % 2 == 0:
            raise ValueError("neighborhood 必须为奇数")
        pad = K // 2
        m = (current == cls).astype(np.float64)
        padded = np.pad(m, pad, mode="constant", constant_values=0.0)
        from scipy.ndimage import uniform_filter
        support = uniform_filter(padded, size=K)[pad:-pad, pad:-pad]
        return support

    # ------------------------------------------------------------------ #
    def simulate(self, landuse_t0: np.ndarray, steps: int = 1,
                 suitability: dict | None = None,
                 constraints: dict | None = None,
                 transition: np.ndarray | None = None,
                 demand: np.ndarray | None = None,
                 mask: np.ndarray | None = None) -> np.ndarray:
        """执行机理约束的 CA-Markov 模拟。

        Parameters
        ----------
        landuse_t0 : (H, W) ndarray
            起始期土地利用栅格。
        steps : int
            模拟期数（如从 2020 到 2035 若一期为 15 年则 steps=1）。
        suitability : dict {class: (H,W) ndarray in [0,1]} | None
            GWRF 输出的各地类空间适宜性表面（S_j）。
        constraints : dict {class: (H,W) ndarray in [0,1]} | None
            GNID 输出的各地类空间约束表面（C_j，含情景阈值约束）。
            未给出的地类视为约束 = 1（无限制）。
        transition : (K,K) ndarray | None
            转移矩阵；None 时使用 fit() 得到的 transition_。
        demand : (K,) ndarray | None
            目标期末各地类像元数；None 时由马尔可夫投影自动计算。
        mask : (H,W) bool ndarray | None
            True 表示允许变化的像元；None 表示全部允许。

        Returns
        -------
        sim : (H, W) ndarray 期末模拟土地利用栅格。
        """
        cur = np.asarray(landuse_t0).copy()
        H, W = cur.shape
        classes = self.classes

        if transition is None:
            if self.transition_ is None:
                raise RuntimeError("未提供转移矩阵且尚未调用 fit()")
            transition = self.transition_
        P = np.asarray(transition, dtype=float)

        if demand is None:
            area = np.array([int((cur == c).sum()) for c in classes],
                            dtype=float)
            demand = markov_demand(area, P, steps)
        demand = np.asarray(demand, dtype=float)
        if demand.shape[0] != self.K:
            raise ValueError("demand 长度必须等于地类数")

        # 逐期迭代
        for step in range(int(steps)):
            # 本步目标面积 = 对需求进行逐步分配
            target = demand if steps == 1 else \
                markov_demand(
                    np.array([int((cur == c).sum()) for c in classes], dtype=float),
                    P, 1)
            cur = self._one_step(cur, P, target, suitability, constraints, mask)
        return cur

    # ------------------------------------------------------------------ #
    def _one_step(self, cur, P, target, suitability, constraints, mask):
        classes = self.classes
        H, W = cur.shape
        rng = np.random.default_rng(self.random_state)

        if mask is None:
            mask = np.ones((H, W), dtype=bool)
        else:
            mask = np.asarray(mask, dtype=bool)

        # 目标面积按整数像元
        target_int = np.floor(target).astype(int)
        # 供需差：需要增加的类
        cur_area = np.array([int((cur == c).sum()) for c in classes])
        delta = target_int - cur_area

        # 打分（对每个像元、每个目标类 j）
        score = np.zeros((self.K, H, W))
        for j, c in enumerate(classes):
            s = np.ones((H, W))
            if suitability is not None and c in suitability:
                s = s * np.clip(np.asarray(suitability[c], dtype=float), 0, 1)
            cons = np.ones((H, W))
            if constraints is not None and c in constraints:
                cons = cons * np.clip(np.asarray(constraints[c], dtype=float), 0, 1)
            neigh = self._neighborhood_support(cur, c)
            score[j] = s * cons * (1.0 + self.alpha * neigh)
        # 禁止变化的像元（mask=False）一律 0 分
        score[:, ~mask] = -1.0

        # 分配：对"需求增加"的类，从高分像元开始逐像元改类
        for j in range(self.K):
            if delta[j] <= 0:
                continue
            # 候选：当前不是 j，且从当前类 i 到 j 的转移概率 > 0
            cur_code = cur.copy()
            can_transition = np.zeros((H, W), dtype=bool)
            for i in range(self.K):
                if P[i, j] > 0:
                    can_transition |= (cur == classes[i])
            can_transition &= (cur != classes[j])
            can_transition &= mask

            if not can_transition.any():
                continue

            cand_score = score[j].copy()
            cand_score[~can_transition] = -1.0
            order = np.argsort(-cand_score, axis=None, kind="stable")
            need = int(delta[j])
            ys, xs = np.unravel_index(order, (H, W))
            placed = 0
            for t in range(order.size):
                if placed >= need:
                    break
                yy, xx = ys[t], xs[t]
                if not can_transition[yy, xx]:
                    continue
                cur[yy, xx] = classes[j]
                placed += 1
        return cur

    # ------------------------------------------------------------------ #
    def retrospective_validation(self, landuse_t0, landuse_observed,
                                 steps: int = 1, suitability=None,
                                 constraints=None) -> dict:
        """回溯检验：以 t0 模拟 steps 期，与实测栅格对比。

        Returns
        -------
        dict: overall_accuracy, kappa, area_observed(每个地类像元数),
              area_simulated, area_error_pct
        """
        sim = self.simulate(landuse_t0, steps=steps,
                            suitability=suitability, constraints=constraints)
        obs = np.asarray(landuse_observed)
        from .evaluate import kappa, overall_accuracy
        out = {
            "simulated": sim,
            "overall_accuracy": overall_accuracy(obs, sim),
            "kappa": kappa(obs, sim),
        }
        area_obs = np.array([int((obs == c).sum()) for c in self.classes],
                            dtype=float)
        area_sim = np.array([int((sim == c).sum()) for c in self.classes],
                            dtype=float)
        out["area_observed"] = area_obs
        out["area_simulated"] = area_sim
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = np.where(area_obs > 0,
                           (area_sim - area_obs) / area_obs * 100.0, 0.0)
        out["area_error_pct"] = pct
        return out


# --------------------------------------------------------------------------- #
# 情景约束表面构造（论文表 6 的三类情景）
# --------------------------------------------------------------------------- #


def make_ecological_constraint(elevation: np.ndarray, slope: np.ndarray,
                               class_target: int,
                               elev_threshold: float = 600.0,
                               slope_threshold: float = 25.0,
                               penalty: float = 0.05) -> np.ndarray:
    """生态保护情景约束面：高海拔、陡坡像元对目标地类（如建设用地扩张）
    施加低约束系数（惩罚），返回 [0,1] 栅格。

    Parameters
    ----------
    elevation, slope : (H, W) ndarray
        高程（m）与坡度（°）。
    class_target : int
        受约束的目标地类编码（仅该类的约束面需要惩罚）。
    elev_threshold, slope_threshold : float
        触发限制的高程/坡度阈值。
    penalty : float
        受限像元的约束系数（越小限制越强）。

    Returns
    -------
    cons : (H, W) ndarray in [0, 1]
    """
    cons = np.ones_like(np.asarray(elevation, dtype=float))
    bad = (elevation >= elev_threshold) | (slope >= slope_threshold)
    cons[bad] = penalty
    return cons


def make_balanced_constraint(elevation: np.ndarray, slope: np.ndarray,
                             gdp: np.ndarray, class_target: int,
                             elev_threshold: float = 800.0,
                             slope_threshold: float = 30.0,
                             gdp_percentile: float = 30.0,
                             eco_penalty: float = 0.1,
                             dev_boost: float = 1.5) -> np.ndarray:
    """均衡发展情景约束面：经济适宜区适度允许开发（>1），生态敏感区强约束。

    Returns
    -------
    cons : (H, W) ndarray in [0, 2]，使用前建议裁剪至 [0,1] 或直接参与打分。
    """
    elev = np.asarray(elevation, dtype=float)
    slp = np.asarray(slope, dtype=float)
    g = np.asarray(gdp, dtype=float)
    cons = np.ones_like(elev)
    sensitive = (elev >= elev_threshold) | (slp >= slope_threshold)
    cons[sensitive] = eco_penalty
    thr = np.percentile(g, gdp_percentile)
    economic = g >= thr
    cons[economic & ~sensitive] = dev_boost
    return cons
