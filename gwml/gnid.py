# -*- coding: utf-8 -*-
"""
gwml.gnid
=========
模块 2（机理解译）之局部因子交互探测器 GNID
（Geographically Non-Stationary / Neighborhood Interaction Detector）。

对应论文：
    英文版 3.3.1 节 "From Geodetector to GNID"；
    中文版 2.2 节 "局部因子交互探测 GNID"。

核心思想
--------
经典地理探测器输出全局 q 值，掩盖了因子的空间分异。GNID 在"任意空间位置 u"
上计算局地 q 值，形成连续 q 值表面：

    q(u) = 1 - Σ_h [ N_h(u) · σ²_h(u) ] / [ N(u) · σ²(u) ]

其中 h = 1..L 为因子分层（对连续因子先分层，如分位数分层）；N_h(u)、σ²_h(u)
为以 u 为中心、带宽为 b 的空间核窗口内第 h 层的有效样本量与加权方差；
N(u)、σ²(u) 为窗口内总有效样本量与总体加权方差（denominator='local'），
或全域样本总量与总方差（denominator='global'，论文口径）。

交互探测
--------
对因子 X1、X2 分别计算 q1(u)、q2(u)，再用二者交叉分层计算联合 q12(u)，
按地理探测器交互作用判别规则逐位置判定：
    - nonlinear_weakening     非线性减弱：q12 < min(q1, q2)
    - univariate_enhancement  单因子增强：min ≤ q12 < max
    - bivariate_enhancement   双因子增强：max ≤ q12 < q1 + q2
    - nonlinear_enhancement   非线性增强：q12 ≥ q1 + q2
    - independent             相互独立：q12 ≈ q1 ≈ q2
"""

from __future__ import annotations

import numpy as np

from .spatial_utils import euclidean_distance, kernel_weights, KERNELS


def quantile_strata(x: np.ndarray, n_strata: int = 5) -> np.ndarray:
    """连续变量按分位数分成 n_strata 层，返回层号（0..n_strata-1）。"""
    x = np.asarray(x, dtype=float)
    qs = np.unique(np.quantile(x, np.linspace(0, 1, n_strata + 1)))
    if qs.size < 2:
        return np.zeros(x.size, dtype=int)
    bins = qs[1:-1]
    labels = np.digitize(x, bins, right=False)
    return labels.astype(int)


class GNID:
    """GNID 局地 q 值表面与因子交互探测。

    Parameters
    ----------
    bandwidth : float | None
        空间核带宽。None 时依据坐标自动估算（近邻距离中位数启发式）。
    kernel : str
        'gaussian'（默认）或 'bisquare'。
    n_strata : int
        连续因子分层层数 L（默认 5）。
    denominator : str
        'global'（论文口径，窗口内层内方差、全域总方差）或 'local'
        （窗口内总体加权方差）。
    bandwidth_factor : float
        bandwidth=None 时的启发式系数：b = factor * median(kNN 距离)。
    k_neighbors : int
        启发式带宽所用的近邻数（默认 8）。
    """

    def __init__(self, bandwidth: float | None = None, kernel: str = "gaussian",
                 n_strata: int = 5, denominator: str = "global",
                 bandwidth_factor: float = 2.0, k_neighbors: int = 8):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.n_strata = n_strata
        if denominator not in ("global", "local"):
            raise ValueError("denominator 必须为 'global' 或 'local'")
        self.denominator = denominator
        self.bandwidth_factor = bandwidth_factor
        self.k_neighbors = k_neighbors

    # ------------------------------------------------------------------ #
    def _resolve_bandwidth(self, coords: np.ndarray) -> float:
        if self.bandwidth is not None:
            return float(self.bandwidth)
        dmat = euclidean_distance(coords)
        n = dmat.shape[0]
        k = min(self.k_neighbors, n - 1)
        if k < 1:
            return 1.0
        kth = np.partition(dmat, k, axis=1)[:, k]
        med = float(np.median(kth[kth > 0]))
        if med <= 0:
            med = float(np.median(dmat[dmat > 0]))
        return self.bandwidth_factor * max(med, 1e-9)

    # ------------------------------------------------------------------ #
    def _q_matrix(self, X: np.ndarray, y: np.ndarray,
                  coords: np.ndarray) -> np.ndarray:
        """对单个因子 X 计算各位置的局地 q 值向量（向量化实现）。

        q(u) = 1 - Σ_h N_h(u)·σ²_h(u) / (N(u)·σ²(u))
        """
        X = np.asarray(X, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        n = y.size
        if n < 2:
            raise ValueError("样本数不足")
        strata = quantile_strata(X, self.n_strata)
        L = int(strata.max()) + 1

        b = self._resolve_bandwidth(coords)
        dmat = euclidean_distance(coords)
        W = kernel_weights(dmat, b, self.kernel)  # n×n
        np.fill_diagonal(W, 1.0)  # 焦点位置自身权重恒为 1

        y2 = y * y
        # 各层加权统计
        num = np.zeros(n)  # Σ_h N_h σ²_h
        for h in range(L):
            mask = (strata == h).astype(float)
            Wh = W * mask[None, :]            # n×n
            Nh = Wh.sum(axis=1)               # (n,)
            ybar = (Wh @ y) / np.maximum(Nh, 1e-12)
            e2 = (Wh @ y2) / np.maximum(Nh, 1e-12)
            var_h = np.maximum(e2 - ybar ** 2, 0.0)
            num += Nh * var_h

        if self.denominator == "global":
            var_g = float(np.var(y))
            if var_g <= 0:
                return np.full(n, 1.0)
            q = 1.0 - num / (n * var_g)
        else:
            Ntot = W.sum(axis=1)
            ybar_t = (W @ y) / np.maximum(Ntot, 1e-12)
            e2_t = (W @ y2) / np.maximum(Ntot, 1e-12)
            var_t = np.maximum(e2_t - ybar_t ** 2, 0.0)
            q = 1.0 - num / np.maximum(Ntot * var_t, 1e-12)

        return np.clip(q, 0.0, 1.0)

    # ------------------------------------------------------------------ #
    def q_surface(self, X: np.ndarray, y: np.ndarray,
                  coords: np.ndarray) -> np.ndarray:
        """单因子局地 q 值表面。

        Parameters
        ----------
        X : (n,) or (n, 1) array_like
            单个候选驱动因子（连续变量）。
        y : (n,) array_like
            目标变量（如居民点演化指标）。
        coords : (n, 2) array_like
            空间坐标。

        Returns
        -------
        q : (n,) ndarray，取值 [0, 1]，越大说明该因子在 u 处解释力越强。
        """
        return self._q_matrix(np.asarray(X), y, np.asarray(coords, dtype=float))

    # ------------------------------------------------------------------ #
    def interaction_surface(self, X1: np.ndarray, X2: np.ndarray,
                            y: np.ndarray, coords: np.ndarray):
        """双因子局地交互探测。

        Returns
        -------
        dict:
            q1, q2, q12 : (n,) ndarray
            q1_plus_q2 : (n,) ndarray
            interaction_type : (n,) ndarray（字符串标签）
            interaction_code : (n,) ndarray（0=独立,1=单因子增强,2=双因子增强,
                               3=非线性增强,4=非线性减弱）
        """
        X1 = np.asarray(X1, dtype=float).ravel()
        X2 = np.asarray(X2, dtype=float).ravel()
        coords = np.asarray(coords, dtype=float)

        q1 = self.q_surface(X1, y, coords)
        q2 = self.q_surface(X2, y, coords)

        # 交叉分层（X1×X2 的笛卡尔分层，层数 L²，上限 25 层）
        s1 = quantile_strata(X1, self.n_strata)
        s2 = quantile_strata(X2, self.n_strata)
        combined = s1 * self.n_strata + s2
        n = y.size
        yarr = np.asarray(y, dtype=float).ravel()
        y2 = yarr * yarr

        b = self._resolve_bandwidth(coords)
        dmat = euclidean_distance(coords)
        W = kernel_weights(dmat, b, self.kernel)
        np.fill_diagonal(W, 1.0)

        num = np.zeros(n)
        for h in range(int(combined.max()) + 1):
            mask = (combined == h).astype(float)
            Wh = W * mask[None, :]
            Nh = Wh.sum(axis=1)
            ybar = (Wh @ yarr) / np.maximum(Nh, 1e-12)
            e2 = (Wh @ y2) / np.maximum(Nh, 1e-12)
            var_h = np.maximum(e2 - ybar ** 2, 0.0)
            num += Nh * var_h

        if self.denominator == "global":
            var_g = float(np.var(yarr))
            q12 = 1.0 - num / (n * var_g)
        else:
            Ntot = W.sum(axis=1)
            ybar_t = (W @ yarr) / np.maximum(Ntot, 1e-12)
            e2_t = (W @ y2) / np.maximum(Ntot, 1e-12)
            var_t = np.maximum(e2_t - ybar_t ** 2, 0.0)
            q12 = 1.0 - num / np.maximum(Ntot * var_t, 1e-12)
        q12 = np.clip(q12, 0.0, 1.0)

        # 交互作用类型判定（地理探测器规则，逐位置）
        codes = np.zeros(n, dtype=int)
        labels = np.empty(n, dtype=object)
        eps = 1e-6
        for i in range(n):
            mn = min(q1[i], q2[i])
            mx = max(q1[i], q2[i])
            s = q1[i] + q2[i]
            if q12[i] < mn - eps:
                codes[i], labels[i] = 4, "nonlinear_weakening"
            elif q12[i] < mx - eps:
                codes[i], labels[i] = 1, "univariate_enhancement"
            elif q12[i] < s - eps:
                codes[i], labels[i] = 2, "bivariate_enhancement"
            elif q12[i] >= s + eps:
                codes[i], labels[i] = 3, "nonlinear_enhancement"
            else:
                codes[i], labels[i] = 0, "independent"

        return {
            "q1": q1, "q2": q2, "q12": q12,
            "q1_plus_q2": q1 + q2,
            "interaction_code": codes,
            "interaction_type": labels,
        }
