# -*- coding: utf-8 -*-
"""
gwml.gwrf
=========
模块 2（机理解译）之核心模型：地理加权随机森林 GWRF
（Geographically Weighted Random Forest）及对照模型 GWR。

对应论文：
    英文版 3.3.2 节 "From RF to GWRF — The Multiscale Breakthrough"；
    中文版 2.2 节 "多尺度地理加权随机森林 GWRF"。

核心思想
--------
标准随机森林给出全局特征重要性但无空间信息。GWRF 在每个空间位置 u 上，
以空间核函数（高斯核，论文式 (6)）对样本加权，独立训练一棵随机森林，
从而得到"逐位置"的局部模型与局部预测：

    w_ij = exp(-0.5 * (d_ij / b)²)

论文强调的 MGWR 式多尺度突破：每个驱动变量拥有独立最优带宽 b_k，
通过黄金分割搜索（小样本）或等间隔搜索（大样本）迭代寻优；
本实现提供单带宽（classic GWRF）与逐变量带宽（multiscale，坐标下降迭代）
两种模式。局部特征重要性提供置换法（默认）与 SHAP 法（可选）。
"""

from __future__ import annotations

import numpy as np

from .spatial_utils import (
    euclidean_distance, kernel_weights, combine_per_variable_weights,
    golden_section_search, equal_interval_search, heuristic_bandwidth_range,
    spatial_kfold_indices,
)
from .evaluate import rmse

# --------------------------------------------------------------------------- #
# 对照模型：地理加权回归 GWR（单带宽，加权最小二乘）
# --------------------------------------------------------------------------- #


class GWR:
    """地理加权回归（GWR）：每个位置用空间核加权的局部线性回归。

    用于与 GWRF 对照（论文表 4/表 8 中的单带宽 GWR）。

    Parameters
    ----------
    bandwidth : float | None
        空间带宽；None 时用空间交叉验证自动寻优。
    kernel : str
        'gaussian' 或 'bisquare'。
    cv_folds : int
        带宽寻优的折数。
    cv_strategy : str
        'spatial'（默认）或 'random'。
    """

    def __init__(self, bandwidth: float | None = None, kernel: str = "gaussian",
                 cv_folds: int = 5, cv_strategy: str = "spatial",
                 tol: float = 1e-3):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.cv_folds = cv_folds
        self.cv_strategy = cv_strategy
        self.tol = tol
        self.fitted_ = None

    def _predict_with_bandwidth(self, X, y, coords, b, test_idx=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n = X.shape[0]
        Xc = np.column_stack([np.ones(n), X])  # 含截距
        dmat = euclidean_distance(coords)
        W = kernel_weights(dmat, b, self.kernel)
        if test_idx is None:
            test_idx = np.arange(n)
        out = np.empty(len(test_idx))
        Xt = np.linalg.pinv(Xc)
        for k, i in enumerate(test_idx):
            w = W[i]
            WX = Xc * w[:, None]
            # beta = (X' W X)^{-1} X' W y
            A = Xc.T @ WX
            beta = np.linalg.pinv(A) @ (Xc.T @ (w * y))
            out[k] = Xc[i] @ beta
        return out

    def fit(self, X, y, coords):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        coords = np.asarray(coords, dtype=float)
        n = X.shape[0]
        if self.bandwidth is None:
            lo, hi = heuristic_bandwidth_range(euclidean_distance(coords))
            fold = spatial_kfold_indices(coords, self.cv_folds,
                                         method=self.cv_strategy)

            def score(b):
                err = np.empty(n)
                for f in np.unique(fold):
                    te = np.where(fold == f)[0]
                    pred = self._predict_with_bandwidth(X, y, coords, b, te)
                    err[te] = y[te] - pred
                return _fast_rmse(err)

            b_opt = golden_section_search(score, lo, hi, tol=self.tol)
            self.bandwidth = b_opt
            self.cv_score_ = score(b_opt)
        else:
            self.cv_score_ = None
        self.fitted_ = self._predict_with_bandwidth(X, y, coords, self.bandwidth)
        return self

    def predict(self, X=None, y=None, coords=None):
        if self.fitted_ is None:
            raise RuntimeError("请先调用 fit()")
        return self.fitted_


def _fast_rmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(err ** 2)))


# --------------------------------------------------------------------------- #
# GWRF：地理加权随机森林
# --------------------------------------------------------------------------- #


class GWRF:
    """地理加权随机森林（GWRF），支持单带宽与多尺度（逐变量带宽）模式。

    Parameters
    ----------
    base_estimator : estimator | None
        任意支持 sample_weight 的回归器，默认
        RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)。
    kernel : str
        'gaussian'（论文默认）或 'bisquare'。
    bandwidth : float | array_like | None
        空间带宽。None 时自动寻优（全局单带宽）；数组（p,）时指定逐变量带宽；
        结合 multiscale=True 自动逐变量寻优。
    multiscale : bool
        True 时逐变量独立寻优带宽（坐标下降迭代，MGWR 思想）。
    bandwidth_search : str
        'golden'（黄金分割，小样本默认）或 'grid'（等间隔，大样本）。
    grid_n : int
        'grid' 时的候选带宽数。
    max_iter : int
        multiscale 坐标下降的最大轮次。
    weight_combine : str
        multiscale 时逐变量权重的合并方式：'mean'（默认）或 'product'。
    cv_folds : int
        带宽寻优与评估的交叉验证折数（论文采用 5 折空间交叉验证）。
    cv_strategy : str
        'spatial'（默认，按坐标 KMeans 分块）或 'random'。
    bandwidth_range : tuple | None
        带宽搜索区间 [lo, hi]；None 时由距离矩阵自动估计。
    max_focal : int | None
        带宽寻优时"每折"最多参与评分的焦点位置数（总抽样上限 = max_focal
        × 折数；例如 5 折、max_focal=24 时每轮 CV 拟合 120 个局部模型）。
        None 表示使用全部位置；大样本建议设置（如 24~40）。
    n_jobs : int
        局部模型并行拟合的进程/线程数（-1 使用全部 CPU）。
    verbose : int
        输出日志级别（0 静默，1 简要，2 详细）。
    random_state : int
    """

    def __init__(self, base_estimator=None, kernel: str = "gaussian",
                 bandwidth=None, multiscale: bool = False,
                 bandwidth_search: str = "golden", grid_n: int = 20,
                 max_iter: int = 10, weight_combine: str = "mean",
                 cv_folds: int = 5, cv_strategy: str = "spatial",
                 bandwidth_range=None, max_focal: int | None = None,
                 n_jobs: int = -1, verbose: int = 0,
                 random_state: int = 42):
        if base_estimator is None:
            from sklearn.ensemble import RandomForestRegressor
            base_estimator = RandomForestRegressor(
                n_estimators=100, random_state=random_state, n_jobs=-1)
        self.base_estimator = base_estimator
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.multiscale = multiscale
        self.bandwidth_search = bandwidth_search
        self.grid_n = grid_n
        self.max_iter = max_iter
        self.weight_combine = weight_combine
        self.cv_folds = cv_folds
        self.cv_strategy = cv_strategy
        self.bandwidth_range = bandwidth_range
        self.max_focal = max_focal
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        self.bandwidth_ = None          # 最终带宽（标量或 (p,) 数组）
        self.fitted_ = None             # (n,) 拟合/预测值
        self.local_importance_ = None   # (n, p) 局部特征重要性
        self.cv_rmse_ = None
        self.X_, self.y_, self.coords_ = None, None, None

    # ------------------------------------------------------------------ #
    def _fit_locations(self, X, y, weights, idx=None, n_jobs: int | None = None):
        """对指定位置 idx（默认全部）用样本权重拟合局部 RF 并预测（并行）。"""
        if idx is None:
            idx = np.arange(X.shape[0])
        idx = np.asarray(idx, dtype=int)

        def _one(i):
            m = self._clone_estimator()
            # 外层并行时避免嵌套线程池过订阅
            if hasattr(m, "n_jobs") and self.n_jobs != 1:
                m.n_jobs = 1
            m.fit(X, y, sample_weight=weights[i])
            return m.predict(X[i:i + 1])[0]

        nj = self.n_jobs if n_jobs is None else n_jobs
        if nj != 1 and len(idx) > 1:
            from sklearn.utils.parallel import Parallel, delayed
            out = Parallel(n_jobs=nj, prefer="threads")(
                delayed(_one)(int(i)) for i in idx)
        else:
            out = [_one(int(i)) for i in idx]
        return np.asarray(out, dtype=float)

    def _clone_estimator(self):
        from sklearn.base import clone
        return clone(self.base_estimator)

    # ------------------------------------------------------------------ #
    def _weights_for_bandwidth(self, dmat, b):
        """由带宽（标量或逐变量数组）生成 n×n 权重矩阵。"""
        if np.isscalar(b):
            return kernel_weights(dmat, float(b), self.kernel)
        b = np.asarray(b, dtype=float).ravel()
        if b.size == 1:
            return kernel_weights(dmat, float(b[0]), self.kernel)
        if b.size != self.p_:
            raise ValueError("逐变量带宽长度必须等于变量数")
        w_list = [kernel_weights(dmat, bk, self.kernel) for bk in b]
        return combine_per_variable_weights(w_list, self.weight_combine)

    # ------------------------------------------------------------------ #
    def _cv_rmse(self, X, y, dmat, fold, b, rng=None):
        """给定带宽 b，空间交叉验证 RMSE（严格留折评估）。

        - 同一折内的样本互不参与训练（test 折样本在训练权重中置 0，
          避免 RF 记忆焦点样本导致 CV 失真）；
        - max_focal 表示"每折最多参与评分的焦点位置数"（总抽样上限 =
          max_focal × 折数），用于加速大样本的带宽寻优。
        """
        W = self._weights_for_bandwidth(dmat, b)
        rng = rng or np.random.default_rng(0)
        err = np.full(len(y), np.nan)
        per_fold = (max(1, self.max_focal) if self.max_focal is not None
                    else None)
        for f in np.unique(fold):
            te = np.where(fold == f)[0]
            if per_fold is not None and len(te) > per_fold:
                te = rng.choice(te, size=per_fold, replace=False)
            Wt = W.copy()
            Wt[np.ix_(te, te)] = 0.0  # 排除同折样本与自身
            pred = self._fit_locations(X, y, Wt, te)
            err[te] = y[te] - pred
        return float(np.sqrt(np.nanmean(err ** 2)))

    # ------------------------------------------------------------------ #
    def _search_bandwidth(self, X, y, dmat, fold):
        lo, hi = self.bandwidth_range or heuristic_bandwidth_range(dmat)
        rng = np.random.default_rng(self.random_state)
        if self.bandwidth_search == "grid":
            return equal_interval_search(
                lambda b: self._cv_rmse(X, y, dmat, fold, b, rng), lo, hi,
                self.grid_n)
        return golden_section_search(
            lambda b: self._cv_rmse(X, y, dmat, fold, b, rng), lo, hi,
            tol=1e-3)

    # ------------------------------------------------------------------ #
    def fit(self, X, y, coords):
        """拟合 GWRF。

        Parameters
        ----------
        X : (n, p) array_like
            驱动因子矩阵。
        y : (n,) array_like
            目标变量。
        coords : (n, 2) array_like
            空间坐标。
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        coords = np.asarray(coords, dtype=float)
        n, p = X.shape
        self.p_ = p
        self.X_, self.y_, self.coords_ = X, y, coords

        dmat = euclidean_distance(coords)
        fold = spatial_kfold_indices(coords, self.cv_folds,
                                     method=self.cv_strategy)

        if self.multiscale:
            b = np.full(p, np.nan)
            # 初始：全局最优单带宽
            if np.isscalar(self.bandwidth) or self.bandwidth is None:
                b0 = (self.bandwidth if self.bandwidth is not None
                      else self._search_bandwidth(X, y, dmat, fold))
                b[:] = b0
            else:
                b = np.asarray(self.bandwidth, dtype=float).ravel().copy()
            # 坐标下降：逐变量寻优（每次以其余变量带宽固定为前提）
            for it in range(self.max_iter):
                b_new = b.copy()
                for k in range(p):
                    def score(bk, b_fix=b_new, kk=k):
                        cand = b_fix.copy()
                        cand[kk] = bk
                        return self._cv_rmse(X, y, dmat, fold, cand,
                                             np.random.default_rng(
                                                 self.random_state + kk))
                    lo, hi = self.bandwidth_range or heuristic_bandwidth_range(dmat)
                    if self.bandwidth_search == "grid":
                        b_new[k] = equal_interval_search(score, lo, hi, self.grid_n)
                    else:
                        b_new[k] = golden_section_search(score, lo, hi, tol=1e-3)
                if np.max(np.abs(np.log10(b_new / np.maximum(b, 1e-12)))) < 0.05:
                    b = b_new
                    if self.verbose:
                        print(f"[GWRF] 多尺度带宽迭代 {it + 1} 收敛")
                    break
                b = b_new
            self.bandwidth_ = b
            W = self._weights_for_bandwidth(dmat, b)
            np.fill_diagonal(W, 0.0)   # 拟合值评估时排除自身
            self.cv_rmse_ = self._cv_rmse(X, y, dmat, fold, b)
        else:
            if self.bandwidth is None:
                b_opt = self._search_bandwidth(X, y, dmat, fold)
                self.cv_rmse_ = self._cv_rmse(X, y, dmat, fold, b_opt)
            else:
                b_opt = float(np.asarray(self.bandwidth).ravel()[0])
                self.cv_rmse_ = self._cv_rmse(X, y, dmat, fold, b_opt)
            self.bandwidth_ = b_opt
            W = kernel_weights(dmat, b_opt, self.kernel)
            np.fill_diagonal(W, 0.0)   # 拟合值评估时排除自身

        if self.verbose:
            print(f"[GWRF] 带宽 = {np.round(self.bandwidth_, 3)}")
            print(f"[GWRF] 空间CV RMSE = {self.cv_rmse_:.4f}")

        self.fitted_ = self._fit_locations(X, y, W, np.arange(n))
        return self

    # ------------------------------------------------------------------ #
    def predict(self, X=None, coords=None):
        """返回逐位置拟合/预测值（GWRF 的预测即逐位置局部模型预测）。"""
        if self.fitted_ is None:
            raise RuntimeError("请先调用 fit()")
        return self.fitted_

    # ------------------------------------------------------------------ #
    def local_importance(self, method: str = "permutation", n_repeats: int = 1):
        """计算逐位置局部特征重要性 (n, p)。

        Parameters
        ----------
        method : str
            'permutation'（默认，置换重要性：局部模型下置换某变量后
            加权 MSE 损失增量）或 'shap'（需要安装可选依赖 shap，对每个
            局部模型计算焦点样本的 |SHAP| 值作为重要性）。
        n_repeats : int
            置换次数（默认 1，越大越稳定但越慢）。

        Returns
        -------
        importance : (n, p) ndarray，行和为 1 的归一化重要性。
        """
        if self.fitted_ is None:
            raise RuntimeError("请先调用 fit()")
        X, y = self.X_, self.y_
        n, p = X.shape
        dmat = euclidean_distance(self.coords_)
        W = self._weights_for_bandwidth(dmat, self.bandwidth_)

        if method == "shap":
            try:
                import shap  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    "SHAP 法需要可选依赖：pip install shap") from e

            def _shap_loc(i):
                m = self._clone_estimator()
                m.fit(X, y, sample_weight=W[i])
                ex = shap.TreeExplainer(m)
                sv = ex.shap_values(X[i:i + 1])
                return np.abs(np.asarray(sv)[0])

            if self.n_jobs != 1 and n > 1:
                from sklearn.utils.parallel import Parallel, delayed
                rows = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                    delayed(_shap_loc)(int(i)) for i in range(n))
            else:
                rows = [_shap_loc(int(i)) for i in range(n)]
            imp = np.asarray(rows, dtype=float)
            imp = imp / np.maximum(imp.sum(axis=1, keepdims=True), 1e-12)
            self.local_importance_ = imp
            return imp

        # 置换法（并行于位置维度；n_repeats 控制稳定性与耗时）
        rng = np.random.default_rng(self.random_state)

        def _loc_perm(i):
            m = self._clone_estimator()
            m.fit(X, y, sample_weight=W[i])
            w = W[i] / max(W[i].sum(), 1e-12)
            base = np.sum(w * (y - m.predict(X)) ** 2)
            gains = np.zeros(p)
            for k in range(p):
                loss = 0.0
                for r in range(n_repeats):
                    Xp = X.copy()
                    Xp[:, k] = rng.permutation(X[:, k])
                    m2 = self._clone_estimator()
                    m2.fit(Xp, y, sample_weight=W[i])
                    loss += np.sum(w * (y - m2.predict(Xp)) ** 2)
                gains[k] = max(loss / n_repeats - base, 0.0)
            return gains

        if self.n_jobs != 1 and n > 1:
            from sklearn.utils.parallel import Parallel, delayed
            rows = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(_loc_perm)(int(i)) for i in range(n))
        else:
            rows = [_loc_perm(int(i)) for i in range(n)]
        imp = np.asarray(rows, dtype=float)
        imp = imp / np.maximum(imp.sum(axis=1, keepdims=True), 1e-12)
        self.local_importance_ = imp
        return imp
