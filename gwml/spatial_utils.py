# -*- coding: utf-8 -*-
"""
gwml.spatial_utils
==================
空间基础工具：欧氏距离、空间核函数、带宽寻优（黄金分割 / 等间隔搜索）、
空间 K 折交叉验证、残差 Moran's I 计算。

对应论文:
    Zhang K, et al. A spatiotemporal geographically weighted machine learning
    framework ... (CEUS 投稿) 之第 3.3 节（GWRF / GNID）所用空间权重与带宽机制。
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# 空间距离与核函数
# --------------------------------------------------------------------------- #


def euclidean_distance(coords: np.ndarray) -> np.ndarray:
    """计算 n×2 坐标矩阵的两两欧氏距离，返回 n×n 对称矩阵。

    Parameters
    ----------
    coords : (n, 2) ndarray
        每个空间单元的地理坐标（x, y）。

    Returns
    -------
    dmat : (n, n) ndarray
        dmat[i, j] = 单元 i 与单元 j 的欧氏距离（对角为 0）。
    """
    coords = np.asarray(coords, dtype=float)
    d = coords[:, None, :] - coords[None, :, :]
    dmat = np.sqrt(np.einsum("ijk,ijk->ij", d, d))
    return dmat


def gaussian_kernel(d: np.ndarray, bandwidth: float) -> np.ndarray:
    """高斯核函数：w = exp(-0.5 * (d / b)^2)。

    论文式 (6)：空间权重函数采用高斯核，b 为带宽。
    """
    d = np.asarray(d, dtype=float)
    b = float(bandwidth)
    if b <= 0:
        raise ValueError("bandwidth 必须为正数")
    return np.exp(-0.5 * (d / b) ** 2)


def bisquare_kernel(d: np.ndarray, bandwidth: float) -> np.ndarray:
    """双平方核函数：w = (1 - (d/b)^2)^2，当 d >= b 时为 0。

    可选核函数，论文默认使用高斯核。
    """
    d = np.asarray(d, dtype=float)
    b = float(bandwidth)
    if b <= 0:
        raise ValueError("bandwidth 必须为正数")
    r = d / b
    r[r >= 1.0] = 1.0
    return np.clip((1.0 - r ** 2) ** 2, 0.0, None)


KERNELS = {
    "gaussian": gaussian_kernel,
    "bisquare": bisquare_kernel,
}


def kernel_weights(dmat: np.ndarray, bandwidth: float,
                   kernel: str = "gaussian") -> np.ndarray:
    """由距离矩阵与带宽生成 n×n 空间权重矩阵（不进行行标准化）。"""
    if kernel not in KERNELS:
        raise ValueError(f"kernel 必须为 {list(KERNELS)} 之一")
    return KERNELS[kernel](dmat, bandwidth)


def combine_per_variable_weights(w_list, mode: str = "mean") -> np.ndarray:
    """合并多变量（多带宽）空间权重。

    论文主张"每个驱动变量拥有独立最优带宽"（MGWR 思想）。在 GWRF 中，
    每个变量的高斯核权重按 `mode` 合并为最终样本权重：

    - 'mean'   : 算术平均，稳健、不会过小（推荐默认）
    - 'product': 乘积，权重随变量数急剧衰减，仅用于强调"全部尺度都邻近"的样本
    """
    w = np.mean(w_list, axis=0) if mode == "mean" else np.prod(w_list, axis=0)
    return w


# --------------------------------------------------------------------------- #
# 带宽寻优
# --------------------------------------------------------------------------- #


def golden_section_search(f, a: float, b: float, tol: float = 1e-3,
                          max_iter: int = 100) -> float:
    """黄金分割搜索：在 [a, b] 内最小化一元函数 f（用于小样本带宽寻优）。

    Parameters
    ----------
    f : callable
        待最小化的函数 f(b) -> float。
    a, b : float
        搜索区间（带宽下限、上限，须 a < b）。
    tol : float
        区间收敛容差。
    max_iter : int
        最大迭代次数。

    Returns
    -------
    x_opt : float
        使 f 最小的带宽近似值。
    """
    gr = (np.sqrt(5.0) - 1.0) / 2.0  # 黄金分割比倒数
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(max_iter):
        if np.abs(b - a) < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)
    return (a + b) / 2.0


def equal_interval_search(f, lo: float, hi: float, n: int = 20) -> float:
    """等间隔（均匀网格）搜索：在 [lo, hi] 内取 n 个候选带宽最小化 f。

    论文指出：大样本时采用等间隔搜索代替黄金分割搜索，降低计算成本。
    """
    if n < 2:
        raise ValueError("n 至少为 2")
    grid = np.linspace(lo, hi, n)
    best_b, best_v = grid[0], np.inf
    for b in grid:
        v = f(b)
        if v < best_v:
            best_b, best_v = b, v
    return float(best_b)


def heuristic_bandwidth_range(dmat: np.ndarray) -> tuple:
    """依据距离矩阵给出带宽搜索的合理区间 [lo, hi]。

    lo = 第 5 百分位非零距离，hi = 第 95 百分位非零距离（覆盖局地~区域尺度）。
    """
    n = dmat.shape[0]
    off = dmat[np.triu_indices(n, k=1)]
    off = off[off > 0]
    if off.size == 0:
        return (1.0, 10.0)
    lo = float(np.percentile(off, 5))
    hi = float(np.percentile(off, 95))
    lo = max(lo, 1e-6)
    hi = max(hi, lo + 1e-6)
    return (lo, hi)


# --------------------------------------------------------------------------- #
# 空间交叉验证
# --------------------------------------------------------------------------- #


def spatial_kfold_indices(coords: np.ndarray, n_folds: int = 5,
                          method: str = "kmeans", random_state: int = 42):
    """生成空间 K 折交叉验证的折标签（KMeans 空间分块或随机划分）。

    Parameters
    ----------
    coords : (n, 2) ndarray
        空间坐标。
    n_folds : int
        折数。
    method : str
        'kmeans'（按坐标聚类分块，训练/测试空间分离，推荐）或 'random'。
    random_state : int

    Returns
    -------
    fold : (n,) int ndarray
        每个样本所属折（0 .. n_folds-1）。同一折内的样本在测试时一起被排除，
        避免空间自相关导致的过拟合评估。
    """
    n = coords.shape[0]
    if method == "random":
        rng = np.random.default_rng(random_state)
        return rng.integers(0, n_folds, size=n)

    try:
        from sklearn.cluster import KMeans
    except ImportError as e:  # pragma: no cover
        raise ImportError("spatial_kfold_indices(method='kmeans') 需要 scikit-learn") from e

    k = min(n_folds, n)
    if n_folds == 1 or n <= n_folds:
        return np.zeros(n, dtype=int)
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    # 坐标标准化后聚类，避免量纲影响
    std = np.std(coords, axis=0)
    std[std == 0] = 1.0
    labels = km.fit_predict(coords / std)
    return labels.astype(int)


# --------------------------------------------------------------------------- #
# 空间自相关
# --------------------------------------------------------------------------- #


def morans_i(residuals: np.ndarray, coords: np.ndarray,
             w_type: str = "inv_dist", k: int = 8,
             distance_cutoff: float | None = None) -> float:
    """残差的全局 Moran's I（行标准化权重）。

    Parameters
    ----------
    residuals : (n,) ndarray
        模型残差。
    coords : (n, 2) ndarray
        空间坐标。
    w_type : str
        'inv_dist'（反距离权重）或 'knn'（k 近邻二元权重）。
    k : int
        w_type='knn' 时的近邻数。
    distance_cutoff : float | None
        'inv_dist' 时仅保留距离 <= cutoff 的邻接（None 表示全部）。

    Returns
    -------
    I : float
        取值通常位于 [-1, 1]，越接近 0 表示残差空间自相关越弱。
    """
    r = np.asarray(residuals, dtype=float)
    n = r.size
    if n < 3:
        return float("nan")
    dmat = euclidean_distance(coords)
    if w_type == "knn":
        W = np.zeros_like(dmat)
        for i in range(n):
            order = np.argsort(dmat[i])[1:k + 1]
            W[i, order] = 1.0
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            W = 1.0 / np.where(dmat > 0, dmat, np.inf)
        np.fill_diagonal(W, 0.0)
        if distance_cutoff is not None:
            W[dmat > distance_cutoff] = 0.0
    rs = W.sum(axis=1)
    rs[rs == 0] = 1.0
    W = W / rs[:, None]  # 行标准化
    z = r - r.mean()
    S0 = W.sum()
    if S0 == 0:
        return float("nan")
    denom = np.sum(z ** 2)
    if denom == 0:
        return 0.0
    I = (n / S0) * (z @ W @ z) / denom
    return float(I)
