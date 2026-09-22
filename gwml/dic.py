# -*- coding: utf-8 -*-
"""
gwml.dic
========
模块 1：时序差分景观指数耦合方法（DIC, Delta-Index Coupling）——格局量化。

对应论文：
    英文版 3.2 节 "Module 1: Delta-Index Coupling (DIC) — Pattern Quantification"；
    中文版 2.1 节 "时序差分景观指数耦合方法（DIC）"。

DIC 通过耦合传统景观指标与其时间变化量，刻画居民点/地类"轨迹"而非"状态"：

    ΔPD   斑块密度变化
    ΔAI   集聚指数变化
    ΔCONN 连通度变化
    ΔFRAC 平均分维数变化

四个差分指标构成四维特征向量，按论文 Table 3 的规则划分为四类演化类型：
    T1 集聚强化型  Aggregation-Intensive   (ΔPD ↓, ΔAI ↑, ΔCONN ↑)
    T2 破碎主导型  Fragmentation-Dominant  (ΔPD ↑, ΔAI ↓, ΔCONN ↓)
    T3 扩张驱动型  Expansion-Driven        (ΔPD ↑, ΔAI ↓, ΔCONN ↑)
    T4 稳定均衡型  Stable-Equilibrium      (各指标 ≈ 0)

也提供基于聚类的替代划分（论文提到使用 Jenks 自然断点聚类），
以及 Fisher-Jenks 自然断点的一维实现。
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# 差分计算
# --------------------------------------------------------------------------- #


def compute_delta(x1, x2, mode: str = "relative", t1=None, t2=None):
    """计算时序变化量 ΔX。

    Parameters
    ----------
    x1, x2 : array_like
        时点 t1、t2 的指标值。
    mode : str
        - 'relative' : (x2 - x1) / |x1|，相对变化率（默认，论文"变化率"口径）
        - 'absolute' : x2 - x1，绝对变化
        - 'annualized': (x2 - x1) / |x1| / (t2 - t1)，年均相对变化率
    t1, t2 : float
        mode='annualized' 时必填的年份。

    Returns
    -------
    ndarray（与输入同形状）。x1 为 0 时相对变化率记为 0。
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    if mode == "absolute":
        return x2 - x1
    denom = np.abs(x1)
    out = np.divide(x2 - x1, denom, out=np.zeros_like(x1, dtype=float),
                    where=denom > 0)
    if mode == "annualized":
        if t1 is None or t2 is None:
            raise ValueError("mode='annualized' 需要提供 t1 与 t2")
        dt = float(t2) - float(t1)
        if dt == 0:
            raise ValueError("t2 必须大于 t1")
        out = out / dt
    elif mode != "relative":
        raise ValueError("mode 必须为 'relative' / 'absolute' / 'annualized'")
    return out


# --------------------------------------------------------------------------- #
# 演化类型划分（规则法 + 聚类法）
# --------------------------------------------------------------------------- #

# 论文 Table 3 的符号模板（1=上升, -1=下降, 0=持平）
_TYPE_TEMPLATES = {
    "T1_aggregation":   {"PD": -1, "AI": 1, "CONN": 1},   # 集聚强化
    "T2_fragmentation": {"PD": 1,  "AI": -1, "CONN": -1},  # 破碎主导
    "T3_expansion":     {"PD": 1,  "AI": -1, "CONN": 1},   # 扩张驱动
    "T4_stable":        {"PD": 0,  "AI": 0, "CONN": 0},    # 稳定均衡
}

# 展示用中文名
_TYPE_NAMES = {
    "T1_aggregation": "T1 集聚强化型",
    "T2_fragmentation": "T2 破碎主导型",
    "T3_expansion": "T3 扩张驱动型",
    "T4_stable": "T4 稳定均衡型",
}


def classify_by_rules(delta_pd, delta_ai, delta_conn,
                      pd_thr: float = 0.1, ai_thr: float = 0.05,
                      conn_thr: float = 0.05,
                      strict: bool = False) -> np.ndarray:
    """按论文 Table 3 的符号规则划分演化类型。

    规则（阈值用于判断"显著上升/下降/持平"）：
        T1 集聚强化：ΔPD ≤ -pd_thr 且 ΔAI ≥ ai_thr 且 ΔCONN ≥ conn_thr
        T2 破碎主导：ΔPD ≥ pd_thr 且 ΔAI ≤ -ai_thr 且 ΔCONN ≤ -conn_thr
        T3 扩张驱动：ΔPD ≥ pd_thr 且 ΔAI ≤ -ai_thr 且 ΔCONN ≥ conn_thr
        T4 稳定均衡：|ΔPD| ≤ pd_thr 且 |ΔAI| ≤ ai_thr 且 |ΔCONN| ≤ conn_thr

    Parameters
    ----------
    delta_pd, delta_ai, delta_conn : array_like
        三个差分指标（相对变化率）。
    pd_thr, ai_thr, conn_thr : float
        显著性阈值。
    strict : bool
        True 时仅按上述规则的完全命中划分；False 时对未完全命中的样本
        按"符号模式与模板的加权距离最小"进行最近类型匹配（推荐）。

    Returns
    -------
    labels : (n,) ndarray of str，取值为 _TYPE_TEMPLATES 的键。
    """
    dp = np.asarray(delta_pd, dtype=float)
    da = np.asarray(delta_ai, dtype=float)
    dc = np.asarray(delta_conn, dtype=float)
    n = dp.size
    lab = np.empty(n, dtype=object)

    for i in range(n):
        s_pd = 1 if dp[i] > pd_thr else (-1 if dp[i] < -pd_thr else 0)
        s_ai = 1 if da[i] > ai_thr else (-1 if da[i] < -ai_thr else 0)
        s_conn = 1 if dc[i] > conn_thr else (-1 if dc[i] < -conn_thr else 0)
        sig = {"PD": s_pd, "AI": s_ai, "CONN": s_conn}

        if strict:
            # 完全命中
            hit = None
            for t, tmpl in _TYPE_TEMPLATES.items():
                if all(sig[k] == tmpl[k] for k in sig):
                    hit = t
                    break
            lab[i] = hit if hit else "T4_stable"
        else:
            # 最近模板匹配（加权距离，PD 权重 1.0，AI/CONN 权重 0.8）
            weights = {"PD": 1.0, "AI": 0.8, "CONN": 0.8}
            best, best_d = None, np.inf
            for t, tmpl in _TYPE_TEMPLATES.items():
                d = sum(weights[k] * abs(sig[k] - tmpl[k]) for k in sig)
                if d < best_d:
                    best, best_d = t, d
            lab[i] = best
    return lab


# --------------------------------------------------------------------------- #
# Fisher-Jenks 自然断点
# --------------------------------------------------------------------------- #


def jenks_breaks(values: np.ndarray, n_classes: int = 4) -> np.ndarray:
    """Fisher-Jenks 自然断点（一维，O(k n²)）。

    Parameters
    ----------
    values : (n,) ndarray
        待分类的连续值。
    n_classes : int
        类别数（断点将数据分成 n_classes 组）。

    Returns
    -------
    breaks : (n_classes + 1,) ndarray
        断点数组（含最小、最大值），第 i 组为 [breaks[i], breaks[i+1])。
    """
    v = np.sort(np.asarray(values, dtype=float))
    n = v.size
    if n < n_classes:
        raise ValueError("样本数少于类别数")
    if n_classes < 2:
        return np.array([v.min(), v.max()])

    # 前缀和加速
    cum = np.concatenate([[0.0], np.cumsum(v)])
    cum2 = np.concatenate([[0.0], np.cumsum(v ** 2)])

    def ssw(i, j):
        """区间 [i, j)（i<j）的组内离差平方和。"""
        cnt = j - i
        s = cum[j] - cum[i]
        s2 = cum2[j] - cum2[i]
        return s2 - s * s / cnt

    INF = np.inf
    # ssw_matrix[k][i] = 前 i 个元素分成 k 组的最小 SSW
    ssw_mat = np.full((n_classes, n + 1), INF)
    # partition[k][i] = 达到该最小值的最后一组分界位置
    part = np.zeros((n_classes, n + 1), dtype=int)

    for i in range(1, n + 1):
        ssw_mat[0, i] = ssw(0, i)
        part[0, i] = 0

    for k in range(1, n_classes):
        for i in range(k + 1, n + 1):
            for j in range(k, i):
                val = ssw_mat[k - 1, j] + ssw(j, i)
                if val < ssw_mat[k, i]:
                    ssw_mat[k, i] = val
                    part[k, i] = j

    # 回溯分界点
    breaks = [n]
    k, i = n_classes - 1, n
    while k >= 0:
        j = part[k, i]
        breaks.append(j)
        i = j
        k -= 1
    breaks = sorted(set(breaks))
    break_values = [v[min(b, n - 1)] for b in breaks]
    return np.asarray(sorted(break_values))


def classify_cluster(delta_matrix: np.ndarray, n_clusters: int = 4,
                     method: str = "kmeans", random_state: int = 42):
    """基于聚类（kmeans / jenks-主成分）的演化类型划分。

    Parameters
    ----------
    delta_matrix : (n, 4) ndarray
        四维差分指标（列顺序：PD, AI, CONN, FRAC）。
    n_clusters : int
        类别数（论文为 4）。
    method : str
        'kmeans'（标准化后 KMeans，默认）或 'jenks'（对第一主成分做 Jenks）。
    random_state : int

    Returns
    -------
    labels : (n,) ndarray of int（0..n_clusters-1）
    centers : (n_clusters, 4) ndarray 聚类中心（标准化空间）
    """
    X = np.asarray(delta_matrix, dtype=float)
    if X.ndim != 2:
        raise ValueError("delta_matrix 必须为二维数组")
    std = X.std(axis=0)
    std[std == 0] = 1.0
    Xs = (X - X.mean(axis=0)) / std

    if method == "jenks":
        from numpy.linalg import eigh
        cov = np.cov(Xs.T)
        w, v = eigh(cov)
        pc1 = Xs @ v[:, -1]
        breaks = jenks_breaks(pc1, n_clusters)
        labels = np.digitize(pc1, breaks[1:-1], right=False)
        centers = np.vstack([Xs[labels == c].mean(axis=0)
                             for c in range(n_clusters)])
        return labels, centers

    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = km.fit_predict(Xs)
    return labels, km.cluster_centers_


# --------------------------------------------------------------------------- #
# 顶层 DIC 分析类
# --------------------------------------------------------------------------- #


class DIC:
    """DIC 格局量化分析器：输入两时点景观指标表，输出差分与演化类型。

    Parameters
    ----------
    mode : str
        差分口径（'relative' / 'absolute' / 'annualized'）。
    t1, t2 : float | None
        年份（annualized 时必填）。
    classify_method : str
        'rules'（论文 Table 3 规则法，默认）或 'cluster'（聚类法）。
    pd_thr, ai_thr, conn_thr : float
        规则法的显著性阈值。
    strict : bool
        规则法是否严格完全命中。
    n_clusters : int
        聚类法的类别数（默认 4）。
    """

    def __init__(self, mode: str = "relative", t1=None, t2=None,
                 classify_method: str = "rules",
                 pd_thr: float = 0.1, ai_thr: float = 0.05,
                 conn_thr: float = 0.05, strict: bool = False,
                 n_clusters: int = 4, random_state: int = 42):
        self.mode = mode
        self.t1, self.t2 = t1, t2
        self.classify_method = classify_method
        self.pd_thr, self.ai_thr, self.conn_thr = pd_thr, ai_thr, conn_thr
        self.strict = strict
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit_transform(self, metrics_df, t1_prefix: str = "t1_",
                      t2_prefix: str = "t2_",
                      metrics: tuple = ("PD", "AI", "CONN", "FRAC")):
        """由指标表计算差分并划分演化类型。

        Parameters
        ----------
        metrics_df : pandas.DataFrame
            每个单元一行；需包含两时点的指标列，如 t1_PD, t2_PD ...
        t1_prefix, t2_prefix : str
            两时点列前缀。
        metrics : tuple
            参与计算的指标名。

        Returns
        -------
        DataFrame：原表 + delta 列（ΔPD, ΔAI, ΔCONN, ΔFRAC）+ type 列
        """
        import pandas as pd
        df = metrics_df.copy()
        for m in metrics:
            c1, c2 = f"{t1_prefix}{m}", f"{t2_prefix}{m}"
            if c1 not in df.columns or c2 not in df.columns:
                raise KeyError(f"缺少列 {c1} 或 {c2}")
            df[f"d{m}"] = compute_delta(df[c1].values, df[c2].values,
                                        mode=self.mode, t1=self.t1, t2=self.t2)

        if self.classify_method == "rules":
            labels = classify_by_rules(
                df["dPD"].values, df["dAI"].values, df["dCONN"].values,
                pd_thr=self.pd_thr, ai_thr=self.ai_thr,
                conn_thr=self.conn_thr, strict=self.strict)
            df["type"] = [f"{lbl}|{_TYPE_NAMES[lbl]}" for lbl in labels]
            df["type_code"] = labels
        else:
            mat = df[[f"d{m}" for m in metrics]].values
            labels, centers = classify_cluster(
                mat, n_clusters=self.n_clusters, random_state=self.random_state)
            df["type_code"] = labels
            df["type"] = [f"cluster_{c}" for c in labels]
            self.cluster_centers_ = centers
        self.labels_ = df["type_code"].values
        self.deltas_ = df[[f"d{m}" for m in metrics]].values
        return df
