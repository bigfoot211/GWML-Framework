# -*- coding: utf-8 -*-
"""
gwml.evaluate
=============
模型精度评价体系（对应论文 2.4 节与表 4）：

    - R² / 调整后 R²          拟合精度
    - RMSE                   拟合精度
    - 残差 Moran's I         残差空间自相关（越小越好，≈0 说明空间结构被捕获）
    - Kappa 系数              分类/格局一致性
    - 空间 K 折交叉验证        防止空间自相关导致的过拟合评估
"""

from __future__ import annotations

import numpy as np

from .spatial_utils import morans_i, spatial_kfold_indices


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def adjusted_r2(y_true, y_pred, n_params: int) -> float:
    """调整后 R²（考虑参数数量惩罚）。"""
    r = r2(y_true, y_pred)
    n = np.asarray(y_true).size
    if n - n_params - 1 <= 0:
        return r
    return float(1.0 - (1.0 - r) * (n - 1) / (n - n_params - 1))


def residual_morans_i(y_true, y_pred, coords, **kwargs) -> float:
    """残差 Moran's I。"""
    resid = np.asarray(y_true, dtype=float).ravel() - \
        np.asarray(y_pred, dtype=float).ravel()
    return morans_i(resid, np.asarray(coords, dtype=float), **kwargs)


def kappa(y_obs, y_pred) -> float:
    """Cohen's Kappa 一致性系数（用于分类结果/格局匹配）。"""
    y_obs = np.asarray(y_obs).ravel()
    y_pred = np.asarray(y_pred).ravel()
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(y_obs, y_pred))


def overall_accuracy(y_obs, y_pred) -> float:
    y_obs = np.asarray(y_obs).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_obs.size == 0:
        return float("nan")
    return float(np.mean(y_obs == y_pred))


def summarize_regression(y_true, y_pred, coords=None,
                         n_params: int | None = None) -> dict:
    """一键汇总回归模型评价指标。

    Returns
    -------
    dict: R², adj_R²(可选), RMSE, 残差Moran's I(需 coords)
    """
    out = {"R2": r2(y_true, y_pred), "RMSE": rmse(y_true, y_pred)}
    if n_params is not None:
        out["adj_R2"] = adjusted_r2(y_true, y_pred, n_params)
    if coords is not None:
        out["residual_Morans_I"] = residual_morans_i(y_true, y_pred, coords)
    return out


# --------------------------------------------------------------------------- #
# 通用空间交叉验证包装
# --------------------------------------------------------------------------- #


def spatial_cv_score(estimator_factory, X, y, coords, n_folds: int = 5,
                     strategy: str = "spatial", verbose: int = 0) -> dict:
    """对"逐位置局部拟合"类模型执行空间 K 折交叉验证。

    Parameters
    ----------
    estimator_factory : callable
        无参工厂函数，返回带 fit(X, y, coords)/predict() 的模型实例。
    X, y, coords : array_like
    n_folds : int
    strategy : str
        'spatial' 或 'random'。
    verbose : int

    Returns
    -------
    dict: fold_rmse(list), mean_rmse, fold_r2(list), mean_r2
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    coords = np.asarray(coords, dtype=float)
    n = len(y)
    fold = spatial_kfold_indices(coords, n_folds, method=strategy)
    fold_rmse, fold_r2 = [], []
    for f in np.unique(fold):
        tr = np.where(fold != f)[0]
        te = np.where(fold == f)[0]
        est = estimator_factory()
        est.fit(X[tr], y[tr], coords[tr])
        pred = est.predict(X[te], coords[te])
        fold_rmse.append(rmse(y[te], pred))
        fold_r2.append(r2(y[te], pred))
        if verbose:
            print(f"  fold {f}: RMSE={fold_rmse[-1]:.4f} R²={fold_r2[-1]:.4f}")
    return {
        "fold_rmse": fold_rmse,
        "mean_rmse": float(np.mean(fold_rmse)),
        "fold_r2": fold_r2,
        "mean_r2": float(np.mean(fold_r2)),
    }
