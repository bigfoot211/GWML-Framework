# -*- coding: utf-8 -*-
"""
gwml
====
时空地理加权机器学习（GWML）框架的 Python 实现。

对应论文：
    Zhang K, Fan Z, Zhang C, Tang Y. A spatiotemporal geographically weighted
    machine learning framework: Exploring driving mechanisms and spatial
    optimization of rural settlement evolution in Hunan, China
    （《Computers, Environment and Urban Systems》投稿）；
    张坤，范正阳，张超正，唐一峰，等. 多地形下湖南省农村居民点人地耦合
    分异及情景优化（《农业工程学报》投稿）。

三大模块（PMP 闭环）：
    DIC                格局量化：时序差分景观指数耦合
    GNID + GWRF        机理解译：局部因子交互探测 + 地理加权随机森林
    SA-CA-Markov       情景预测：机理约束的元胞自动机-马尔可夫模型

典型用法：
    from gwml import GWRF
    model = GWRF(multiscale=True, verbose=1)
    model.fit(X, y, coords)
    y_hat = model.predict()
    importance = model.local_importance()
"""

from .metrics import patch_metrics, metrics_from_tiles, rasterize_polygons
from .dic import (DIC, compute_delta, classify_by_rules, classify_cluster,
                  jenks_breaks, _TYPE_NAMES)
from .gnid import GNID, quantile_strata
from .gwrf import GWRF, GWR
from .sa_ca_markov import (SACAMarkov, transition_matrix, markov_demand,
                           make_ecological_constraint,
                           make_balanced_constraint)
from .evaluate import (rmse, r2, adjusted_r2, kappa, overall_accuracy,
                       morans_i, residual_morans_i, summarize_regression,
                       spatial_cv_score)
from .spatial_utils import (euclidean_distance, gaussian_kernel,
                            bisquare_kernel, kernel_weights,
                            golden_section_search, equal_interval_search,
                            spatial_kfold_indices)

__version__ = "1.0.0"

__all__ = [
    # metrics
    "patch_metrics", "metrics_from_tiles", "rasterize_polygons",
    # DIC
    "DIC", "compute_delta", "classify_by_rules", "classify_cluster",
    "jenks_breaks",
    # GNID / GWRF
    "GNID", "quantile_strata", "GWRF", "GWR",
    # SA-CA-Markov
    "SACAMarkov", "transition_matrix", "markov_demand",
    "make_ecological_constraint", "make_balanced_constraint",
    # evaluate
    "rmse", "r2", "adjusted_r2", "kappa", "overall_accuracy", "morans_i",
    "residual_morans_i", "summarize_regression", "spatial_cv_score",
    # spatial utils
    "euclidean_distance", "gaussian_kernel", "bisquare_kernel",
    "kernel_weights", "golden_section_search", "equal_interval_search",
    "spatial_kfold_indices",
]
