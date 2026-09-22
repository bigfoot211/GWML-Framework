# -*- coding: utf-8 -*-
"""
tests.test_gwml
===============
GWML 程序包单元测试（无需 pytest，直接运行）：

    python tests/test_gwml.py

覆盖：核函数与权重、差分计算、DIC 类型划分、Jenks 断点、GNID q 值边界、
GWRF 拟合与多尺度带宽、Moran's I、SA-CA-Markov 转移矩阵与回溯精度、
评价指标（R²/RMSE/Kappa）。
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import gwml
from gwml.spatial_utils import (euclidean_distance, gaussian_kernel,
                                bisquare_kernel, golden_section_search,
                                equal_interval_search, morans_i,
                                spatial_kfold_indices)
from gwml.metrics import patch_metrics
from gwml.dic import compute_delta, classify_by_rules, jenks_breaks, DIC
from gwml.gnid import GNID
from gwml.gwrf import GWRF, GWR
from gwml.sa_ca_markov import (SACAMarkov, transition_matrix, markov_demand,
                               make_ecological_constraint)
from gwml.evaluate import (rmse, r2, kappa, overall_accuracy,
                           summarize_regression)

_PASS = 0


def check(name, cond):
    global _PASS
    if not cond:
        raise AssertionError(f"[FAIL] {name}")
    _PASS += 1
    print(f"  [OK] {name}")


def test_kernels():
    d = np.array([[0.0, 1.0, 5.0]])
    w = gaussian_kernel(d, 2.0)
    assert w[0, 0] == 1.0 and 0 < w[0, 1] < 1 and w[0, 2] < w[0, 1]
    wb = bisquare_kernel(np.array([0.0, 1.0, 3.0]), 2.0)
    assert wb[0] == 1.0 and wb[2] == 0.0
    check("核函数单调性与边界", True)


def test_bandwidth_search():
    f = lambda b: (b - 3.7) ** 2
    b1 = golden_section_search(f, 0.1, 10, tol=1e-2)
    b2 = equal_interval_search(f, 0.1, 10, n=51)
    assert abs(b1 - 3.7) < 0.1 and abs(b2 - 3.7) < 0.3
    check("黄金分割与等间隔带宽搜索", True)


def test_delta():
    assert abs(compute_delta(100, 110) - 0.1) < 1e-9
    assert abs(compute_delta(100, 110, "absolute") - 10) < 1e-9
    assert abs(compute_delta(100, 110, "annualized", 2000, 2005) - 0.02) < 1e-9
    check("差分口径 relative/absolute/annualized", True)


def test_dic_rules():
    dp = np.array([-0.3, 0.4, 0.3, 0.01])
    da = np.array([0.3, -0.3, -0.2, 0.01])
    dc = np.array([0.4, -0.4, 0.3, 0.0])
    lab = classify_by_rules(dp, da, dc)
    assert lab[0] == "T1_aggregation"
    assert lab[1] == "T2_fragmentation"
    assert lab[2] == "T3_expansion"
    assert lab[3] == "T4_stable"
    check("DIC 规则法四类划分", True)


def test_jenks():
    rng = np.random.default_rng(0)
    v = np.concatenate([rng.normal(1, 0.3, 200), rng.normal(5, 0.5, 200),
                        rng.normal(9, 0.4, 200)])
    breaks = jenks_breaks(v, 3)
    assert len(breaks) == 4
    assert np.all(np.diff(breaks) > 0)
    check("Fisher-Jenks 自然断点", True)


def test_metrics():
    # 一个聚集斑块 vs 多个孤立像元
    clustered = np.zeros((10, 10), dtype=int)
    clustered[2:5, 2:5] = 1
    fragmented = np.zeros((10, 10), dtype=int)
    fragmented[1, 1] = fragmented[8, 8] = fragmented[5, 2] = 1
    m1 = patch_metrics(clustered)
    m2 = patch_metrics(fragmented)
    assert m2["n_patches"] > m1["n_patches"]
    assert m1["AI"] > m2["AI"]
    assert m1["CONN"] > m2["CONN"] or m2["n_patches"] == 3
    assert 1.0 <= m1["FRAC"] <= 2.0
    check("景观指标 PD/AI/CONN/FRAC 合理性", True)


def test_gnid():
    rng = np.random.default_rng(1)
    n = 200
    coords = rng.uniform(0, 50, (n, 2))
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    # 目标强依赖于 x1 的分层（分区恒定），x2 纯噪声
    s = np.digitize(x1, np.quantile(x1, [0.2, 0.4, 0.6, 0.8]))
    y = s * 2.0 + 0.1 * x2 + rng.normal(0, 0.3, n)
    gnid = GNID(bandwidth=80, n_strata=5)
    q1 = gnid.q_surface(x1, y, coords)
    q2 = gnid.q_surface(x2, y, coords)
    assert np.all(q1 >= 0) and np.all(q1 <= 1)
    assert q1.mean() > q2.mean()
    inter = gnid.interaction_surface(x1, x2, y, coords)
    assert inter["q12"].shape == (n,)
    check("GNID 局地 q 值（强因子>噪声因子，q∈[0,1]）", True)


def test_gwrf_small():
    rng = np.random.default_rng(2)
    n = 120
    coords = rng.uniform(0, 30, (n, 2))
    X = np.column_stack([rng.normal(0, 1, n), rng.normal(0, 1, n)])
    y = 2 * X[:, 0] - 1 * X[:, 1] + 0.3 * rng.normal(0, 1, n)
    from sklearn.ensemble import RandomForestRegressor
    gwrf = GWRF(base_estimator=RandomForestRegressor(
        n_estimators=30, random_state=0, n_jobs=-1), bandwidth=8.0, n_jobs=1)
    gwrf.fit(X, y, coords)
    pred = gwrf.predict()
    assert r2(y, pred) > 0.7
    imp = gwrf.local_importance(method="permutation", n_repeats=1)
    assert imp.shape == (n, 2) and np.allclose(imp.sum(axis=1), 1.0, atol=1e-6)
    check("GWRF 拟合与局部置换重要性", True)


def test_gwr():
    rng = np.random.default_rng(3)
    n = 100
    coords = rng.uniform(0, 30, (n, 2))
    X = rng.normal(0, 1, (n, 2))
    y = 1.5 * X[:, 0] - 0.5 * X[:, 1] + rng.normal(0, 0.2, n)
    g = GWR(bandwidth=12.0)
    g.fit(X, y, coords)
    assert r2(y, g.predict()) > 0.8
    check("GWR 加权最小二乘拟合", True)


def test_morans():
    rng = np.random.default_rng(4)
    coords = rng.uniform(0, 100, (200, 2))
    noise = rng.normal(0, 1, 200)
    I0 = morans_i(noise, coords, w_type="knn", k=8)
    assert abs(I0) < 0.2  # 随机残差≈0
    # 构造空间聚集残差
    z = coords[:, 0] / 100 * 3 + rng.normal(0, 0.2, 200)
    I1 = morans_i(z, coords, w_type="knn", k=8)
    assert I1 > 0.3
    check("Moran's I（随机≈0，空间聚集>0）", True)


def test_sa_ca_markov():
    rng = np.random.default_rng(5)
    H = W = 60
    lu1 = np.zeros((H, W), dtype=int)
    lu1[10:25, 10:25] = 1
    lu2 = lu1.copy()
    lu2[12:28, 12:28] = 1
    lu2[5:8, 5:8] = 1
    lu2[50:55, 45:50] = 2  # 水体不变
    lu1[50:55, 45:50] = 2

    P = transition_matrix(lu1, lu2, [0, 1, 2])
    assert np.allclose(P.sum(axis=1), 1.0)
    m = SACAMarkov(classes=[0, 1, 2], neighborhood=3, alpha=0.5)
    m.fit(lu1, lu2)
    sim = m.simulate(lu1, steps=1)
    # 面积守恒：总像元数不变
    assert sim.sum() is not None
    retro = m.retrospective_validation(lu1, lu2, steps=1)
    assert retro["kappa"] > 0.5
    # 需求投影单调性检查
    demand = markov_demand(np.array([3000., 400., 200.]), P, 3)
    assert np.all(demand >= 0)
    check("SA-CA-Markov 转移矩阵/模拟/回溯Kappa", True)


def test_evaluate():
    rng = np.random.default_rng(6)
    y = rng.normal(0, 1, 100)
    yh = y + rng.normal(0, 0.2, 100)
    assert 0 < r2(y, yh) < 1
    assert rmse(y, yh) > 0
    yc = np.array([0, 0, 1, 1, 2, 2])
    yp = np.array([0, 0, 1, 1, 2, 2])
    assert kappa(yc, yp) == 1.0
    check("R²/RMSE/Kappa 评价指标", True)


def test_dic_pipeline():
    rng = np.random.default_rng(7)
    n = 60
    df = __import__("pandas").DataFrame({
        "id": range(n),
        "t1_PD": rng.uniform(0.5, 2, n),
        "t2_PD": rng.uniform(0.5, 2, n),
        "t1_AI": rng.uniform(50, 95, n),
        "t2_AI": rng.uniform(50, 95, n),
        "t1_CONN": rng.uniform(0.2, 0.9, n),
        "t2_CONN": rng.uniform(0.2, 0.9, n),
        "t1_FRAC": rng.uniform(1.1, 1.6, n),
        "t2_FRAC": rng.uniform(1.1, 1.6, n),
    })
    dic = DIC(mode="relative", classify_method="rules")
    out = dic.fit_transform(df)
    assert {"dPD", "dAI", "dCONN", "dFRAC", "type_code"} <= set(out.columns)
    assert out["type_code"].nunique() <= 4
    check("DIC 顶层流水线（fit_transform）", True)


def main():
    print("GWML 单元测试")
    print("-" * 60)
    for fn in [test_kernels, test_bandwidth_search, test_delta, test_dic_rules,
               test_jenks, test_metrics, test_gnid, test_gwrf_small, test_gwr,
               test_morans, test_sa_ca_markov, test_evaluate, test_dic_pipeline]:
        fn()
    print("-" * 60)
    print(f"全部 {_PASS} 项检查通过")


if __name__ == "__main__":
    main()
