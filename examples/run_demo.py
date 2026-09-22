# -*- coding: utf-8 -*-
"""
examples.run_demo
=================
GWML 框架端到端演示：合成数据上依次运行三大模块
（DIC → GNID/GWRF → SA-CA-Markov），并输出模型对照与结果文件。

运行：
    python examples/run_demo.py --outdir ./output --n 600
或：
    python -m gwml.cli demo --outdir ./output --n 600
"""

from __future__ import annotations

import os
import sys

# 保证从任意工作目录运行时都能导入 gwml 包与 examples 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

import numpy as np
import pandas as pd


def run_demo(outdir: str = "./output", n_villages: int = 600, seed: int = 42,
             verbose: bool = True, skip_slow: bool = False,
             importance_n: int = 200):
    """执行完整 GWML 演示流水线。

    skip_slow=True 时跳过最耗时的局部重要性；
    importance_n：局部重要性抽样位置数（越大越全但越慢，默认 200）。
    """
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()

    from examples.make_demo_data import generate_village_table, \
        generate_landuse_rasters
    from gwml.dic import DIC, _TYPE_NAMES
    from gwml.gnid import GNID
    from gwml.gwrf import GWRF, GWR
    from gwml.sa_ca_markov import (SACAMarkov, make_ecological_constraint,
                                   make_balanced_constraint)
    from gwml.evaluate import summarize_regression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression

    log = []
    def p(*a):
        if verbose:
            print(*a)
        log.append(" ".join(str(x) for x in a))

    # ------------------------------------------------------------- #
    # 0. 数据准备
    # ------------------------------------------------------------- #
    p("=" * 72)
    p("GWML 框架端到端演示")
    p("=" * 72)
    p("[0/5] 生成合成数据 ...")
    df = generate_village_table(n_villages, seed)
    rast = generate_landuse_rasters(seed)
    feats = ["elevation", "slope", "road_density", "gdp_per_capita",
             "pop_density", "river_dist", "farmland_ratio"]
    X = df[feats].values
    y = df["y_area_change"].values
    coords = df[["x", "y"]].values

    # ------------------------------------------------------------- #
    # 1. 模块一：DIC 格局量化
    # ------------------------------------------------------------- #
    p(f"[1/5] 模块1 DIC 时序差分景观指数（n={len(df)} 个村庄）...")
    dic = DIC(mode="relative", classify_method="rules")
    df_dic = dic.fit_transform(df, t1_prefix="t1_", t2_prefix="t2_")
    stats = df_dic.groupby("type_code").size().sort_values(ascending=False)
    p("  演化类型统计（type_code | 名称 | 数量）:")
    for code, cnt in stats.items():
        p(f"    {code:<16s} {_TYPE_NAMES.get(code, ''):<12s} {cnt}")
    df_dic.to_csv(os.path.join(outdir, "01_dic_result.csv"), index=False)

    # ------------------------------------------------------------- #
    # 2. 模块二：GWRF + GNID 机理解译
    # ------------------------------------------------------------- #
    p("[2/5] 模块2 GWRF 多尺度带宽寻优（空间5折交叉验证）...")
    base_rf = RandomForestRegressor(n_estimators=60, random_state=seed, n_jobs=-1)
    gwrf = GWRF(base_estimator=base_rf, multiscale=True,
                bandwidth_search="grid", grid_n=6, max_iter=1,
                cv_folds=5, cv_strategy="spatial", max_focal=24,
                n_jobs=-1, verbose=int(verbose))
    t1 = time.time()
    gwrf.fit(X, y, coords)
    bw = np.asarray(gwrf.bandwidth_).ravel()
    p(f"  多尺度带宽(km): "
      f"{dict(zip(feats, [round(float(v), 2) for v in bw]))}")
    p(f"  空间CV RMSE = {gwrf.cv_rmse_:.4f} （耗时 {time.time() - t1:.0f}s）")

    y_hat = gwrf.predict()
    met = summarize_regression(y, y_hat, coords, n_params=len(feats) + 1)
    p(f"  GWRF 拟合: R²={met['R2']:.4f} RMSE={met['RMSE']:.4f} "
      f"残差Moran's I={met['residual_Morans_I']:.4f}")

    # 对照模型
    p("  模型对照（OLS / GWR / RF / GWRF）...")
    m_ols = LinearRegression().fit(X, y)
    pred_ols = m_ols.predict(X)
    m_rf = RandomForestRegressor(n_estimators=60, random_state=seed, n_jobs=-1)
    m_rf.fit(X, y)
    pred_rf = m_rf.predict(X)
    m_gwr = GWR(kernel="gaussian", tol=1e-2)
    m_gwr.fit(X, y, coords)
    pred_gwr = m_gwr.predict()
    rows = [
        ("OLS",  summarize_regression(y, pred_ols, coords, len(feats) + 1)),
        ("GWR",  summarize_regression(y, pred_gwr, coords, len(feats) + 1)),
        ("RF",   summarize_regression(y, pred_rf, coords, None)),
        ("GWRF", met),
    ]
    pd.DataFrame([{**{"model": k}, **v} for k, v in rows]).to_csv(
        os.path.join(outdir, "02_model_comparison.csv"), index=False)
    for k, v in rows:
        p(f"    {k:5s} R²={v['R2']:.4f}  RMSE={v['RMSE']:.4f}  "
          f"Moran's I={v.get('residual_Morans_I', float('nan')):.4f}")

    # 局部特征重要性（置换法，可选跳过；抽样位置 + 增量写盘，防中断丢进度）
    if not skip_slow:
        p("  局部特征重要性（置换法，并行）...")
        t2 = time.time()
        n_total = len(y)
        idx_imp = np.arange(n_total)
        if importance_n is not None and importance_n < n_total:
            rng_i = np.random.default_rng(seed)
            idx_imp = np.sort(rng_i.choice(n_total, size=importance_n,
                                           replace=False))
        imp = gwrf.local_importance(method="permutation", n_repeats=1)
        imp_sub = imp[idx_imp]
        p(f"    完成（耗时 {time.time() - t2:.0f}s，抽样 {len(idx_imp)} 位置），"
          f"各因子全域均值重要性：")
        for k, f in enumerate(feats):
            p(f"      {f:<16s} {imp_sub[:, k].mean():.4f}")
        csv_path = os.path.join(outdir, "03_gwrf_local_importance.csv")
        pd.DataFrame(imp, columns=feats).to_csv(csv_path, index=False)
        meta_path = os.path.join(outdir, "03_importance_sampled_idx.npy")
        np.save(meta_path, idx_imp)

    # GNID 局地 q 值与交互
    p("  GNID 局地 q 值与因子交互探测（gdp × pop_density → 目标）...")
    gnid = GNID(n_strata=5, denominator="global")
    inter = gnid.interaction_surface(df["gdp_per_capita"].values,
                                     df["pop_density"].values, y, coords)
    from collections import Counter
    cnt = Counter(inter["interaction_type"])
    p("  交互类型分布: " + ", ".join(f"{k}={v}" for k, v in cnt.items()))
    out_gnid = df.copy()
    out_gnid["q_gdp"] = inter["q1"]
    out_gnid["q_pop"] = inter["q2"]
    out_gnid["q_interaction"] = inter["q12"]
    out_gnid["interaction_type"] = inter["interaction_type"]
    out_gnid.to_csv(os.path.join(outdir, "04_gnid_interaction.csv"), index=False)

    # ------------------------------------------------------------- #
    # 3. 模块三：SA-CA-Markov 情景模拟
    # ------------------------------------------------------------- #
    p("[3/5] 模块3 SA-CA-Markov 回溯检验与三情景模拟...")
    lu1, lu2 = rast["landuse_t1"], rast["landuse_t2"]
    elevation, slope, gdp_r = rast["elevation"], rast["slope"], rast["gdp"]
    suit = {1: rast["suitability_1"], 0: None, 2: None}

    # 3.1 回溯检验：用 t1→t2 转移矩阵，从 t1 模拟 t2
    model = SACAMarkov(classes=[0, 1, 2], neighborhood=5, alpha=1.0)
    model.fit(lu1, lu2)
    retro = model.retrospective_validation(lu1, lu2, steps=1, suitability=suit)
    p(f"  回溯检验: 总体精度={retro['overall_accuracy']:.4f} "
      f"Kappa={retro['kappa']:.4f}")
    area_obs = retro["area_observed"]
    area_sim = retro["area_simulated"]
    for c, ao, as_, er in zip(model.classes, area_obs, area_sim,
                              retro["area_error_pct"]):
        p(f"    地类{c}: 实测像元={int(ao)} 模拟像元={int(as_)} "
          f"面积误差={er:+.2f}%")

    # 3.2 三情景模拟（从 t2 模拟下一期；情景锚定需求 + 约束面）
    from gwml.sa_ca_markov import markov_demand
    area_t2 = np.array([int((lu2 == c).sum()) for c in model.classes],
                       dtype=float)
    d_bau = markov_demand(area_t2, model.transition_, 1)

    def _anchored_demand(d, growth_scale):
        """情景锚定需求：目标地类扩张规模按情景缩放，差额回补背景类。"""
        d2 = d.copy()
        d2[1] = d[1] * growth_scale
        d2[0] = d[0] + (d[1] - d2[1])
        return d2

    cons_eco = {1: make_ecological_constraint(elevation, slope, 1,
                                              elev_threshold=500,
                                              slope_threshold=22, penalty=0.05),
                0: None, 2: None}
    cons_bal = {1: np.clip(make_balanced_constraint(elevation, slope, gdp_r, 1,
                                                    elev_threshold=650,
                                                    slope_threshold=26,
                                                    eco_penalty=0.1),
                           0, 1), 0: None, 2: None}

    scenarios = {
        "S1_BAU": (None, None),
        "S2_EcoProtect": (cons_eco, _anchored_demand(d_bau, 0.7)),
        "S3_Balanced": (cons_bal, _anchored_demand(d_bau, 0.85)),
    }
    sims = {}
    for name, (cons, dem) in scenarios.items():
        s = model.simulate(lu2, steps=1, suitability=suit, constraints=cons,
                           demand=dem)
        sims[name] = s
        area1 = int((s == 1).sum())
        change = (area1 - int((lu2 == 1).sum())) / \
            max(int((lu2 == 1).sum()), 1) * 100
        p(f"  {name:<14s} 居民点像元={area1} 相对 t2 变化={change:+.1f}%")
        np.save(os.path.join(outdir, f"05_sim_{name}.npy"), s)
    np.save(os.path.join(outdir, "05_landuse_t2.npy"), lu2)

    # 情景指标汇总表
    pd.DataFrame([
        {"scenario": k,
         "settlement_pixels": int((v == 1).sum()),
         "change_pct": round((int((v == 1).sum()) - int((lu2 == 1).sum())) /
                             max(int((lu2 == 1).sum()), 1) * 100, 2)}
        for k, v in sims.items()
    ]).to_csv(os.path.join(outdir, "06_scenario_summary.csv"), index=False)

    # ------------------------------------------------------------- #
    # 4. 汇总
    # ------------------------------------------------------------- #
    p("[4/5] 汇总报告写入 summary.txt")
    with open(os.path.join(outdir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("GWML 框架演示运行摘要\n")
        f.write("=" * 60 + "\n")
        f.write("\n".join(log))
        f.write(f"\n\n总耗时: {time.time() - t0:.1f}s\n")
    p(f"完成！全部结果已写入 {outdir}（总耗时 {time.time() - t0:.1f}s）")
    return outdir


def main(outdir="./output", n_villages=600, seed=42, verbose=True,
         skip_slow=False, importance_n=200):
    run_demo(outdir=outdir, n_villages=n_villages, seed=seed,
             verbose=verbose, skip_slow=skip_slow,
             importance_n=importance_n)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="./output")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-slow", action="store_true")
    ap.add_argument("--importance-n", type=int, default=200)
    args = ap.parse_args()
    run_demo(outdir=args.outdir, n_villages=args.n, seed=args.seed,
             skip_slow=args.skip_slow, importance_n=args.importance_n)
