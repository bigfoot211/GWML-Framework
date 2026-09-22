# -*- coding: utf-8 -*-
"""gwml.cli 命令行入口：python -m gwml.cli <demo|dic|gwrf|gnid> ..."""

from __future__ import annotations

import argparse


def _read_csv(path):
    import pandas as pd
    return pd.read_csv(path)


def cmd_demo(args):
    from examples.run_demo import main as demo_main
    demo_main(outdir=args.outdir, n_villages=args.n, seed=args.seed,
              verbose=True, skip_slow=args.skip_slow,
              importance_n=args.importance_n)
    return 0


def cmd_dic(args):
    df = _read_csv(args.csv)
    from gwml.dic import DIC
    dic = DIC(mode=args.mode, t1=args.t1, t2=args.t2,
              classify_method=args.method,
              pd_thr=args.pd_thr, ai_thr=args.ai_thr, conn_thr=args.conn_thr)
    out = dic.fit_transform(df, t1_prefix=args.t1_prefix,
                            t2_prefix=args.t2_prefix)
    out.to_csv(args.output, index=False)
    print(out.groupby("type_code").size())
    print(f"结果已写入 {args.output}")
    return 0


def cmd_gwrf(args):
    df = _read_csv(args.csv)
    feats = args.features or [c for c in df.columns
                              if c not in (args.lon, args.lat, args.target)]
    X = df[feats].values
    y = df[args.target].values
    coords = df[[args.lon, args.lat]].values

    from gwml.gwrf import GWRF, GWR
    from gwml.evaluate import summarize_regression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression

    print(f"样本数 n={len(y)}，变量 p={len(feats)}: {feats}")
    m_gwrf = GWRF(multiscale=args.multiscale, bandwidth_search=args.search,
                  cv_folds=args.cv_folds, verbose=1)
    m_gwrf.fit(X, y, coords)
    print("GWRF 评价:", summarize_regression(y, m_gwrf.predict(), coords,
                                             n_params=len(feats) + 1))
    print("带宽:", m_gwrf.bandwidth_)

    if args.importance:
        imp = m_gwrf.local_importance(method=args.importance)
        import pandas as pd
        pd.DataFrame(imp, columns=feats).to_csv(
            args.outdir + "/gwrf_local_importance.csv", index=False)
        print("局部特征重要性已保存")

    if args.benchmark:
        m_ols = LinearRegression().fit(X, y)
        pred_ols = m_ols.predict(X)
        m_rf = RandomForestRegressor(n_estimators=100, random_state=42)
        m_rf.fit(X, y)
        pred_rf = m_rf.predict(X)
        m_gwr = GWR(kernel="gaussian")
        m_gwr.fit(X, y, coords)
        pred_gwr = m_gwr.predict()
        rows = [
            ("OLS", summarize_regression(y, pred_ols, coords, len(feats) + 1)),
            ("GWR", summarize_regression(y, pred_gwr, coords, len(feats) + 1)),
            ("RF", summarize_regression(y, pred_rf, coords, None)),
            ("GWRF", summarize_regression(y, m_gwrf.predict(), coords,
                                          len(feats) + 1)),
        ]
        import pandas as pd
        pd.DataFrame([{**{"model": k}, **v} for k, v in rows]).to_csv(
            args.outdir + "/model_comparison.csv", index=False)
        print("\n模型对照表（写入 model_comparison.csv）:")
        for k, v in rows:
            print(f"  {k:6s} R²={v['R2']:.4f} RMSE={v['RMSE']:.4f} "
                  f"Moran's I={v.get('residual_Morans_I', float('nan')):.4f}")
    return 0


def cmd_gnid(args):
    df = _read_csv(args.csv)
    if len(args.features) != 2:
        raise SystemExit("gnid 需要恰好两个特征（双因子交互探测）")
    f1, f2 = args.features
    y = df[args.target].values
    coords = df[[args.lon, args.lat]].values

    from gwml.gnid import GNID
    gnid = GNID(bandwidth=args.bandwidth, n_strata=args.n_strata,
                denominator=args.denominator)
    res = gnid.interaction_surface(df[f1].values, df[f2].values, y, coords)
    out = df.copy()
    out["q_" + f1] = res["q1"]
    out["q_" + f2] = res["q2"]
    out["q_12"] = res["q12"]
    out["interaction_type"] = res["interaction_type"]
    out.to_csv(args.output, index=False)
    from collections import Counter
    print(Counter(res["interaction_type"]))
    print(f"结果已写入 {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gwml",
        description="GWML 时空地理加权机器学习框架命令行工具")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo", help="运行端到端演示")
    d.add_argument("--outdir", default="./output", help="输出目录")
    d.add_argument("--n", type=int, default=600, help="合成村庄数量")
    d.add_argument("--seed", type=int, default=42)
    d.add_argument("--skip-slow", action="store_true",
                   help="跳过最耗时的局部重要性")
    d.add_argument("--importance-n", type=int, default=200,
                   help="局部重要性抽样位置数（默认 200）")
    d.set_defaults(func=cmd_demo)

    d = sub.add_parser("dic", help="DIC 差分与演化类型划分")
    d.add_argument("--csv", required=True)
    d.add_argument("--t1-prefix", default="t1_")
    d.add_argument("--t2-prefix", default="t2_")
    d.add_argument("--mode", default="relative",
                   choices=["relative", "absolute", "annualized"])
    d.add_argument("--t1", type=float, default=None)
    d.add_argument("--t2", type=float, default=None)
    d.add_argument("--method", default="rules", choices=["rules", "cluster"])
    d.add_argument("--pd-thr", type=float, default=0.1)
    d.add_argument("--ai-thr", type=float, default=0.05)
    d.add_argument("--conn-thr", type=float, default=0.05)
    d.add_argument("--output", default="./dic_result.csv")
    d.set_defaults(func=cmd_dic)

    d = sub.add_parser("gwrf", help="GWRF 拟合与模型对照")
    d.add_argument("--csv", required=True)
    d.add_argument("--lon", required=True)
    d.add_argument("--lat", required=True)
    d.add_argument("--target", required=True)
    d.add_argument("--features", action="append", default=None)
    d.add_argument("--multiscale", action="store_true")
    d.add_argument("--search", default="golden", choices=["golden", "grid"])
    d.add_argument("--cv-folds", type=int, default=5)
    d.add_argument("--importance", choices=["permutation", "shap"], default=None)
    d.add_argument("--benchmark", action="store_true")
    d.add_argument("--outdir", default="./output")
    d.set_defaults(func=cmd_gwrf)

    d = sub.add_parser("gnid", help="GNID 局地 q 值与交互探测")
    d.add_argument("--csv", required=True)
    d.add_argument("--lon", required=True)
    d.add_argument("--lat", required=True)
    d.add_argument("--target", required=True)
    d.add_argument("--features", action="append", required=True)
    d.add_argument("--bandwidth", type=float, default=None)
    d.add_argument("--n-strata", type=int, default=5)
    d.add_argument("--denominator", default="global",
                   choices=["global", "local"])
    d.add_argument("--output", default="./gnid_result.csv")
    d.set_defaults(func=cmd_gnid)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
