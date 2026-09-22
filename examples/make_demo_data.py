# -*- coding: utf-8 -*-
"""
examples.make_demo_data
=======================
生成 GWML 演示用的合成数据（模仿湖南省 1990/2005/2020 三期农村居民点研究）：

1. 村庄级样本表（village_table.csv）：
   - id, x, y                     空间单元（村庄）坐标（单位 km）
   - 驱动因子：elevation(高程m), slope(坡度°), road_density(道路密度),
     gdp_per_capita(人均GDP, 千元), pop_density(人口密度, 人/km²),
     river_dist(距河流距离, km), farmland_ratio(耕地占比)
   - target y：居民点面积变化率（%），由"空间非平稳 + 非线性"的因子响应构造
   - 两时点景观指标：t1_PD, t1_AI, t1_CONN, t1_FRAC, t2_*（由每个村庄的
     小栅格模拟斑块形态后计算）

2. 土地利用栅格（SA-CA-Markov 用）：
   - landuse_t1.npy / landuse_t2.npy：150×150 栅格，0=非居民点, 1=居民点,
     2=水体（不变）
   - elevation.npy / slope.npy / gdp.npy：情景约束面
   - suitability_1.npy：GWRF 风格的地类 1（居民点）适宜性表面

说明：所有数据均为演示用途的合成数据，不代表真实湖南省数据。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd


def _smoothed_field(shape, rng, scale=1.0, octaves=3):
    """多层叠加的平滑随机场（模拟地形/经济等空间连续因子）。"""
    H, W = shape
    out = np.zeros(shape)
    for o in range(1, octaves + 1):
        k = 2 ** o
        coarse = rng.standard_normal((max(H // k, 2), max(W // k, 2)))
        from scipy.ndimage import zoom
        up = zoom(coarse, (H / coarse.shape[0], W / coarse.shape[1]), order=1)
        out += up / o
    out = (out - out.mean()) / (out.std() + 1e-12) * scale
    return out


def _village_tile(pattern: str, rng: np.random.Generator, size: int = 12):
    """生成村庄小栅格斑块形态模板（确定性放置，保证四类演化信号清晰）。

    pattern: 'stable' | 'expanding' | 'aggregating' | 'fragmenting'
    连通度 CONN 采用图论连通（conn_threshold=2 像元）。
    """
    a = np.zeros((size, size), dtype=np.int8)

    def cluster(arr, cy, cx, r):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                yy, xx = cy + dy, cx + dx
                if 0 <= yy < size and 0 <= xx < size and (abs(dy) + abs(dx)) <= r:
                    arr[yy, xx] = 1
        return arr

    def singles(arr, pts):
        for (y, x) in pts:
            arr[y, x] = 1
        return arr

    if pattern == "stable":
        a1 = cluster(np.zeros((size, size), dtype=np.int8), 3, 3, 2)
        a1 = cluster(a1, 8, 8, 2)
        return a1.copy(), a1.copy()
    if pattern == "expanding":
        # t1：两个分离小簇（CONN 低）；t2：大斑块+卫星簇+多个零星像元
        #    → PD↑、AI↓、CONN↑（扩张驱动 T3 信号）
        a1 = cluster(np.zeros((size, size), dtype=np.int8), 2, 2, 1)
        a1 = cluster(a1, 10, 10, 1)
        a2 = cluster(np.zeros((size, size), dtype=np.int8), 5, 5, 2)
        a2 = cluster(a2, 7, 5, 1)          # 卫星簇（质心距离 2，阈值内连通）
        a2 = singles(a2, [(1, 1), (10, 10), (1, 10), (10, 1)])
        return a1, a2
    if pattern == "aggregating":
        # t1：三个分散小簇；t2：一个紧凑大斑块 → PD↓、AI↑、CONN↑
        a1 = cluster(np.zeros((size, size), dtype=np.int8), 2, 2, 1)
        a1 = cluster(a1, 9, 2, 1)
        a1 = cluster(a1, 2, 9, 1)
        a2 = cluster(np.zeros((size, size), dtype=np.int8), 5, 5, 3)
        return a1, a2
    if pattern == "fragmenting":
        # t1：一个紧凑大斑块；t2：四个分散零星像元 → PD↑、AI↓、CONN↓
        a1 = cluster(np.zeros((size, size), dtype=np.int8), 5, 5, 3)
        a2 = singles(np.zeros((size, size), dtype=np.int8),
                     [(2, 2), (9, 2), (2, 9), (9, 9)])
        return a1, a2
    raise ValueError(pattern)


def generate_village_table(n_villages: int = 900, seed: int = 42,
                           tile_size: int = 12) -> pd.DataFrame:
    """生成村庄级样本表（含两时点景观指标与目标变量）。"""
    from gwml.metrics import patch_metrics

    rng = np.random.default_rng(seed)
    # 网格坐标（模拟县域范围 ~120km × 120km，间距 4km）
    side = int(np.ceil(np.sqrt(n_villages)))
    xs = np.linspace(0, 120, side)
    ys = np.linspace(0, 120, side)
    gx, gy = np.meshgrid(xs, ys)
    gx, gy = gx.ravel()[:n_villages], gy.ravel()[:n_villages]

    # 平滑随机场因子
    base = _smoothed_field((side, side), rng, scale=1.0)
    elevation = 50 + 450 * (base + 1) / 2            # 50 ~ 500 m
    slope = np.clip(2 + 25 * (base + 1) / 2, 0, 30)  # 2 ~ 27°
    road = np.clip(0.5 + 1.5 * _smoothed_field((side, side), rng, 0.8), 0, None)
    gdp = np.clip(2 + 20 * (1 - (base + 1) / 2) * _smoothed_field(
        (side, side), rng, 0.6), 0, None)
    pop = np.clip(50 + 400 * (1 - (base + 1) / 2), 0, None)
    riv = np.clip(0.2 + 8 * _smoothed_field((side, side), rng, 1.0), 0, None)
    farm = np.clip(0.2 + 0.6 * (1 - (base + 1) / 2) *
                   _smoothed_field((side, side), rng, 0.5), 0, 1)

    feats = np.column_stack([
        elevation.ravel()[:n_villages], slope.ravel()[:n_villages],
        road.ravel()[:n_villages], gdp.ravel()[:n_villages],
        pop.ravel()[:n_villages], riv.ravel()[:n_villages],
        farm.ravel()[:n_villages]])

    # 目标：居民点面积变化率（%）—— 空间非平稳 + 非线性响应
    # 响应系数随"地貌区位"变化：西部(低x)地形主导，东部(高x)经济主导
    xr = gx.ravel()[:n_villages] / 120.0
    w_elev = 1.5 - 1.3 * xr        # 高程作用在西部强
    w_gdp = 0.2 + 1.6 * xr         # 经济作用在东部强
    z = (
        w_elev * (elevation.ravel()[:n_villages] / 300.0) ** 2 * 1.0
        - w_elev * (slope.ravel()[:n_villages] / 20.0) * 0.8
        + w_gdp * np.log1p(gdp.ravel()[:n_villages]) * 2.0
        + 0.3 * (pop.ravel()[:n_villages] / 300.0)
        - 0.25 * (riv.ravel()[:n_villages] / 5.0)
        + 0.6 * (farm.ravel()[:n_villages])
        + 0.15 * (road.ravel()[:n_villages] * (pop.ravel()[:n_villages] > 200))  # 阈值效应
    )
    noise = rng.normal(0, 1.2, n_villages)
    y = z + noise  # 目标变量（%）

    # 两时点景观指标：按四种演化类型模板生成小栅格（图论连通阈值 2 像元）
    types = rng.choice(["stable", "expanding", "aggregating", "fragmenting"],
                       n_villages, p=[0.25, 0.30, 0.25, 0.20])
    rows = []
    for i in range(n_villages):
        t1, t2 = _village_tile(types[i], rng, tile_size)
        m1 = patch_metrics(t1, cell_size=1.0, conn_threshold=2.0)
        m2 = patch_metrics(t2, cell_size=1.0, conn_threshold=2.0)
        rows.append({
            "id": i,
            "x": gx.ravel()[i], "y": gy.ravel()[i],
            "elevation": feats[i, 0], "slope": feats[i, 1],
            "road_density": feats[i, 2], "gdp_per_capita": feats[i, 3],
            "pop_density": feats[i, 4], "river_dist": feats[i, 5],
            "farmland_ratio": feats[i, 6],
            "y_area_change": y[i],
            "t1_PD": m1["PD"], "t1_AI": m1["AI"], "t1_CONN": m1["CONN"],
            "t1_FRAC": m1["FRAC"],
            "t2_PD": m2["PD"], "t2_AI": m2["AI"], "t2_CONN": m2["CONN"],
            "t2_FRAC": m2["FRAC"],
            "true_type": types[i],
        })
    return pd.DataFrame(rows)


def generate_landuse_rasters(seed: int = 42, size: int = 150):
    """生成 SA-CA-Markov 演示用土地利用栅格与约束面。

    Returns
    -------
    dict: landuse_t1, landuse_t2, elevation, slope, gdp, suitability_1
    """
    rng = np.random.default_rng(seed)
    elev = 50 + 500 * (1 - _smoothed_field((size, size), rng, 0.8) * 0.3 + 0.3)
    elev = np.clip(elev, 30, 900)
    slope = np.clip(1 + 28 * (elev - elev.min()) / (elev.max() - elev.min()), 0, 35)
    gdp = np.clip(3 + 25 * (1 - (elev - elev.min()) / (elev.max() - elev.min())) +
                  _smoothed_field((size, size), rng, 2.0), 0, None)

    # 初始土地利用：居民点聚集在低海拔东南，水体固定一块
    lu1 = np.zeros((size, size), dtype=np.int8)
    r = rng.random((size, size))
    lu1[(elev < 250) & (r < 0.25)] = 1
    lu1[110:125, 30:45] = 2  # 水体（不变类）

    # 演化到 t2：居民点向低海拔、低坡度、经济好的区域扩张，同时受
    # "独立空间偏好"（如道路/水源邻近度）影响 —— 后者部分落在生态敏感区，
    # 使生态/均衡情景约束能够实际发挥作用（机理 + 偏好复合的真实演化）。
    prob = 0.5 * (1 - (elev - elev.min()) / (elev.max() - elev.min())) + \
        0.3 * (1 - slope / 35) + 0.2 * (gdp / gdp.max())
    mix_field = np.clip(0.5 + 0.5 * _smoothed_field((size, size), rng, 0.6), 0, 1)
    prob_mix = np.clip(0.65 * prob + 0.35 * mix_field, 0, 1)
    r2 = rng.random((size, size))
    lu2 = lu1.copy()
    grow = (lu1 == 0) & (prob_mix > 0.60) & (r2 < prob_mix) & (lu1 != 2)
    lu2[grow] = 1
    lu2[lu1 == 2] = 2

    # GWRF 风格适宜性表面（地类 1）= 上述复合机制概率
    suitability_1 = np.clip(prob_mix, 0, 1)

    return {
        "landuse_t1": lu1, "landuse_t2": lu2,
        "elevation": elev, "slope": slope, "gdp": gdp,
        "suitability_1": suitability_1,
    }


def save_demo(outdir: str, n_villages: int = 900, seed: int = 42):
    """生成并保存全部演示数据到 outdir。"""
    import os
    os.makedirs(outdir, exist_ok=True)
    df = generate_village_table(n_villages, seed)
    df.to_csv(os.path.join(outdir, "village_table.csv"), index=False)
    rast = generate_landuse_rasters(seed)
    for k, v in rast.items():
        np.save(os.path.join(outdir, f"{k}.npy"), v)
    print(f"演示数据已写入 {outdir}")
    print(f"  村庄样本: {len(df)} 行 × {df.shape[1]} 列 (village_table.csv)")
    print(f"  土地利用栅格: landuse_t1/t2, elevation, slope, gdp, suitability_1")
    return df, rast


if __name__ == "__main__":
    save_demo("./demo_data", n_villages=900, seed=42)
