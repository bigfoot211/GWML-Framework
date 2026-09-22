# -*- coding: utf-8 -*-
"""
gwml.metrics
============
景观格局指标计算（斑块尺度，基于二值栅格 numpy 数组实现）。

对应论文模块 1（DIC，Delta-Index Coupling）所依赖的四项核心指标：
    PD      斑块密度   Patch Density
    AI      集聚指数   Aggregation Index
    CONN    景观连通度 Landscape Connectivity（图论式，基于斑块图）
    FRAC    平均分维数 Fractal Dimension (mean)

说明
----
- 输入为二值栅格：1 表示居民点/目标地类，0 表示非目标地类。
- 所有指标均以"像元"为基本单位计算；`cell_size` 仅用于换算为实际面积
  （如 km²），对差分指标的符号与相对大小不构成影响。
- CONN 采用图论连通思想：斑块为节点，距离不超过 `conn_threshold`
  （单位为像元）的斑块视为连通，取"最大连通成分面积 / 地类总面积"作为
  连通度，取值范围 [0, 1]，越大越连通。该指标对应论文中
  "Probability of Connectivity / IIC" 的简化实现。
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

_EPS = 1e-12


def _as_binary(arr) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError("栅格必须为二维数组")
    return (arr != 0).astype(np.int8)


def _patch_perimeters(labels: np.ndarray, n_patches: int) -> np.ndarray:
    """逐斑块统计周长：每个目标像元与其 4 邻域中"异类/背景"像元的共享边数。"""
    count = np.zeros(n_patches + 1, dtype=float)
    padded = np.pad(labels, 1, mode="constant", constant_values=0)
    H, W = labels.shape
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if abs(di) + abs(dj) != 1:
                continue  # 4 邻域
            nb = padded[1 + di:1 + di + H, 1 + dj:1 + dj + W]
            diff = (labels != nb) & (labels > 0)
            np.add.at(count, labels[diff], 1.0)
    return count[1:]  # 去掉背景


def patch_metrics(binary: np.ndarray, cell_size: float = 1.0,
                  conn_threshold: float | None = None,
                  connectivity: int = 4) -> dict:
    """计算二值栅格的四项核心景观指标。

    Parameters
    ----------
    binary : (H, W) ndarray
        二值栅格（非 0 即目标地类）。
    cell_size : float
        单个像元的边长（单位自定，如 km），用于面积换算。
    conn_threshold : float | None
        斑块连通阈值（像元数）。None 时使用 8 邻接（斑块直接相接视为连通）。
    connectivity : int
        斑块标记的邻接方式（4 或 8），用于 ndimage.label。

    Returns
    -------
    dict 含 PD, AI, CONN, FRAC, n_patches, total_area(像元), class_area(像元),
          largest_component_area(像元), component_n
    """
    arr = _as_binary(binary)
    H, W = arr.shape
    n_cells = H * W
    class_area = int(arr.sum())
    if class_area == 0:
        return {
            "PD": 0.0, "AI": 0.0, "CONN": 0.0, "FRAC": 0.0,
            "n_patches": 0, "total_area": n_cells, "class_area": 0,
            "largest_component_area": 0.0, "component_n": 0,
            "perimeter": 0.0, "area_km2": 0.0,
        }

    # 斑块标记（连通域）
    structure = np.ones((3, 3), dtype=int) if connectivity == 8 else \
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
    labels, n_patches = ndimage.label(arr, structure=structure)

    areas = np.bincount(labels.ravel())[1:].astype(float)  # 各斑块面积(像元)

    # --- PD: 斑块密度（每 10^4 像元斑块数，即标准化的斑块丰度）--- #
    pd = n_patches / (class_area / 1e4) if class_area else 0.0

    # --- AI: 集聚指数（同类邻接占比）--- #
    # g_ii: 目标像元之间共享边的数量；g_ik: 目标像元与非目标像元的边界数
    padded = np.pad(arr, 1, mode="constant", constant_values=0)
    like_adj = 0
    cross_adj = 0
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            if abs(di) + abs(dj) != 1:
                continue  # 4 邻域
            nb = padded[1 + di:1 + di + H, 1 + dj:1 + dj + W]
            same = (arr == 1) & (nb == 1)
            diff = (arr == 1) & (nb == 0)
            like_adj += int(same.sum())
            cross_adj += int(diff.sum())
    g_ii = like_adj // 2  # 每条共享边被统计两次
    ai = 100.0 * g_ii / (g_ii + cross_adj) if (g_ii + cross_adj) > 0 else 0.0

    # --- CONN: 景观连通度（图论式）--- #
    if conn_threshold is None:
        # 8 邻接下直接相接的斑块视为一个成分
        comp_labels, component_n = ndimage.label(
            (labels > 0).astype(np.int8),
            structure=np.ones((3, 3), dtype=int))
        comp_areas = np.bincount(comp_labels.ravel())[1:].astype(float)
    else:
        # 以斑块质心距离 <= conn_threshold 构建邻接图
        cy = ndimage.center_of_mass(
            np.ones_like(labels), labels=labels, index=np.arange(1, n_patches + 1))
        cents = np.asarray(cy, dtype=float)
        d = cents[:, None, :] - cents[None, :, :]
        dmat = np.sqrt((d ** 2).sum(axis=2))
        adj = dmat <= float(conn_threshold)
        n_nodes = n_patches
        parent = list(range(n_nodes))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if adj[i, j]:
                    union(i, j)
        comp_of = np.array([find(i) for i in range(n_nodes)])
        comp_areas = np.zeros(comp_of.max() + 1)
        for i in range(n_nodes):
            comp_areas[comp_of[i]] += areas[i]
        component_n = int(len(comp_areas))

    largest_component_area = float(comp_areas.max())
    conn = largest_component_area / class_area if class_area else 0.0

    # --- FRAC: 平均分维数 --- #
    perims = _patch_perimeters(labels, n_patches)
    frac = np.ones(n_patches, dtype=float)   # 单像元斑块分维约定为 1.0
    valid = (areas > 1) & (perims > 0)
    # FRAGSTATS: FRAC = 2 * ln(0.25 * P) / ln(A)
    frac[valid] = 2.0 * np.log(0.25 * perims[valid]) / np.log(areas[valid])
    frac = np.clip(frac, 1.0, 2.0)
    frac_mn = float(np.nanmean(frac)) if n_patches else 0.0

    return {
        "PD": float(pd),
        "AI": float(ai),
        "CONN": float(conn),
        "FRAC": float(frac_mn),
        "n_patches": int(n_patches),
        "total_area": n_cells,
        "class_area": class_area,
        "largest_component_area": largest_component_area,
        "component_n": int(component_n),
        "perimeter": float(perims.sum()),
        "area_km2": float(class_area * cell_size ** 2),
    }


def metrics_from_tiles(tiles_t1: dict, tiles_t2: dict,
                       cell_size: float = 1.0,
                       conn_threshold: float | None = None) -> dict:
    """批量计算每个空间单元（村庄）两个时点的四项景观指标。

    Parameters
    ----------
    tiles_t1 : dict {unit_id: 2D ndarray}
        时点 t1 每个单元的二值栅格。
    tiles_t2 : dict {unit_id: 2D ndarray}
        时点 t2 每个单元的二值栅格。
    cell_size : float
        像元边长（单位自定）。
    conn_threshold : float | None
        连通阈值（像元）。

    Returns
    -------
    dict {unit_id: {t1: {指标...}, t2: {指标...}}}
    """
    out = {}
    for uid in tiles_t1:
        out[uid] = {
            "t1": patch_metrics(tiles_t1[uid], cell_size, conn_threshold),
            "t2": patch_metrics(tiles_t2[uid], cell_size, conn_threshold),
        }
    return out


# --------------------------------------------------------------------------- #
# 可选：矢量面 -> 二值栅格（需 geopandas / rasterio，可选依赖）
# --------------------------------------------------------------------------- #


def rasterize_polygons(gdf, value_field: str | None = None,
                       transform=None, out_shape=None,
                       all_touched: bool = True):
    """将面矢量（GeoDataFrame）栅格化为二值/数值栅格。

    Parameters
    ----------
    gdf : GeoDataFrame
        要素几何（面）。
    value_field : str | None
        用于栅格值的字段；None 时输出二值（有要素=1）。
    transform : tuple
        affine 变换 (a, b, c, d, e, f)（仿射参数，供 rasterio）。
    out_shape : tuple
        输出栅格形状 (H, W)。未给出时按要素外包络与 30m 像元推算。
    all_touched : bool

    Returns
    -------
    (array, transform, out_shape)
    """
    try:
        import rasterio
        from rasterio.features import rasterize
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "rasterize_polygons 需要可选依赖 rasterio（pip install rasterio）") from e

    if transform is None or out_shape is None:
        bounds = gdf.total_bounds
        res = 30.0  # 默认 30 m
        width = max(int(np.ceil((bounds[2] - bounds[0]) / res)), 1)
        height = max(int(np.ceil((bounds[3] - bounds[1]) / res)), 1)
        transform = rasterio.transform.from_origin(bounds[0], bounds[3], res, res)
        out_shape = (height, width)
    values = gdf[value_field].values if value_field else 1
    shapes = ((geom, v) for geom, v in zip(gdf.geometry, values))
    arr = rasterize(shapes, out_shape=out_shape, transform=transform,
                    fill=0, all_touched=all_touched)
    return arr.astype(np.float64), transform, out_shape
