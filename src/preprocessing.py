"""
数据预处理模块：包含dasymetric mapping将县级社会经济数据 disaggregate 到村级
基于 GeoRRDI Framework (2025)^[3]^
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import yaml


class DasymetricMapper:
    """
    将县级GDP、人口密度等数据通过dasymetric mapping分配到村级
    使用1km分辨率夜间灯光和人口栅格作为辅助信息
    """

    def __init__(self, config_path="config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

    def load_ancillary_rasters(self):
        """加载辅助栅格：夜间灯光 + 人口密度"""
        self.nightlight = rasterio.open(self.cfg["data"]["dasymetric"]["nightlight_raster"])
        self.population = rasterio.open(self.cfg["data"]["dasymetric"]["population_raster"])

        # 统一分辨率为1km
        self.nightlight_data = self.nightlight.read(1)
        self.pop_data = self.population.read(1)

        # 归一化辅助变量
        scaler = MinMaxScaler()
        self.nightlight_norm = scaler.fit_transform(
            self.nightlight_data.reshape(-1, 1)
        ).reshape(self.nightlight_data.shape)
        self.pop_norm = scaler.fit_transform(
            self.pop_data.reshape(-1, 1)
        ).reshape(self.pop_data.shape)

        # 组合权重：夜间灯光60% + 人口40%
        self.weight_raster = 0.6 * self.nightlight_norm + 0.4 * self.pop_norm
        self.weight_raster = self.weight_raster / self.weight_raster.sum()

    def disaggregate_county_to_village(self, county_gdp_path, county_pop_path, village_gdf):
        """
        将县级GDP和人口密度分配到村级
        
        Parameters:
        -----------
        county_gdp_path : str, 县级GDP CSV
        county_pop_path : str, 县级人口 CSV
        village_gdf : GeoDataFrame, 村级边界
        
        Returns:
        --------
        village_gdf : GeoDataFrame, 添加了disaggregated GDP和人口密度
        """
        # 加载县级数据
        county_gdp = pd.read_csv(county_gdp_path)  # columns: county_id, gdp_per_capita
        county_pop = pd.read_csv(county_pop_path)   # columns: county_id, pop_density

        # 空间连接：村庄 -> 所属县
        village_gdf = village_gdf.merge(
            county_gdp[["county_id", "gdp_per_capita"]],
            on="county_id", how="left"
        )
        village_gdf = village_gdf.merge(
            county_pop[["county_id", "pop_density"]],
            on="county_id", how="left"
        )

        # 计算每个村庄的权重
        village_gdf["dasymetric_weight"] = village_gdf.apply(
            self._calculate_village_weight, axis=1
        )

        # 分配GDP和人口密度
        village_gdf["gdp_per_capita_village"] = (
            village_gdf["gdp_per_capita"] * village_gdf["dasymetric_weight"]
        )
        village_gdf["pop_density_village"] = (
            village_gdf["pop_density"] * village_gdf["dasymetric_weight"]
        )

        return village_gdf

    def _calculate_village_weight(self, row):
        """计算单个村庄的dasymetric权重"""
        geom = row["geometry"]
        bounds = geom.bounds  # (minx, miny, maxx, maxy)

        # 从栅格中提取该村庄范围内的权重值
        window = rasterio.windows.from_bounds(*bounds, self.nightlight.transform)
        w = self.weight_raster[
            int(window.row_off):int(window.row_off + window.height),
            int(window.col_off):int(window.col_off + window.width)
        ]
        return w.sum() / w.size  # 归一化

    def build_19_indicator_system(self, village_gdf, dem_path, road_path, river_path):
        """
        构建19指标体系：生态环境、土地利用结构、聚落动态、公共服务可达性
        ^[4]^
        """
        indicators = {}

        # === 生态环境指标 (4个) ===
        with rasterio.open(dem_path) as src:
            dem = src.read(1)
            transform = src.transform
            indicators["elevation"] = self._zonal_stats(village_gdf, dem, transform, "mean")
            indicators["slope"] = self._calc_slope(dem, transform)

        with rasterio.open(road_path) as src:
            road_raster = src.read(1)
            indicators["road_density"] = self._zonal_stats(village_gdf, road_raster, transform, "sum")

        with rasterio.open(river_path) as src:
            river_raster = src.read(1)
            indicators["dist_to_river"] = self._zonal_stats(village_gdf, river_raster, transform, "mean")

        # === 土地利用结构指标 (5个) ===
        indicators["cultivated_ratio"] = village_gdf["cultivated_area"] / village_gdf["total_area"]
        indicators["forest_ratio"] = village_gdf["forest_area"] / village_gdf["total_area"]
        indicators["water_ratio"] = village_gdf["water_area"] / village_gdf["total_area"]
        indicators["builtup_ratio"] = village_gdf["builtup_area"] / village_gdf["total_area"]
        indicators["secondary_tertiary_ratio"] = village_gdf["secondary_tertiary_emp"] / village_gdf["total_emp"]

        # === 聚落动态指标 (5个) ===
        indicators["patch_density"] = village_gdf["n_patches"] / village_gdf["total_area"]
        indicators["aggregation_index"] = village_gdf["ai"]
        indicators["connectivity"] = village_gdf["connectivity"]
        indicators["fractal_dimension"] = village_gdf["fractal_dim"]
        indicators["pop_change_rate"] = village_gdf["pop_2020"] / village_gdf["pop_1990"] - 1

        # === 公共服务可达性指标 (5个) ===
        indicators["dist_to_county_center"] = village_gdf["dist_county_center"]
        indicators["dist_to_prefecture"] = village_gdf["dist_prefecture"]
        indicators["dist_to_provincial_road"] = village_gdf["dist_prov_road"]
        indicators["school_density"] = village_gdf["n_schools"] / village_gdf["total_area"]
        indicators["clinic_density"] = village_gdf["n_clinics"] / village_gdf["total_area"]

        return pd.DataFrame(indicators, index=village_gdf.index)

    def _zonal_stats(self, gdf, raster, transform, stat="mean"):
        """计算矢量区域内的栅格统计值"""
        from rasterio.mask import mask
        results = []
        for _, row in tqdm(gdf.iterrows(), desc="Zonal stats"):
            geom = [row["geometry"].__geo_interface__]
            out_image, out_transform = mask(rasterio.open(raster), geom, crop=True)
            data = out_image[0]
            data = data[data != raster.nodata]
            if stat == "mean":
                results.append(np.mean(data) if len(data) > 0 else np.nan)
            elif stat == "sum":
                results.append(np.sum(data) if len(data) > 0 else np.nan)
        return results

    def _calc_slope(self, dem, transform):
        """从DEM计算坡度"""
        from rasterio.enums import Resampling
        # 简化计算：使用numpy梯度
        dy, dx = np.gradient(dem, transform[4], transform[0])  # pixel size
        slope = np.arctan(np.sqrt(dx**2 + dy**2)) * 180 / np.pi
        return slope


def main_preprocessing():
    """预处理主流程"""
    mapper = DasymetricMapper()
    mapper.load_ancillary_rasters()

    # 加载村级边界
    villages = gpd.read_file("data/raw/hunan_villages_1990_2005_2020.gpkg")

    # Dasymetric mapping
    villages = mapper.disaggregate_county_to_village(
        "data/raw/hunan_gdp_county.csv",
        "data/raw/hunan_pop_county.csv",
        villages
    )

    # 构建19指标
    indicators = mapper.build_19_indicator_system(
        villages,
        "data/raw/hunan_dem_30m.tif",
        "data/raw/hunan_roads.tif",
        "data/raw/hunan_rivers.tif"
    )

    # 合并
    villages = pd.concat([villages, indicators], axis=1)
    villages.to_file("data/processed/villages_with_indicators.gpkg", driver="GPKG")
    print(f"✅ 预处理完成: {len(villages)} 个村庄, 19个指标")


if __name__ == "__main__":
    main_preprocessing()
