"""
Data Preprocessing Module: Includes dasymetric mapping to disaggregate county-level socioeconomic data to village units
Based on the GeoRRDI Framework (2025)^[3]^
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
    Disaggregate county-level GDP, population density and other socioeconomic metrics down to village scale via dasymetric mapping
    Uses 1km resolution nighttime light and population raster datasets as ancillary weighting layers
    """

    def __init__(self, config_file_path="config.yaml"):
        with open(config_file_path, mode="r") as config_file:
            self.config_dict = yaml.safe_load(config_file)

    def load_ancillary_raster_datasets(self):
        """Load auxiliary raster inputs: nighttime light intensity and population density grids"""
        self.nightlight_raster_reader = rasterio.open(self.config_dict["data"]["dasymetric"]["nightlight_raster"])
        self.population_raster_reader = rasterio.open(self.config_dict["data"]["dasymetric"]["population_raster"])

        # Standardize spatial resolution to 1 kilometer
        self.nightlight_raw_data = self.nightlight_raster_reader.read(1)
        self.population_raw_data = self.population_raster_reader.read(1)

        # Min-max normalization of auxiliary weighting variables
        minmax_scaler = MinMaxScaler()
        self.nightlight_normalized = minmax_scaler.fit_transform(
            self.nightlight_raw_data.reshape(-1, 1)
        ).reshape(self.nightlight_raw_data.shape)

        self.population_normalized = minmax_scaler.fit_transform(
            self.population_raw_data.reshape(-1, 1)
        ).reshape(self.population_raw_data.shape)

        # Composite weighting layer: 60% nighttime light + 40% population grid
        self.composite_weight_raster = 0.6 * self.nightlight_normalized + 0.4 * self.population_normalized
        self.composite_weight_raster = self.composite_weight_raster / self.composite_weight_raster.sum()

    def disaggregate_county_statistics_to_villages(self, county_gdp_csv_path, county_pop_csv_path, village_boundary_gdf):
        """
        Distribute aggregated county-level GDP and population density values to individual village polygons

        Parameters
        ----------
        county_gdp_csv_path : str
            File path to CSV containing county GDP records
        county_pop_csv_path : str
            File path to CSV containing county population statistics
        village_boundary_gdf : GeoDataFrame
            Geospatial dataset with village administrative boundaries

        Returns
        -------
        GeoDataFrame
            Original village boundary dataset appended with disaggregated village-level GDP and population density metrics
        """
        # Import county aggregated socioeconomic tables
        county_gdp_dataset = pd.read_csv(county_gdp_csv_path)  # Fields: county_id, gdp_per_capita
        county_pop_dataset = pd.read_csv(county_pop_csv_path)   # Fields: county_id, pop_density

        # Spatial attribute join: match each village to its parent administrative county
        village_boundary_gdf = village_boundary_gdf.merge(
            county_gdp_dataset[["county_id", "gdp_per_capita"]],
            on="county_id",
            how="left"
        )
        village_boundary_gdf = village_boundary_gdf.merge(
            county_pop_dataset[["county_id", "pop_density"]],
            on="county_id",
            how="left"
        )

        # Calculate dasymetric weighting coefficient for every village unit
        village_boundary_gdf["dasymetric_weight_coefficient"] = village_boundary_gdf.apply(
            self._calculate_single_village_weight,
            axis=1
        )

        # Distribute county aggregated metrics to village scale using spatial weights
        village_boundary_gdf["village_gdp_per_capita"] = (
            village_boundary_gdf["gdp_per_capita"] * village_boundary_gdf["dasymetric_weight_coefficient"]
        )
        village_boundary_gdf["village_population_density"] = (
            village_boundary_gdf["pop_density"] * village_boundary_gdf["dasymetric_weight_coefficient"]
        )

        return village_boundary_gdf

    def _calculate_single_village_weight(self, record_row):
        """Compute dasymetric weighting value for one individual village polygon"""
        village_geometry = record_row["geometry"]
        min_x, min_y, max_x, max_y = village_geometry.bounds

        # Extract raster pixel values overlapping the village boundary extent
        extraction_window = rasterio.windows.from_bounds(
            min_x, min_y, max_x, max_y,
            self.nightlight_raster_reader.transform
        )
        window_pixel_matrix = self.composite_weight_raster[
            int(extraction_window.row_off): int(extraction_window.row_off + extraction_window.height),
            int(extraction_window.col_off): int(extraction_window.col_off + extraction_window.width)
        ]
        return window_pixel_matrix.sum() / window_pixel_matrix.size  # Area-normalized weight

    def construct_nineteen_indicator_system(self, village_gdf, dem_file_path, road_raster_path, river_raster_path):
        """
        Build integrated system of 19 explanatory indicators covering four categories:
        eco-environmental metrics, land use composition, settlement dynamics, public service accessibility
        ^[4]^
        """
        indicator_storage_dict = {}

        # ========== Eco-Environmental Indicators (4 total) ==========
        with rasterio.open(dem_file_path) as dem_reader:
            dem_grid = dem_reader.read(1)
            raster_transform_matrix = dem_reader.transform
            indicator_storage_dict["elevation_mean"] = self._compute_zonal_statistic(
                village_gdf, dem_grid, raster_transform_matrix, stat_method="mean"
            )
            indicator_storage_dict["slope_mean_degree"] = self._generate_slope_raster(dem_grid, raster_transform_matrix)

        with rasterio.open(road_raster_path) as road_reader:
            road_intensity_grid = road_reader.read(1)
            indicator_storage_dict["road_network_density"] = self._compute_zonal_statistic(
                village_gdf, road_intensity_grid, raster_transform_matrix, stat_method="sum"
            )

        with rasterio.open(river_raster_path) as river_reader:
            river_distance_grid = river_reader.read(1)
            indicator_storage_dict["mean_distance_to_river"] = self._compute_zonal_statistic(
                village_gdf, river_distance_grid, raster_transform_matrix, stat_method="mean"
            )

        # ========== Land Use Composition Indicators (5 total) ==========
        indicator_storage_dict["cultivated_land_ratio"] = village_gdf["cultivated_area"] / village_gdf["total_village_area"]
        indicator_storage_dict["forest_land_ratio"] = village_gdf["forest_area"] / village_gdf["total_village_area"]
        indicator_storage_dict["water_body_ratio"] = village_gdf["water_area"] / village_gdf["total_village_area"]
        indicator_storage_dict["built_up_ratio"] = village_gdf["builtup_area"] / village_gdf["total_village_area"]
        indicator_storage_dict["secondary_tertiary_employment_ratio"] = village_gdf["secondary_tertiary_emp"] / village_gdf["total_employment"]

        # ========== Settlement Landscape Dynamic Indicators (5 total) ==========
        indicator_storage_dict["patch_density"] = village_gdf["patch_count"] / village_gdf["total_village_area"]
        indicator_storage_dict["aggregation_index"] = village_gdf["aggregation_index"]
        indicator_storage_dict["landscape_connectivity"] = village_gdf["connectivity_metric"]
        indicator_storage_dict["mean_fractal_dimension"] = village_gdf["fractal_dim"]
        indicator_storage_dict["population_change_rate"] = village_gdf["pop_2020"] / village_gdf["pop_1990"] - 1

        # ========== Public Service Accessibility Indicators (5 total) ==========
        indicator_storage_dict["distance_to_county_admin_center"] = village_gdf["dist_county_center"]
        indicator_storage_dict["distance_to_prefecture_city"] = village_gdf["dist_prefecture"]
        indicator_storage_dict["distance_to_provincial_highway"] = village_gdf["dist_prov_road"]
        indicator_storage_dict["school_density_per_area"] = village_gdf["school_count"] / village_gdf["total_village_area"]
        indicator_storage_dict["clinic_density_per_area"] = village_gdf["clinic_count"] / village_gdf["total_village_area"]

        return pd.DataFrame(indicator_storage_dict, index=village_gdf.index)

    def _compute_zonal_statistic(self, geodataframe, raster_grid, raster_transform, stat_method="mean"):
        """Calculate aggregate raster statistics within each vector polygon zone"""
        from rasterio.mask import mask
        zone_stat_output_list = []
        for _, row_record in tqdm(geodataframe.iterrows(), desc="Calculating Zonal Statistics"):
            polygon_geojson = [row_record["geometry"].__geo_interface__]
            clipped_raster_array, _ = mask(rasterio.open(raster_grid), polygon_geojson, crop=True)
            valid_pixel_values = clipped_raster_array[0]
            valid_pixel_values = valid_pixel_values[valid_pixel_values != raster_grid.nodata]

            if len(valid_pixel_values) == 0:
                zone_stat_output_list.append(np.nan)
            elif stat_method == "mean":
                zone_stat_output_list.append(np.mean(valid_pixel_values))
            elif stat_method == "sum":
                zone_stat_output_list.append(np.sum(valid_pixel_values))
        return zone_stat_output_list

    def _generate_slope_raster(self, dem_grid, raster_transform):
        """Derive terrain slope grid from input DEM elevation raster"""
        # Simplified gradient calculation using numpy differential operator
        vertical_gradient, horizontal_gradient = np.gradient(
            dem_grid,
            raster_transform[4],
            raster_transform[0]
        )
        slope_degree_grid = np.arctan(np.sqrt(horizontal_gradient ** 2 + vertical_gradient ** 2)) * 180 / np.pi
        return slope_degree_grid


def execute_full_preprocessing_pipeline():
    """Main execution workflow for entire geospatial preprocessing module"""
    dasymetric_tool = DasymetricMapper()
    dasymetric_tool.load_ancillary_raster_datasets()

    # Import raw village administrative boundary layer
    village_boundary_dataset = gpd.read_file("data/raw/hunan_villages_1990_2005_2020.gpkg")

    # Run dasymetric disaggregation workflow
    village_boundary_dataset = dasymetric_tool.disaggregate_county_statistics_to_villages(
        county_gdp_csv_path="data/raw/hunan_gdp_county.csv",
        county_pop_csv_path="data/raw/hunan_pop_county.csv",
        village_boundary_gdf=village_boundary_dataset
    )

    # Construct full 19-dimensional indicator dataset
    integrated_indicator_table = dasymetric_tool.construct_nineteen_indicator_system(
        village_gdf=village_boundary_dataset,
        dem_file_path="data/raw/hunan_dem_30m.tif",
        road_raster_path="data/raw/hunan_roads.tif",
        river_raster_path="data/raw/hunan_rivers.tif"
    )

    # Merge geospatial boundaries with computed indicator attributes
    final_village_dataset = pd.concat([village_boundary_dataset, integrated_indicator_table], axis=1)
    final_village_dataset.to_file("data/processed/villages_with_indicators.gpkg", driver="GPKG")
    print(f"✅ Preprocessing pipeline finished successfully: {len(final_village_dataset)} village units with 19 integrated indicators")


if __name__ == "__main__":
    execute_full_preprocessing_pipeline()