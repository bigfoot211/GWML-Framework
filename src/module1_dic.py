"""
Module 1: Delta-Index Coupling (DIC) — Change-Sensitive Composite Index
Constructed based on traditional landscape metrics plus temporal derivatives
^[5]^
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.cluster import KMeans
from tqdm import tqdm


class DeltaIndexCoupling:
    """
    DIC Module: Calculate ΔPD, ΔAI, ΔCONN, ΔFRAC, then perform classification via Jenks Natural Breaks
    """

    def __init__(self, time_points=[1990, 2005, 2020]):
        self.time_points = time_points

    def compute_landscape_metrics(self, settlements_gdf, time_column="year"):
        """
        Calculate conventional landscape metrics: PD (Patch Density), AI (Aggregation Index),
        CONN (Connectivity), FRAC (Fractal Dimension)
        """
        metric_records = {}

        for year in self.time_points:
            temporal_subset = settlements_gdf[settlements_gdf[settlements_gdf[time_column]] == year]

            # Patch Density (PD)
            total_surface_area = temporal_subset["geometry"].area.sum()
            patch_count = len(temporal_subset)
            metric_records[f"PD_{year}"] = patch_count / (total_surface_area / 10**6)  # Unit: per square kilometer

            # Simplified Aggregation Index (AI)
            metric_records[f"AI_{year}"] = temporal_subset["ai"].mean() if "ai" in temporal_subset.columns else np.nan

            # IIC-based Connectivity Metric
            metric_records[f"CONN_{year}"] = temporal_subset["connectivity"].mean() if "connectivity" in temporal_subset.columns else np.nan

            # Mean Fractal Dimension
            metric_records[f"FRAC_{year}"] = temporal_subset["fractal_dim"].mean() if "fractal_dim" in temporal_subset.columns else np.nan

        return pd.DataFrame(metric_records)

    def compute_delta_metrics(self, landscape_metric_df):
        """
        Compute delta change metrics: ΔPD, ΔAI, ΔCONN, ΔFRAC
        """
        delta_records = {}

        # ΔPD = PD value difference between two time stages (inter-period and total temporal change)
        delta_records["ΔPD_90_05"] = landscape_metric_df["PD_2005"] - landscape_metric_df["PD_1990"]
        delta_records["ΔPD_05_20"] = landscape_metric_df["PD_2020"] - landscape_metric_df["PD_2005"]
        delta_records["ΔPD_total"] = landscape_metric_df["PD_2020"] - landscape_metric_df["PD_1990"]

        delta_records["ΔAI_90_05"] = landscape_metric_df["AI_2005"] - landscape_metric_df["AI_1990"]
        delta_records["ΔAI_05_20"] = landscape_metric_df["AI_2020"] - landscape_metric_df["AI_2005"]
        delta_records["ΔAI_total"] = landscape_metric_df["AI_2020"] - landscape_metric_df["AI_1990"]

        delta_records["ΔCONN_90_05"] = landscape_metric_df["CONN_2005"] - landscape_metric_df["CONN_1990"]
        delta_records["ΔCONN_05_20"] = landscape_metric_df["CONN_2020"] - landscape_metric_df["CONN_2005"]
        delta_records["ΔCONN_total"] = landscape_metric_df["CONN_2020"] - landscape_metric_df["CONN_1990"]

        delta_records["ΔFRAC_90_05"] = landscape_metric_df["FRAC_2005"] - landscape_metric_df["FRAC_1990"]
        delta_records["ΔFRAC_05_20"] = landscape_metric_df["FRAC_2020"] - landscape_metric_df["FRAC_2005"]
        delta_records["ΔFRAC_total"] = landscape_metric_df["FRAC_2020"] - landscape_metric_df["FRAC_1990"]

        return pd.DataFrame(delta_records)

    def jenks_natural_break_classification(self, delta_metric_df, class_number=4):
        """
        Classify spatial units into 4 evolutionary types using Jenks Natural Breaks algorithm
        ^[6]^

        Evolution Type | ΔPD | ΔAI | ΔCONN | Semantic Interpretation
        T1: Aggregation-Intensive | Sharp Decline | Sharp Rise | Sharp Rise | Rural settlement consolidation
        T2: Fragmentation-Dominant | Sharp Rise | Sharp Decline | Sharp Decline | Severe village patch splitting
        T3: Expansion-Driven | Moderate Rise | Moderate Decline | Moderate Rise | Physical outward expansion of settlements
        T4: Stable-Equilibrium | Near zero | Near zero | Near zero | Negligible landscape structural changes
        """
        from jenkspy import jenks_breaks

        # Input feature matrix based on total change metrics ΔPD_total, ΔAI_total, ΔCONN_total
        feature_matrix = delta_metric_df[["ΔPD_total", "ΔAI_total", "ΔCONN_total"]].values

        # Generate Jenks break thresholds for each metric column separately
        pd_break_points = jenks_breaks(feature_matrix[:, 0], nb_class=class_number)
        ai_break_points = jenks_breaks(feature_matrix[:, 1], nb_class=class_number)
        conn_break_points = jenks_breaks(feature_matrix[:, 2], nb_class=class_number)

        # Classification rule function for each spatial unit row
        def assign_evolution_category(row_data):
            pd_value, ai_value, conn_value = row_data["ΔPD_total"], row_data["ΔAI_total"], row_data["ΔCONN_total"]

            pd_category = np.digitize([pd_value], pd_break_points)[0]
            ai_category = np.digitize([ai_value], ai_break_points)[0]
            conn_category = np.digitize([conn_value], conn_break_points)[0]

            # T1: Extremely low PD, extremely high AI & CONN
            if pd_category <= 1 and ai_category >= 2 and conn_category >= 2:
                return "T1: Aggregation-Intensive"
            # T2: Extremely high PD, extremely low AI & CONN
            elif pd_category >= 2 and ai_category <= 1 and conn_category <= 1:
                return "T2: Fragmentation-Dominant"
            # T3: Moderate PD growth, slight AI decline, moderate CONN growth
            elif pd_category == 2 and ai_category == 1 and conn_category == 2:
                return "T3: Expansion-Driven"
            # T4: All metrics show trivial fluctuation near zero
            else:
                return "T4: Stable-Equilibrium"

        delta_metric_df["evolution_type"] = delta_metric_df.apply(assign_evolution_category, axis=1)
        return delta_metric_df

    def execute_full_dic_workflow(self, settlements_gdf):
        """Execute the complete Delta-Index Coupling computational pipeline"""
        print("📊 Running Delta-Index Coupling (DIC) Workflow...")

        landscape_metrics_output = self.compute_landscape_metrics(settlements_gdf)
        delta_metrics_output = self.compute_delta_metrics(landscape_metrics_output)
        classified_delta_results = self.jenks_natural_break_classification(delta_metrics_output)

        # Merge computed metrics and classification labels back to original geospatial dataset
        final_output_gdf = settlements_gdf.merge(classified_delta_results, left_index=True, right_index=True)

        # Count sample quantity and proportion for each evolutionary type
        type_statistics = final_output_gdf["evolution_type"].value_counts()
        print("\n📈 DIC Evolution Classification Summary Statistics:")
        for category_label, sample_count in type_statistics.items():
            proportion_pct = sample_count / len(final_output_gdf) * 100
            print(f"  {category_label}: {sample_count} villages ({proportion_pct:.1f}%)")

        # Core regional comparative analysis: connectivity disparity between two geographical zones
        dongting_plain_subset = final_output_gdf[final_output_gdf["region"] == "Dongting Lake plain"]
        wuling_mountain_subset = final_output_gdf[final_output_gdf["region"] == "Wuling Mountains"]

        print(f"\n🔗 Inter-Regional Connectivity Change Comparison:")
        print(f"  Dongting Lake plain: Mean total ΔCONN = {dongting_plain_subset['ΔCONN_total'].mean():+.2f}")
        print(f"  Wuling Mountains: Mean total ΔCONN = {wuling_mountain_subset['ΔCONN_total'].mean():+.2f}")
        print(f"  ")

        return final_output_gdf


if __name__ == "__main__":
    village_geodata = gpd.read_file("data/processed/villages_with_indicators.gpkg")
    dic_model = DeltaIndexCoupling()
    dic_calculation_results = dic_model.execute_full_dic_workflow(village_geodata)
    dic_calculation_results.to_file("data/processed/dic_results.gpkg", driver="GPKG")