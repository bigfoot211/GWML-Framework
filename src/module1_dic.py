"""
Module 1: Delta-Index Coupling (DIC) — 变化敏感的复合指标
基于传统景观指标 + 时间导数构建
^[5]^
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.cluster import KMeans
from tqdm import tqdm


class DeltaIndexCoupling:
    """
    DIC模块：计算ΔPD, ΔAI, ΔCONN, ΔFRAC，然后用Jenks Natural Breaks分类
    """

    def __init__(self, time_points=[1990, 2005, 2020]):
        self.time_points = time_points

    def compute_landscape_metrics(self, settlements_gdf, time_col="year"):
        """
        计算传统景观指标：PD(斑块密度), AI(聚集度), CONN(连通性), FRAC(分形维数)
        """
        metrics = {}

        for t in self.time_points:
            subset = settlements_gdf[settlements_gdf[time_col] == t]

            # Patch Density (PD)
            total_area = subset["geometry"].area.sum()
            n_patches = len(subset)
            metrics[f"PD_{t}"] = n_patches / (total_area / 1e6)  # per km²

            # Aggregation Index (AI) — 简化版
            metrics[f"AI_{t}"] = subset["ai"].mean() if "ai" in subset.columns else np.nan

            # Connectivity (IIC-based)
            metrics[f"CONN_{t}"] = subset["connectivity"].mean() if "connectivity" in subset.columns else np.nan

            # Fractal Dimension
            metrics[f"FRAC_{t}"] = subset["fractal_dim"].mean() if "fractal_dim" in subset.columns else np.nan

        return pd.DataFrame(metrics)

    def compute_delta_indices(self, metrics_df):
        """
        计算Delta指标：ΔPD, ΔAI, ΔCONN, ΔFRAC
        
        """
        deltas = {}

        # ΔPD = PD_2020 - PD_1990 (也计算中间段)
        deltas["ΔPD_90_05"] = metrics_df["PD_2005"] - metrics_df["PD_1990"]
        deltas["ΔPD_05_20"] = metrics_df["PD_2020"] - metrics_df["PD_2005"]
        deltas["ΔPD_total"] = metrics_df["PD_2020"] - metrics_df["PD_1990"]

        deltas["ΔAI_90_05"] = metrics_df["AI_2005"] - metrics_df["AI_1990"]
        deltas["ΔAI_05_20"] = metrics_df["AI_2020"] - metrics_df["AI_2005"]
        deltas["ΔAI_total"] = metrics_df["AI_2020"] - metrics_df["AI_1990"]

        deltas["ΔCONN_90_05"] = metrics_df["CONN_2005"] - metrics_df["CONN_1990"]
        deltas["ΔCONN_05_20"] = metrics_df["CONN_2020"] - metrics_df["CONN_2005"]
        deltas["ΔCONN_total"] = metrics_df["CONN_2020"] - metrics_df["CONN_1990"]

        deltas["ΔFRAC_90_05"] = metrics_df["FRAC_2005"] - metrics_df["FRAC_1990"]
        deltas["ΔFRAC_05_20"] = metrics_df["FRAC_2020"] - metrics_df["FRAC_2005"]
        deltas["ΔFRAC_total"] = metrics_df["FRAC_2020"] - metrics_df["FRAC_1990"]

        return pd.DataFrame(deltas)

    def jenks_classify(self, deltas_df, n_classes=4):
        """
        Jenks Natural Breaks分类为4种演化类型
        ^[6]^
        
        Type | ΔPD | ΔAI | ΔCONN | Interpretation
        T1: Aggregation-Intensive | ↓↓ | ↑↑ | ↑↑ | Villages consolidating
        T2: Fragmentation-Dominant | ↑↑ | ↓↓ | ↓↓ | Villages splitting
        T3: Expansion-Driven | ↑ | ↓ | ↑ | Physical expansion
        T4: Stable-Equilibrium | ≈0 | ≈0 | ≈0 | Minimal change
        """
        from jenkspy import jenks_breaks

        # 使用ΔPD, ΔAI, ΔCONN三个指标进行分类
        features = deltas_df[["ΔPD_total", "ΔAI_total", "ΔCONN_total"]].values

        # 逐列Jenks breaks
        breaks_pd = jenks_breaks(features[:, 0], nb_class=n_classes)
        breaks_ai = jenks_breaks(features[:, 1], nb_class=n_classes)
        breaks_conn = jenks_breaks(features[:, 2], nb_class=n_classes)

        # 分类逻辑
        def classify(row):
            pd, ai, conn = row["ΔPD_total"], row["ΔAI_total"], row["ΔCONN_total"]

            pd_class = np.digitize([pd], breaks_pd)[0]
            ai_class = np.digitize([ai], breaks_ai)[0]
            conn_class = np.digitize([conn], breaks_conn)[0]

            # T1: PD↓↓(0), AI↑↑(3), CONN↑↑(3)
            if pd_class <= 1 and ai_class >= 2 and conn_class >= 2:
                return "T1: Aggregation-Intensive"
            # T2: PD↑↑(3), AI↓↓(0), CONN↓↓(0)
            elif pd_class >= 2 and ai_class <= 1 and conn_class <= 1:
                return "T2: Fragmentation-Dominant"
            # T3: PD↑(2), AI↓(1), CONN↑(2)
            elif pd_class == 2 and ai_class == 1 and conn_class == 2:
                return "T3: Expansion-Driven"
            # T4: all ≈0
            else:
                return "T4: Stable-Equilibrium"

        deltas_df["evolution_type"] = deltas_df.apply(classify, axis=1)
        return deltas_df

    def run(self, settlements_gdf):
        """运行完整DIC流程"""
        print("📊 Running Delta-Index Coupling (DIC)...")

        metrics = self.compute_landscape_metrics(settlements_gdf)
        deltas = self.compute_delta_indices(metrics)
        classified = self.jenks_classify(deltas)

        # 合并回原始数据
        result = settlements_gdf.merge(classified, left_index=True, right_index=True)

        # 统计各类型数量
        type_counts = result["evolution_type"].value_counts()
        print("\n📈 DIC Classification Results:")
        for t, n in type_counts.items():
            print(f"  {t}: {n} villages ({n/len(result)*100:.1f}%)")

        # 关键发现：东西部连通性差异
        dongting = result[result["region"] == "Dongting Lake plain"]
        wuling = result[result["region"] == "Wuling Mountains"]

        print(f"\n🔗 Connectivity Dynamics:")
        print(f"  Dongting Lake plain: ΔCONN = {dongting['ΔCONN_total'].mean():+.2f}")
        print(f"  Wuling Mountains: ΔCONN = {wuling['ΔCONN_total'].mean():+.2f}")
        print(f"  ")

        return result


if __name__ == "__main__":
    villages = gpd.read_file("data/processed/villages_with_indicators.gpkg")
    dic = DeltaIndexCoupling()
    dic_result = dic.run(villages)
    dic_result.to_file("data/processed/dic_results.gpkg", driver="GPKG")
