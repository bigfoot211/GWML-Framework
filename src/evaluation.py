"""
Model Comparison Protocol — 三维评估框架
Following the GeoRRDI framework (2025)

| Dimension         | Metric                  | Purpose              |
|-------------------|-------------------------|----------------------|
| Accuracy          | RMSE, R²               | Overall fit          |
| Spatial Autocorr  | Moran's I of residuals | Spatial structure    |
| Ordinal Consist   | Kappa coefficient       | Classification cons. |

引用: Results Section 4.2.1
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from pysal.lib import weights
from pysal.explore.esda import Moran
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import cohen_kappa_score


def three_d_evaluation(y_true, y_pred, coordinates, model_name="GWRF"):
    """
    三维评估：Accuracy + Spatial Autocorrelation + Ordinal Consistency
    
    引用: GeoRRDI Framework (2025)
    """
    
    # ─── Dimension 1: Accuracy ───
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # ─── Dimension 2: Spatial Autocorrelation of Residuals ───
    residuals = y_true - y_pred
    w = weights.KNN.from_array(coordinates, k=5)
    w.transform = 'r'
    moran = Moran(residuals, w)
    morans_i = moran.I
    
    # ─── Dimension 3: Ordinal Consistency (Kappa) ───
    # 将连续预测转为有序分类用于Kappa计算
    y_true_class = pd.qcut(y_true, q=4, labels=[1,2,3,4])
    y_pred_class = pd.qcut(y_pred, q=4, labels=[1,2,3,4])
    kappa = cohen_kappa_score(y_true_class, y_pred_class)
    
    results = {
        "model": model_name,
        "R2": round(r2, 3),
        "RMSE": round(rmse, 3),
        "Moran's_I": round(morans_i, 3),
        "Kappa": round(kappa, 3)
    }
    
    print(f"\n📊 {model_name} Evaluation:")
    print(f"   R² = {results['R2']} | RMSE = {results['RMSE']}")
    print(f"   Moran's I = {results['Moran's_I']} | Kappa = {results['Kappa']}")
    
    return results


def compare_all_models(y_true, y_pred_dict, coordinates):
    """
    对比五种模型: OLS, GWR, RF, Geodetector, GWRF, GNID
    
    引用: Results Table 4.2.1
    | Model    | R²    | RMSE  | Moran's I | Kappa |
    | OLS      | 0.543 | 1.892 | 0.312     | 0.456 |
    | GWR      | 0.671 | 1.523 | 0.187     | 0.589 |
    | RF       | 0.734 | 1.387 | N/A       | 0.634 |
    | Geodetect| 0.612 | —     | —         | 0.521 |
    | GWRF     | 0.812 | 1.156 | 0.043     | 0.789 |
    | GNID     | 0.778 | 1.234 | 0.067     | 0.745 |
    """
    comparison = []
    for name, preds in y_pred_dict.items():
        result = three_d_evaluation(y_true, preds, coordinates, name)
        comparison.append(result)
    
    df = pd.DataFrame(comparison)
    print("\n" + "="*60)
    print(df.to_string(index=False))
    print("="*60)
    
    return df
