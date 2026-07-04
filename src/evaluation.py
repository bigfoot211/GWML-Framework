"""
Model Comparison Protocol — Three-Dimensional Evaluation Framework
Following the GeoRRDI framework (2025)

| Dimension         | Metric                  | Purpose              |
|-------------------|-------------------------|----------------------|
| Accuracy          | RMSE, R²               | Overall fitting performance |
| Spatial Autocorrelation  | Moran's I of residuals | Residual spatial pattern detection |
| Ordinal Consistency   | Kappa coefficient       | Classification consistency measurement |

Citation: Results Section 4.2.1
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from pysal.lib import weights
from pysal.explore.esda import Moran
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import cohen_kappa_score


def three_dimensional_evaluation(y_true, y_pred, coordinates, model_name="GWRF"):
    """
    Three-dimensional model evaluation: Accuracy + Spatial Autocorrelation + Ordinal Consistency
    
    Citation: GeoRRDI Framework (2025)
    """
    
    # Dimension 1: Prediction Accuracy
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Dimension 2: Spatial Autocorrelation of Model Residuals
    residuals = y_true - y_pred
    spatial_weights = weights.KNN.from_array(coordinates, k=5)
    spatial_weights.transform = 'r'
    moran_test = Moran(residuals, spatial_weights)
    morans_i_value = moran_test.I
    
    # Dimension 3: Ordinal Consistency (Cohen's Kappa)
    # Discretize continuous values into ordered quantile categories for Kappa calculation
    y_true_categories = pd.qcut(y_true, q=4, labels=[1, 2, 3, 4])
    y_pred_categories = pd.qcut(y_pred, q=4, labels=[1, 2, 3, 4])
    kappa_score = cohen_kappa_score(y_true_categories, y_pred_categories)
    
    evaluation_output = {
        "model": model_name,
        "R2": round(r2, 3),
        "RMSE": round(rmse, 3),
        "Moran's_I": round(morans_i_value, 3),
        "Kappa": round(kappa_score, 3)
    }
    
    print(f"\n📊 {model_name} Model Evaluation Metrics:")
    print(f"   R² = {evaluation_output['R2']} | RMSE = {evaluation_output['RMSE']}")
    print(f"   Moran's I = {evaluation_output['Moran's_I']} | Kappa = {evaluation_output['Kappa']}")
    
    return evaluation_output


def compare_all_candidate_models(y_true, prediction_dictionary, coordinate_data):
    """
    Comparative evaluation for multiple spatial models: OLS, GWR, RF, Geodetector, GWRF, GNID
    
    Citation: Results Table 4.2.1
    | Model    | R²    | RMSE  | Moran's I | Kappa |
    | OLS      | 0.543 | 1.892 | 0.312     | 0.456 |
    | GWR      | 0.671 | 1.523 | 0.187     | 0.589 |
    | RF       | 0.734 | 1.387 | N/A       | 0.634 |
    | Geodetect| 0.612 | —     | —         | 0.521 |
    | GWRF     | 0.812 | 1.156 | 0.043     | 0.789 |
    | GNID     | 0.778 | 1.234 | 0.067     | 0.745 |
    """
    comparison_record_list = []
    for model_label, predicted_values in prediction_dictionary.items():
        single_model_result = three_dimensional_evaluation(y_true, predicted_values, coordinate_data, model_label)
        comparison_record_list.append(single_model_result)
    
    comparison_dataframe = pd.DataFrame(comparison_record_list)
    print("\n" + "="*60)
    print(comparison_dataframe.to_string(index=False))
    print("="*60)
    
    return comparison_dataframe