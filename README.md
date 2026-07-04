# GWML Framework — Full Source Code \& Data Preprocessing Scripts

All scripts below implement the integrated GWML framework described in reference literature, consisting of three core functional modules: **DIC \(Pattern Quantification\)**, **GNID \+ GWRF \(Mechanism Decoding\)**, and **SA\-CA\-Markov \(Scenario Forecasting\)**, alongside a full geospatial data preprocessing pipeline\.

## 📁 Project Directory Structure

```Plain Text
GWML_Framework/
├── data/
│   ├── raw/                    # Raw unprocessed input datasets
│   ├── processed/              # Cleaned geospatial datasets after preprocessing
│   └── indicators/             # Storage for the 19-dimensional integrated indicator system
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # Data cleaning workflow + dasymetric spatial disaggregation
│   ├── module1_dic.py          # Delta-Index Coupling pattern analysis module
│   ├── module2_mechanism.py    # GNID detector, GWRF model and SHAP interpretability tools
│   ├── module3_prediction.py   # Scenario-Anchored CA-Markov land use simulation
│   └── evaluation.py           # Multi-model quantitative benchmark evaluation
├── config.yaml
├── main.py
└── requirements.txt
```

## 📊 Core Output Summary Table

|Module|Core Output Deliverables|Key Quantitative Results|
|---|---|---|
|DIC|Four rural settlement evolution zones|T1:27\.4%, T2:20\.1%, T3:37\.4%, T4:15\.1%|
|GWRF|Model fitting performance metrics|R²=0\.812, Moran's I=0\.043; outperforms OLS\(0\.543\), GWR\(0\.671\), RF\(0\.734\)|
|MGWR|Multi\-scale spatial bandwidth parameters|Road network:4\.2km, GDP factor:47\.6km, Elevation:128\.3km|
|SHAP|Spatially heterogeneous feature importance|CZT urban cluster:GDP \(0\.31\); Xiangxi mountainous zone:Elevation \(0\.38\)|
|SA\-CA\-Markov|2035 future land use scenario projections|Balanced Scenario \(S3\): \+6\.8% settlement area, \+0\.11 connectivity change, \-4\.3% population density shift|

## Usage Instructions

Save all provided code files to their corresponding paths in the directory structure, then run the command `python main.py` to execute the complete GWML analytical pipeline\. All input geospatial and tabular datasets must be placed under the file paths defined in `config.yaml`\.

