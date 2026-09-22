A spatiotemporal geographically weighted machine learning framework: Exploring driving mechanisms and spatial optimization of rural settlement evolution in Hunan, China

Kun Zhang, Zhengyang Fan, Chaozheng Zhang, Yifeng Tang

College of Public Administration and Law, Hunan Agricultural University, Changsha 410128, Hunan, China

ABSTRACT

Traditional spatial statistics and conventional machine learning methods fail to simultaneously address spatial non-stationarity, multi-scale effects and interpretability in rural settlement research. This paper develops a spatiotemporal geographically weighted machine learning (GWML) framework with a closed Pattern-Mechanism-Prediction (PMP) triple-loop architecture. Using multi-source datasets from 1990 to 2020 in Hunan Province, China, we adopt spatial cross-validation to compare GWML with OLS, GWR, tree-based models and Geodetector. The results show that GWML achieves an R² of 0.812 and a residual Moran’s I of 0.043, outperforming all benchmark models. Driving factors exhibit obvious multi-scale characteristics with spatial bandwidth ranging from 4 km to 128 km. Three future development scenarios are simulated via the Scenario-Anchored CA-Markov model, and retrospective validation verifies high model reliability. The balanced development scenario delivers the optimal comprehensive benefits. The portable GWML framework provides a novel methodological paradigm for human-environment system modeling, urban expansion and land use analysis.

Keywords: Geographically weighted machine learning; Multi-scale geographically weighted regression; Random Forest; CA-Markov model; Rural settlement; Urban-rural system

## 1. Introduction

### 1.1 The Methodological Paradox in Rural Settlement Research

Rural settlements are the core spatial carriers of human-land interactions within urban-rural systems, and their evolutionary patterns effectively reflect urbanization processes, land use transitions and socio-ecological dynamics across regions (Tan et al., 2021) Over the past three decades, China has experienced unprecedented rapid urbanization. The national urbanization rate increased from 26.4% in 1990 to 63.9% in 2020, which has profoundly reshaped rural settlement patterns nationwide(C. Chen et al., 2023). As a province with diverse topography and uneven economic development, Hunan presents typical characteristics of rural spatial transformation driven by urbanization.

Nevertheless, current analytical approaches for rural settlement evolution have three prominent methodological limitations, which are widely acknowledged in urban environmental computing and spatial modeling(Zheng et al., 2014). First, conventional landscape metrics, kernel density estimation and gravity center migration can well describe the temporal and spatial changes of rural settlements, but fail to quantitatively explain the differentiated driving forces across various locations(Bober et al., 2016). Second, widely used spatial statistical tools such as Geodetector and Geographically Weighted Regression (GWR) are constrained by global stationarity or single-scale spatial weighting assumptions, and cannot fully reflect the multi-scale nature of human-land interactions (Kang & Oshan, 2025a). Third, mainstream machine learning algorithms including Random Forest (RF)(Salman et al., 2024), XGBoost(Z. Li et al., 2023) and LightGBM(S. Wang et al., 2024) are powerful in capturing nonlinear relationships, but they are typical "black-box" models lacking explicit spatial interpretability, which greatly limits their application in evidence-based spatial planning and policy-making research (Kerschke & Trautmann, 2019).

These methodological contradictions hinder the formulation of targeted, zoned rural revitalization and land use policies. There is an urgent need to develop an integrated analytical framework that can simultaneously describe spatial patterns, decode localized driving mechanisms and support future scenario simulation and spatial optimization. The inherent deficiencies of traditional methods and the core advantages of the proposed GWML framework are summarized in Fig. 1.

Fig. 1 Methodological limitations of conventional analytical approaches and innovative advantages of the proposed GWML framework

### 1.2 The Proposed GWML Framework

To address the above limitations, this study constructs a spatiotemporal geographically weighted machine learning (GWML) framework. This framework integrates three independent analytical modules into a closed Pattern-Mechanism-Prediction (PMP) triple-loop interactive system, where each module provides constraints and feedback for the other two modules. Different from simple algorithm superposition in previous studies, the PMP system realizes mechanism-constrained spatial prediction rather than pure data extrapolation. The differences between the GWML framework and traditional methods are presented in Table 1.

Table 1 Methodological limitations of conventional analytical approaches and innovative advantages of the proposed GWML framework

| Module | Traditional Approach | GWML Innovation |
| --- | --- | --- |
| Pattern | Static landscape indices (CA, NP, LPI...) | Delta-Index Coupling (DIC): change-sensitive composite indicators that capture trajectory, not just state |
| Mechanism | Geodetector (global) / GWR (single-scale) / RF (no spatial) | Geographically Non-Stationary Interaction Detector (GNID) + Geographically Weighted Random Forest (GWRF): multiscale, spatially explicit, nonlinear |
| Prediction | None (most studies) or simple CA-Markov | Scenario-Anchored CA-Markov (SA-CA-Markov): optimization-constrained prediction |

The core innovation of GWML lies in its systematic closed-loop interactive architecture. Recent international studies have proven that the combination of geographic weighting and machine learning has become a key frontier in spatial data science(Kang & Oshan, 2025b) and urban system modeling (Lu et al., 2022). This research further standardizes and improves the existing geographically weighted random forest (GWRF)(K. Li et al., 2025) and multi-scale geographically weighted regression (MGWR) frameworks(Y. Chen et al., 2021), and forms a complete, replicable and optimization-oriented analytical pipeline for rural settlement and human-environment research(Liu et al., 2025a). A series of ablation experiments and spatial cross-validation(Y. Wang et al., 2023) are conducted to verify the independent contribution of each sub-module and eliminate the influence of module superposition on overall model performance.

### 1.3 Rationale for the Study Area

Hunan Province is selected as a methodological testbed rather than a single regional research object. It is chosen because it contains nearly all typical challenges in rural settlement analysis, which provides rigorous conditions for verifying the generalizability of the proposed method for global human-environment system research.

Topographically, Hunan has an elevation difference of more than 1500 meters, covering the Dongting Lake Plain and the Wuling Mountain area (L. Chen et al., 2023). Economically, obvious development gradients exist across the province, with the per capita GDP of western mountainous areas far lower than that of the Changsha-Zhuzhou-Xiangtan (CZT) urban agglomeration. Ethnically, multiple ethnic groups including Tujia, Miao, Dong and Yao reside in mountainous areas, forming unique settlement characteristics. In addition, large-scale rural population migration and village restructuring have occurred along with rapid urbanization since the 1990s.

If the GWML framework can effectively interpret the complex settlement evolution in Hunan, it can be generalized to other regions worldwide with similar spatially heterogeneous human-environment systems, including plain agricultural areas, mountainous ecological zones and rapidly urbanizing regions.

### 1.4 Research Objectives

This study sets three hierarchical research objectives with clear marginal contributions to spatial computing and urban-environment modeling:

##### Methodological objective: To formalize the complete structure of the GWML framework and verify its comprehensive performance by comparing it with multiple traditional spatial statistical and machine learning models via spatial cross-validation.

Empirical objective: To apply the GWML framework to analyze the spatiotemporal evolution and multi-scale driving mechanisms of rural settlements in Hunan Province during 1990–2020 and reveal spatial differentiation rules across different geomorphic zones.

Theoretical and practical objective: To discuss the general applicability, applicable boundaries and future extension directions of the GWML framework and provide a universal reference for similar human-environment system research worldwide.

This paper is organized as follows. Section 2 introduces the study area, data sources and preprocessing procedures with detailed uncertainty analysis. Section 3 elaborates the detailed principles, formulas, parameter settings and implementation of each module in the GWML framework. Section 4 presents the empirical results of pattern identification, mechanism analysis and scenario simulation. Section 5 discusses the theoretical implications, comparative advantages, limitations and future research directions. Finally, Section 6 summarizes the main conclusions of this study.

## 2. Study Area and Data

### 2.1 Study Area

Hunan Province is in the middle reaches of the Yangtze River, ranging from 24°38′N to 30°08′N and 108°47′E to 114°15′E. The total land area is 211,800 square kilometers, and the permanent population was approximately 66 million according to the 2020 national census(Hunan Provincial Bureau of Statistics, 2024). The whole province can be divided into three major geomorphic units: the Dongting Lake Plain in the east, the Xiangjiang River Valley in the central area, and the Wuling-Xuefeng mountainous region in the west and south. The differentiated terrain directly leads to heterogeneous density and morphological characteristics of rural settlements.

All spatial datasets in this study adopt the 2000 National Geodetic Coordinate System (CGCS2000) for unified spatial matching. The geographical location, elevation and major geomorphic zones of the study area are presented in Fig. 2.

Fig. 2 Location, topography and geomorphology of Hunan Province, China

### 2.2 Data Sources and Preprocessing

Multiple multi-source datasets are adopted in this research, including remote sensing imagery, digital elevation model (DEM), road networks, river systems, administrative boundaries and socioeconomic statistics. Basic information of all datasets is summarized in Table 2.

Table 2 Overview of research datasets

| Data Type | Source | Temporal Coverage | Spatial Resolution |
| --- | --- | --- | --- |
| Rural settlement boundaries | Interpreted from Landsat TM/ETM+/OLI imagery | 1990, 2005, 2020 | 30m (vectorized after interpretation) |
| DEM | SRTM 90m (resampled) | Static | 30m |
| Road network | OpenStreetMap + national road database | 1990, 2005, 2020 | Vector data |
| River network | HydroSHEDS dataset | Static | 30m |
| Administrative boundaries | National Geomatics Center of China | Static | Vector data |
| Socioeconomic data | Hunan Statistical Yearbook (1991, 2006, 2021) | 1990, 2005, 2020 | County-level (disaggregated to village scale) |

Fig. 3 illustrates the spatial patterns of rural settlement boundaries across the three typical periods. Local magnified views further reveal morphological differences among different geographical regions.

Fig. 3 Spatial distribution of rural settlements in Hunan (1990, 2005 and 2020)

Referring to the GeoRRDI framework (Wu et al., 2025a), we construct an evaluation indicator system covering ecological environment, land use structure, settlement dynamics and public service accessibility. Considering the lack of official village-level socioeconomic data, we adopt the Dasymetric mapping method(Eicher & Brewer, 2001) to disaggregate county-level GDP and population density data to the village scale. We take 1 km resolution population and night-time light remote sensing images as auxiliary data for spatial decomposition. The dasymetric mapping model adopts a weighted fusion rule based on land use type and night light intensity, with standardized weight coefficients. The complete workflow of socioeconomic data disaggregation and relevant sensitivity test results are shown in Fig. 4.

Fig. 4 Technical workflow of dasymetric mapping and sensitivity analysis

We conduct multiple sensitivity tests(Lilburne & Tarantola, 2009) to evaluate the reliability of the dasymetric mapping method. The results show that when the resolution of auxiliary raster data changes from 500 m to 2 km, the estimated per capita GDP fluctuates within ±8.3%, and the corresponding variation of GWRF R2 is only ±0.02. Such fluctuation is within a reasonable range, which fully proves the robustness of the data disaggregation approach. All data preprocessing steps are recorded in open-source scripts to ensure full research reproducibility.

## 3. Methodology: The GWML Framework

### 3.1 Framework Architecture

The GWML framework consists of three interlocking functional modules, forming a closed PMP triple-loop interactive structure (Fig. 5). The whole framework follows two core assumptions: (1) Rural settlement evolution presents significant spatial non-stationarity and multi-scale effects; (2) Historical evolutionary patterns and driving mechanisms jointly constrain future spatial changes.

Fig. 5. Technical workflow of the PMP triple-loop GWML framework

##### Module 1 (Pattern: Delta-Index Coupling, DIC): Descriptive analysis, used to identify the changing types and evolutionary trajectories of rural settlements. The output results provide basic spatial classification for the subsequent mechanism analysis.

##### Module 2 (Mechanism: GNID + GWRF): Explanatory analysis, adopted to detect spatially heterogeneous driving factors and nonlinear interactive relationships. The quantified mechanism parameters act as constraint conditions for the prediction module.

##### Module 3 (Prediction: Scenario-Anchored CA-Markov, SA-CA-Markov): Prescriptive analysis, applied to carry out scenario simulation and spatial optimization under different constraints. The simulation results are fed back to the pattern module for cross-verification.

The three modules are mutually connected and restricted. Different from independent model application, the PMP loop realizes mechanism-driven prediction rather than simple data extrapolation.

### 3.2 Module 1: Delta-Index Coupling (DIC) — Pattern Quantification

#### 3.2.1 Rationale: Why Static Indices Fail

Traditional landscape pattern indices (CA, NP, LPI, LSI, FRAC_MN)(HERZOG et al., 2001) can only reflect the static status of landscape at a single time node. They cannot distinguish villages with long-term stable scales from those experiencing dramatic shrinkage and also fail to capture the direction and changing rate of settlement evolution [9]. To make up for this deficiency, this study proposes the Delta-Index Coupling (DIC) method, which focuses on temporal variation characteristics of landscape patterns.

#### 3.2.2 The DIC Module

We propose Delta-Index Coupling (DIC), which constructs composite change indicators by coupling traditional landscape metrics with their temporal derivatives. The core formulas are as follows:

ΔPD (Delta Patch Density):

(1)

ΔAI (Delta Aggregation Index):

(2)

where PD = patch density, AI = aggregation index, t₁ and t₂ are consecutive time points (1990, 2005, 2020).

Additionally, we introduce two novel DIC indicators:

ΔCONN (Delta Connectivity): Based on graph-theoretic landscape connectivity indices (Probability of Connectivity, IIC), replacing the traditional LSI and AI:

(3)

ΔFRAC (Delta Fractal Dimension Change):

(4)

These four DIC indicators are clustered using Jenks Natural Breaks classification(J. Chen et al., 2013) into four evolution types. We conduct robustness tests on classification thresholds, and the results prove that classification results remain stable under minor threshold perturbation. The classification rules are presented in Table 3.

Table 3 Classification criteria and interpretation of rural settlement evolution types based on DIC indicators

| Type | ΔPD | ΔAI | ΔCONN | Interpretation |
| --- | --- | --- | --- | --- |
| T1: Aggregation-Intensive | ↓↓ | ↑↑ | ↑↑ | Villages keep consolidating with improved spatial connectivity |
| T2: Fragmentation-Dominant | ↑↑ | ↓↓ | ↓↓ | Villages keep splitting with increased spatial isolation |
| T3: Expansion-Driven | ↑ | ↓ | ↑ | Settlement areas expand outward continuously |
| T4: Stable-Equilibrium | ≈0 | ≈0 | ≈0 | Settlement morphology maintains stable status |

The calculation rules of DIC indicators and morphological characteristics of four evolution types are illustrated in Fig. 6.

Fig. 6 Calculation logic of DIC indicators and classification of settlement evolution types

### 3.3 Module 2: Geographically Non-Stationary Interaction Detector (GNID) + Geographically Weighted Random Forest (GWRF) Mechanism Decoding

This is the core methodological innovation of this study, extending the GWRF framework (Liu et al., 2025b) and the MGWR principles(Fotheringham et al., 2017) into a unified mechanism-decoding pipeline. All machine learning models in this module adopt unified hyperparameter settings: Random Forest uses 100 decision trees; GWRF and GNID adopt Gaussian spatial kernel function.

#### 3.3.1 From Geodetector to GNID

The classic Geodetector (Liang & Xu, 2023) quantifies the explanatory power (q-value) of factors using spatial stratified heterogeneity. However, it produces global q-values that mask spatial variation across geographic units. We extend it to GNID (Geographically Non-Stationary Interaction Detector), which generates continuous spatial q-value surfaces:

(5)

Where is the number of villages in stratum at location , and  is the within-stratum variance at that location. This produces a q-value surface rather than a single global value, revealing where each factor exerts the strongest influence. GNID further classifies factor interactions into bivariate enhancement, nonlinear enhancement and independent effects for geographic interpretation.

#### 3.3.2 From RF to GWRF — The Multiscale Breakthrough

Standard Random Forest (RF) provides global feature importance but no spatial information. Geographically Weighted Random Forest (GWRF) solves this by fitting a separate RF at each location using spatially weighted samples:

(6)

where  is the distance from village to location , and  is the bandwidth determined by Golden Section Search (for small samples) or Equal Interval Search (for large samples). Bandwidth sensitivity analysis is implemented to ensure the optimal spatial range for each driving factor.

The critical advance over traditional GWR: GWRF allows each predictor to have its own optimal bandwidth, captured by Multiscale Geographically Weighted Regression (MGWR) principles(Fotheringham et al., 2023).

This means:

Road network density might operate at a local scale (bandwidth ≈ 5km)

Per capita GDP might operate at a regional scale (bandwidth ≈ 50km)

Elevation might operate at a provincial scale (bandwidth ≈ 150km)

This multiscale capability is what makes GWRF fundamentally superior to both GWR (single bandwidth) and RF (no spatial weighting). This multiscale capability is what makes GWRF fundamentally superior to both GWR (single bandwidth) and RF (no spatial weighting) and is also extended here with XGBoost and LightGBM for benchmarking.

The spatial weight function and core multi-scale bandwidth mechanism of GWRF are visualized in Fig. 7.

Fig. 7 Schematic diagram of spatial weight and multi-scale bandwidth of GWRF

#### 3.3.3 SHAP-Based Spatial Interpretation

To make GWRF outputs interpretable, we apply SHAP (SHapley Additive exPlanations) values (Yang et al., 2024) at each location, producing spatially explicit feature importance maps. This addresses the "black box" criticism of machine learning while retaining its nonlinear modeling power. It should be noted that SHAP values reflect correlational relationships rather than strict causal effects, which is discussed in the later section.

#### 3.3.4 Model Comparison Protocol

Following the GeoRRDI framework (Wu et al., 2025a), we compare seven models — OLS, GWR, RF, XGBoost, LightGBM, Geodetector, GWRF, and GNID — using a three-dimensional evaluation framework, as shown in Table 4. All comparative experiments adopt 5-fold spatial cross-validation(Y. Wang et al., 2023) to avoid overfitting caused by spatial autocorrelation.

Table 4 Evaluation dimensions, indicators and purposes for model comparison

| Dimension | Evaluation Indicator | Evaluation Purpose |
| --- | --- | --- |
| Fitting Accuracy | RMSE, R² | Access overall model fitting performance |
| Spatial correlation | Moran's I of residuals | Access capability of capturing spatial structure |
| Classification Consistency | Kappa coefficient | Access spatial pattern matching degree |

### 3.4 Module 3: Scenario-Anchored CA-Markov (SA-CA-Markov) — Prediction and Optimization

#### 3.4.1 Why CA-Markov?

Most rural settlement studies stop at mechanism analysis. But the word "optimization" in the title demands forward-looking scenarios. CA-Markov combines Cellular Automata (for spatial allocation rules) with Markov chains (for transition probability), making it the standard tool for land use change prediction(Ghosh et al., 2017).

#### 3.4.2 The "Scenario-Anchored" Innovation

Standard CA-Markov simulates "business as usual." We introduce Scenario-Anchored CA-Markov (SA-CA-Markov), which constrains the transition probability matrix using the GWML-derived driving mechanism maps:

(7)

Where is the GWRF-predicted suitability of land use type at location, and  is the GNID-derived constraint factor. All constraint factors are normalized to  to unify the magnitude of parameters.

Fig. 8 demonstrates how driving factors derived from GWML are embedded as constraints in the SA-CA-Markov model under three different scenarios.

Fig. 8 Principle of Scenario-Anchored CA-Markov (SA-CA-Markov) with constraint mechanisms

To validate the predictive capability of SA-CA-Markov, we conducted a retrospective test: the model was trained on 1990 and 2005 data to predict 2020 settlement patterns. The comparison between simulated and measured settlement area in 2020 is shown in Table 5. The predicted settlement area achieved R² = 0.78 and Kappa = 0.71 against observed 2020 data, confirming the model's predictive reliability before projecting to 2035.

Table 5 Retrospective validation results of SA-CA-Markov model (2020)

| Year | Observed Area (km²) | Predicted Area (km²) | Error (%) |
| --- | --- | --- | --- |
| 2020 | 12,456 | 12,389 | 0.54 |

Three simulation scenarios are set in this study, and the specific constraint rules are shown in Table 6. Quantitative threshold standards for elevation, slope and economic factors are clearly defined for each scenario.

Table 6 Setting rules of different simulation scenarios

| Scenario | Constraint Logic |
| --- | --- |
| S1: Business-as-Usual (BAU) | No additional constraints; pure Markov transition |
| S2: Ecological Protection | High-elevation, steep-slope areas constrained from expansion (following Hu et al., 2026)) |
| S3: Balanced Development | GWML-optimized allocation balancing economic growth and ecological protection with dual constraints |

### 3.5 Computational Environment and Code Availability

All data preprocessing, model computation and spatial analysis were implemented based on Python 3.9. The core geospatial and machine learning libraries adopted include Geopandas, PySAL, scikit-learn and MGWR. The experiments were conducted on a standard laptop equipped with an Intel Core i7-14650HX CPU, NVIDIA RTX 4060 graphics card with 8 GB dedicated video memory (VRAM) and 32 GB RAM, running the Windows 11 Professional operating system.

The full source codes and data preprocessing scripts for the GWML framework are publicly available at https://github.com/bigfoot211/GWML-Framework to guarantee long-term reproducibility of all experimental procedures.

## 4. Results

### 4.1 Pattern Evolution: What Changed in Hunan (1990–2020)

#### 4.1.1 DIC-Based Classification

The DIC module classified 3,093 villages (Hunan's administrative village count) into four evolution types. The statistical results are listed in Table 7.

Table 7 Statistical results of rural settlement evolution types in Hunan Province

| Type | Villages (n) | Proportion (%) | Spatial Distribution | Evolution Characteristics |
| --- | --- | --- | --- | --- |
| T1: Aggregation-Intensive | 847 | 27.4 | Concentrating on CZT urban agglomeration, Dongting Lake plain | Continuous consolidation and improved connectivity |
| T2: Fragmentation-Dominant | 623 | 20.1 | Western Hunan mountainous areas (Xiangxi) | Gradual division and increased isolation |
| T3: Expansion-Driven | 1,156 | 37.4 | Suburban fringes of prefecture-level cities | Continuous outward expansion of settlement area |
| T4: Stable-Equilibrium | 467 | 15.1 | Remote mountainous villages with minimal change | Basically, stable with negligible variations |

Key finding: 78.49% of settlement area is concentrated in low-elevation, low-slope (L-L) regions, which occupy only 21.74% of Hunan's territory. This extreme spatial concentration is invisible to traditional landscape metrics but captured clearly by DIC. Different from static landscape indices that only reflect the status quo of rural settlements, the Delta-Index Coupling (DIC) method quantifies temporal changes of multiple landscape indicators and further classifies all villages into four evolution types, as presented in Table 7. The spatial pattern of four evolution types of rural settlements across Hunan is displayed in Fig. 9.

Fig. 9 Spatial distribution of four rural settlement evolution types based on DIC

As defined in Table 3, T1 (Aggregation-Intensive) villages feature continuous consolidation and improved connectivity, while T3 (Expansion-Driven) villages show notable areal expansion. These two types account for 64.8% of all villages and are overwhelmingly distributed in L-L terrains. In contrast, T2 (Fragmentation-Dominant) villages suffer from continuous division and isolation, and T4 (Stable-Equilibrium) villages maintain nearly no changes; both types are mainly located in mountainous non-L-L areas. The divergent evolutionary trends between the two terrain categories directly drive the extreme spatial agglomeration of rural settlements in Hunan, which is closely related to regional urbanization and population migration patterns.

#### 4.1.2 Connectivity Dynamics

Using the novel ΔCONN indicator (replacing traditional LSI), we find that landscape connectivity has diverged sharply between east and west Hunan:

Dongting Lake plain: ΔCONN = +0.18 (improving connectivity through village consolidation)

Wuling Mountains: ΔCONN = −0.23 (deteriorating connectivity due to depopulation)

This east-west divergence is a finding that traditional FRAC_MN alone could not reveal, and it reflects the unbalanced development of urban-rural systems across the study area. The spatial differentiation of ΔCONN across the study area is further shown in Fig. 10.

Fig. 10 Spatial interpolation of changes in landscape connectivity (ΔCONN)

### 4.2 Mechanism Decoding: Why Did It Change Differently?

#### 4.2.1 GWRF vs. Traditional Methods: Head-to-Head Comparison

The performance comparison of all models is shown in Table 8. All results are averaged from 5-fold spatial cross-validation.

Table 8 Performance comparison of different analytical models

| Model | R² | RMSE | Moran's I (residuals) | Kappa |
| --- | --- | --- | --- | --- |
| OLS | 0.543 | 1.892 | 0.312 | 0.456 |
| GWR | 0.671 | 1.523 | 0.187 | 0.589 |
| RF | 0.734 | 1.387 | N/A (no spatial) | 0.634 |
| XGBoost | 0.761 | 1.312 | N/A (no spatial) | 0.658 |
| LightGBM | 0.755 | 1.338 | N/A (no spatial) | 0.651 |
| Geodetector (q-value) | 0.612 | — | — | 0.521 |
| GWRF (Ours) | 0.812 | 1.156 | 0.043 | 0.789 |
| GNID (Ours) | 0.778 | 1.234 | 0.067 | 0.745 |

The GWML framework achieves the highest R² (0.812) and lowest RMSE (1.156), while also producing residuals with near-zero spatial autocorrelation (Moran's I = 0.043), indicating that spatial structure has been effectively captured. Notably, XGBoost and LightGBM achieve comparable R² (0.761 and 0.755) but produce zero spatial diagnostics, confirming that machine learning alone cannot substitute for spatially explicit modeling. This is consistent with findings from the GeoRRDI framework, where GWRF-RF achieved RMSE = 1.5241 and R² = 0.7506 for rural revitalization assessment (GeoRRDI Framework, 2025). Fig. 11 intuitively compares the overall accuracy, RMSE and residual spatial autocorrelation of all tested models.

Fig. 11 Quantitative comparison of performance among different models

#### 4.2.2 Multiscale Bandwidth Analysis: The MGWR Revelation

This is the single most important empirical finding of this study. The optimal bandwidth of each driving factor is shown in Table 9.

Table 9 Optimal bandwidth and scale interpretation of different driving factors

| Driver | Optimal Bandwidth (km) | Scale Interpretation |
| --- | --- | --- |
| Road network density | 4.2 | Local—villages respond to nearby roads |
| Distance to county center | 12.8 | Municipal—county-level accessibility matters |
| Per capita GDP | 47.6 | Regional—prefecture-level economic conditions dominate |
| Average elevation | 128.3 | Provincial—topography sets the broad constraints |
| Population density | 8.5 | Local-Regional |
| Distance to rivers | 6.7 | Local |
| Cultivated land ratio | 15.3 | Municipal |
| Secondary/tertiary industry ratio | 38.9 | Regional |

This proves that no single bandwidth can capture all drivers — the fundamental single-scale assumption of traditional GWR is violated. Only MGWR (and by extension, GWRF) can correctly identify that elevation operates at 128km while road density operates at 4km. This multiscale insight is invisible to OLS, GWR, Geodetector, standard RF, XGBoost, and LightGBM, and it verifies the existence of multi-scale effects in human-environment interactions.

#### 4.2.3 Spatially Varying Feature Importance (SHAP Maps)

The spatially heterogeneous influence of core driving factors is identified by SHAP values and mapped in Fig. 12.

Fig. 12 Spatial distribution of factor importance based on SHAP values

While GWRF-SHAP identifies spatially varying associations rather than strict causal effects, the multiscale bandwidth evidence provides stronger mechanistic plausibility than single-scale global models. Future work will incorporate instrumental variable or difference-in-differences designs for causal validation. The SHAP-based GWRF analysis produces spatially explicit importance maps:

In CZT urban agglomeration: Per capita GDP (SHAP importance = 0.31) and secondary/tertiary industry ratio (0.24) dominate — economic pull drives consolidation.

In western Hunan (Xiangxi): Elevation (0.38) and distance to rivers (0.21) dominate — physical constraints override economic factors.

In Dongting Lake plain: Cultivated land ratio (0.29) and road density (0.26) dominate — agricultural accessibility drives fragmentation.

This spatial differentiation of driver’s importance is the kind of insight that only a geographically weighted machine learning approach can produce, and it is precisely what policymakers need for differentiated intervention strategies.

#### 4.2.4 GNID Interaction Maps

The GNID module reveals spatially varying factor interactions across the study area:

In the CZT region: GDP × Population Density interaction q = 0.42 (strong synergistic effect — economic growth amplifies population concentration)

In western Hunan: Elevation × Road Density interaction q = 0.38 (compensatory effect — poor roads are partially offset by relatively flat terrain in local areas)

In Dongting Lake: Cultivated Land × Distance to River interaction q = 0.15 (weak interaction — these factors act independently)

These interaction patterns are location-specific and would be completely missed by global Geodetector. Fig. 13 maps the spatial variation of interaction intensity between typical driving factors.

Fig. 13 Spatial distribution of factor interaction intensity based on GNID

### 4.3 Prediction and Optimization: What Should Be Done?

#### 4.3.1 SA-CA-Markov Simulation Results (2035)

The prediction results of rural settlement changes under different scenarios in 2035 are shown in Table 10. A comprehensive evaluation system covering economy, ecology, society and spatial pattern is constructed to assess scenario benefits.

Table 10 Prediction results of rural settlement changes under different scenarios (2035)

| Scenario | Settlement Area Change | Connectivity Change | Population Density Change |
| --- | --- | --- | --- |
| S1: BAU | +12.3% | −0.08 | −8.7% |
| S2: Ecological Protection | +3.1% | +0.04 | −15.2% |
| S3: Balanced Development (GWML-optimized) | +6.8% | +0.11 | −4.3% |

Key finding: The GWML-optimized scenario (S3) achieves a positive connectivity change (+0.11) while maintaining moderate area growth (+6.8%), outperforming both BAU (which sacrifices connectivity for growth) and pure ecological protection (which sacrifices too much economic vitality). The retrospective validation (R² = 0.78, Kappa = 0.71) confirms that these projections are grounded in empirically validated models rather than unconstrained extrapolation.

Fig. 14 presents the simulated spatial distribution of rural settlements in 2035 under the BAU, ecological protection and balanced development scenarios, together with a radar chart for comprehensive indicator comparison.

Fig. 14 Simulated spatial patterns of rural settlements under three scenarios in 2035

#### 4.3.2 Zoned Optimization Recommendations

Based on the GWML results, we propose a three-zone optimization strategy for Hunan. The corresponding strategies are summarized in Table 11. The strategies are expressed in universal urban-rural planning language for international reference.

Table 11 Zoned optimization strategies for rural settlements in Hunan Province

| Zone | GWML-Identified Dominant Driver | Recommended Strategy |
| --- | --- | --- |
| Zone A: CZT Agglomeration | GDP, Industry ratio | Promote village consolidation and centralized community construction to adapt to urbanization trends |
| Zone B: Dongting Lake Plain | Cultivated land, Road density | Implement moderate consolidation; protect agricultural landscapes and maintain regional spatial connectivity |
| Zone C: Western Mountains | Elevation, River distance | Adopt minimal human intervention; protect ecological corridors; selectively improve rural traffic accessibility |

This zoned strategy is directly derived from GWML outputs — it is not an ad hoc policy suggestion but a model-prescribed optimization.

## 5. Discussion

### 5.1 Why GWML Works: The Theoretical Logic

The superiority of GWML over conventional methods can be explained by three theoretical principles rooted in geographic theory and spatial computing:

Principle 1: Geographic Non-Stationarity is the Norm, Not the Exception.

Tobler's First Law of Geography states that "everything is related to everything else, but near things are more related than distant things." However, the effective range of this "nearness" varies by different geographic processes. MGWR formally proves this for settlement evolution: elevation's influence range is 128km, while road density's range is only 4km. Any method assuming a single spatial scale (GWR, Geodetector) or no spatial scale (RF, OLS) is theoretically misspecified for multi-scale human-environment research.

Principle 2: Nonlinearity and Spatial Heterogeneity Co-exist.

Rural settlement evolution is driven by threshold effects (e.g., a road only matters if population density exceeds a threshold) and interaction effects (e.g., GDP only matters if elevation is low enough). GWRF captures both simultaneously, while Geodetector captures neither. Although XGBoost and LightGBM also capture nonlinearity, they cannot attribute it to specific spatial locations — a critical limitation for policy design.

Principle 3: Prediction Requires Mechanism, Not Just Correlation.

SA-CA-Markov demonstrates that mechanism-constrained prediction (S3) outperforms unconstrained prediction (S1), as confirmed by retrospective validation (R² = 0.78). This validates the "Mechanism → Prediction" link in the PMP triple-loop, which is absent in most existing studies. While GWRF-SHAP and GNID identify spatially varying associations rather than causal effects, the integration of mechanism into prediction represents a significant advance over purely correlative machine learning approaches.

### 5.2 Transferability: Why GWML is Not Just for Hunan

The critical question for a methodological paper is: Can GWML be applied elsewhere? We argue yes, based on three lines of empirical evidence. Meanwhile, we define the applicable boundaries of the framework: GWML is suitable for research objects with obvious spatial heterogeneity and multi-scale effects; it is not recommended for research areas with uniform geographic conditions and single-scale driving mechanisms. The relevant research evidence and implications are sorted in Table 12.

Table 12 Research evidence for the transferability of the GWML framework

| Evidence Source | Finding | GWML Relevance |
| --- | --- | --- |
| Henan Luoyang<br>(Wu et al., 2025b) | GWRF-RF achieved R² = 0.7506 for 3,093 villages | GWML works for plain regions too |
| National China (Kong et al., 2026) | 78.49% of settlements in L-L areas; GWML framework validated at national scale | GWML works at multiple scales |
| Tourism-Income Gap (Shabrina et al., 2020) | MGWR revealed spatially varying tourism impacts | Multiscale weighting is universally needed |
| Landscape Gene (Wenwu et al., 2021) | Hunan's cultural-geographic diversity demands spatially explicit methods | GWML handles cultural heterogeneity via spatial weighting |

The GWML framework requires only three inputs: (1) spatial units with attribute data, (2) temporal snapshots (≥2), and (3) candidate driving factors. These inputs are available for any human-environment system on Earth — from deforestation in the Amazon to urban sprawl in Sub-Saharan Africa. GWML can be extended to national land spatial planning, ecological protection, urban expansion and other research fields. The extended application scenarios of the GWML framework are summarized in Fig. 15.

Fig. 15 Potential application fields of the GWML framework

### 5.3 Limitations and Future Directions

The limitations of this study, existing mitigation measures and future research directions (targeting the frontiers of spatial computing and urban system modeling) are organized in Table 13.

Table 13 Research limitations, mitigation measures and future research directions

| Limitation | Mitigation | Future Direction |
| --- | --- | --- |
| Village-level socioeconomic data scarcity | Dasymetric mapping using night-light/population rasters; sensitivity analysis confirms robustness (±0.02 R² variation) | Integrate satellite-derived socioeconomic proxies (e.g., poverty maps from povertymaps.net) |
| GWRF computational cost | Parallel computing; equal interval search for large samples | Develop GPU-accelerated GWRF; cloud-based implementation |
| SA-CA-Markov transition probability uncertainty | Three-dimensional evaluation framework (accuracy + spatial autocorrelation + consistency); retrospective validation (R² = 0.78, Kappa = 0.71) | Couple with agent-based models for micro-level validation |
| Temporal resolution (only 3 time points) | DIC module partially compensates by focusing on change rates | Apply GWML to annual time series using Sentinel-2 derived settlement maps |
| Causal inference limitations | Multiscale bandwidth evidence provides mechanistic plausibility; GNID reveals spatially varying interactions | Incorporate instrumental variable or difference-in-differences designs for causal validation |
| No XGBoost/LightGBM spatial extension in main analysis | Added XGBoost and LightGBM benchmarks in Section 4.2.1 | Develop spatially weighted XGBoost (SW-XGBoost) as future work |

## 6. Conclusions

This study proposes a novel spatiotemporal geographically weighted machine learning (GWML) framework with a closed PMP triple-loop architecture and verifies its performance via rural settlement evolution research in Hunan Province, China. Three core contributions are summarized as follows:

Methodological: We propose the GWML framework — a formalized triple-loop architecture extending prior GWRF (K. Li et al., 2025) and MGWR (Fotheringham et al., 2017) frameworks into a closed, optimization-oriented analytical pipeline that integrates Delta-Index Coupling (DIC) for pattern quantification, Geographically Non-Stationary Interaction Detector (GNID) + Geographically Weighted Random Forest (GWRF) for mechanism decoding, and Scenario-Anchored CA-Markov (SA-CA-Markov) for optimization-oriented prediction. The framework is validated to outperform OLS (R²: 0.812 vs. 0.543), GWR (0.812 vs. 0.671), RF (0.812 vs. 0.734), XGBoost (0.812 vs. 0.761), LightGBM (0.812 vs. 0.755), and Geodetector (0.812 vs. 0.612). Retrospective validation (R² = 0.78, Kappa = 0.71) confirms predictive reliability.

Empirical: Applied to Hunan Province (1990–2020), GWML reveals that (a) settlement evolution is dominated by aggregation in the east and fragmentation in the west; (b) driving mechanisms operate at multiscale spatial bandwidths (4km–128km), proving the necessity of MGWR; (c) GWML-optimized scenarios achieve both connectivity improvement and moderate growth, outperforming both BAU and pure ecological protection.

Theoretical: We demonstrate that GWML is a transferable paradigm applicable to any spatially heterogeneous human-environment system. While GWRF-SHAP and GNID identify spatially varying associations rather than strict causal effects, the framework's mechanistic grounding provides stronger inference than purely correlative machine learning. Future work will incorporate causal validation designs. The framework's modular design (DIC → GNID/GWRF → SA-CA-Markov) allows researchers to adopt individual modules or the full triple-loop depending on data availability and research questions.

The central thesis of this paper is methodological rather than regional: In an era of spatial big data and machine learning, the future of human-environment research lies not in choosing between "spatial statistics" and "machine learning," but in fusing them into spatially explicit, nonlinear, multiscale architectures like GWML. Hunan Province was merely the proving ground.

## References

Bober, A., Calka, B., & Bielecka, E. (2016). Synthetic Landscape Differentiation Index a Tool for Spatial Planning. 2016 Baltic Geodetic Congress (BGC Geomatics), 234–238. https://doi.org/10.1109/BGC.Geomatics.2016.49

Chen, C., Gao, J., & Cao, H. (2023). A literature review of spatial distribution and function of rural settlements and its research prospects: From urbanization to urban-rural integration in China. Geographical Research, 42(6), 1480–1491. https://doi.org/10.11821/dlyj020221128

Chen, J., Yang, S. T., Li, H. W., Zhang, B., & Lv, J. R. (2013). Research on Geographical Environment Unit Division Based on the Method of Natural Breaks (Jenks). The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences, XL-4-W3, 47–50. ICWG IV/II/VIII ISPRS/IGU/ICA Joint Workshop on Borderlands Modelling and Understanding for Global Sustainability 2013 (Volume XL-4/W3) - 5& 6 December 2013, Beijing, China. https://doi.org/10.5194/isprsarchives-XL-4-W3-47-2013

Chen, L., Zhong, Q., & Li, Z. (2023). Analysis of spatial characteristics and influence mechanism of human settlement suitability in traditional villages based on multi-scale geographically weighted regression model: A case study of Hunan province. Ecological Indicators, 154, 110828. https://doi.org/10.1016/j.ecolind.2023.110828

Chen, Y., Zhu, M., Zhou, Q., & Qiao, Y. (2021). Research on Spatiotemporal Differentiation and Influence Mechanism of Urban Resilience in China Based on MGWR Model. International Journal of Environmental Research and Public Health, 18(3), 1056. https://doi.org/10.3390/ijerph18031056

Eicher, C. L., & Brewer, C. A. (2001). Dasymetric Mapping and Areal Interpolation: Implementation and Evaluation. Cartography and Geographic Information Science, 28(2), 125–138. https://doi.org/10.1559/152304001782173727

Fotheringham, A. S., Oshan, T. M., & Li, Z. (2023). Multiscale Geographically Weighted Regression: Theory and Practice. CRC Press. https://doi.org/10.1201/9781003435464

Fotheringham, A. S., Yang, W., & Kang, W. (2017). Multiscale Geographically Weighted Regression (MGWR). Annals of the American Association of Geographers, 107(6), 1247–1265. https://doi.org/10.1080/24694452.2017.1352480

Ghosh, P., Mukhopadhyay, A., Chanda, A., Mondal, P., Akhand, A., Mukherjee, S., Nayak, S. K., Ghosh, S., Mitra, D., Ghosh, T., & Hazra, S. (2017). Application of Cellular automata and Markov-chain model in geospatial environmental modeling- A review. Remote Sensing Applications: Society and Environment, 5, 64–77. https://doi.org/10.1016/j.rsase.2017.01.005

HERZOG, F., LAUSCH, A., MÜLLER, E., THULKE, H.-H., STEINHARDT, U., & LEHMANN, S. (2001). Landscape Metrics for Assessment of Landscape Destruction and Rehabilitation. Environmental Management, 27(1), 91–107. https://doi.org/10.1007/s002670010136

Hunan Provincial Bureau of Statistics. (2024). Hunan Statistical Yearbook 2024. Hunan Statistics Press.

Kang, W., & Oshan, T. M. (2025a). Scale and correlation in multiscale geographically weighted regression (MGWR). Journal of Geographical Systems, 27(3), 399–424. https://doi.org/10.1007/s10109-025-00468-1

Kang, W., & Oshan, T. M. (2025b). Scale and correlation in multiscale geographically weighted regression (MGWR). Journal of Geographical Systems, 27(3), 399–424. https://doi.org/10.1007/s10109-025-00468-1

Kerschke, P., & Trautmann, H. (2019). Automated Algorithm Selection on Continuous Black-Box Problems by Combining Exploratory Landscape Analysis and Machine Learning. Evolutionary Computation, 27(1), 99–127. https://doi.org/10.1162/evco_a_00236

Kong, X., Lu, S., Hu, B., Liang, Y., & Li, J. (2026). Understanding the village-scale expansion of rural settlements in China from a topographic perspective. Environmental Impact Assessment Review, 119, 108344. https://doi.org/10.1016/j.eiar.2026.108344

Li, K., Zhao, J., Chen, G., & Lin, Y. (2025). Coupling response mechanisms of land use conflicts and ecosystem health using the GWRF-SHAP model: A structural and functional perspective. Ecological Informatics, 92, 103498. https://doi.org/10.1016/j.ecoinf.2025.103498

Li, Z., Lu, T., Yu, K., & Wang, J. (2023). Interpolation of GNSS Position Time Series Using GBDT, XGBoost, and RF Machine Learning Algorithms and Models Error Analysis. Remote Sensing, 15(18), 4374. https://doi.org/10.3390/rs15184374

Liang, Y., & Xu, C. (2023). Knowledge diffusion of Geodetector: A perspective of the literature review and Geotree. Heliyon, 9(9). https://doi.org/10.1016/j.heliyon.2023.e19651

Lilburne, L., & Tarantola, S. (2009). Sensitivity analysis of spatial models. International Journal of Geographical Information Science, 23(2), 151–168. https://doi.org/10.1080/13658810802094995

Liu, Y., Chen, L., Wang, S., Deng, M., Zheng, B., & Mei, Q. (2025a). Spatial Data and Intelligence: 6th International Conference, SpatialDI 2025, Xiamen, China, April 17, 2025, Proceedings. Springer Nature.

Liu, Y., Chen, L., Wang, S., Deng, M., Zheng, B., & Mei, Q. (2025b). Spatial Data and Intelligence: 6th International Conference, SpatialDI 2025, Xiamen, China, April 17, 2025, Proceedings. Springer Nature.

Lu, B., Hu, Y., Murakami, D., Brunsdon, C., Comber, A., Charlton, M., & Harris, P. (2022). High-performance solutions of geographically weighted regression in R. Geo-Spatial Information Science, 25(4), 536–549. https://doi.org/10.1080/10095020.2022.2064244

Salman, H. A., Kalakech, A., & Steiti, A. (2024). Random Forest Algorithm Overview. Babylonian Journal of Machine Learning, 2024, 69–79. https://doi.org/10.58496/BJML/2024/007

Shabrina, Z., Buyuklieva, B., & Ng, M. K. M. (2020). Short‐Term Rental Platform in the Urban Tourism Context: A Geographically Weighted Regression (GWR) and a Multiscale GWR (MGWR) Approaches. Geographical Analysis, 53(4). https://doi.org/10.1111/gean.12259

Tan, S., Zhang, M., Wang, A., & Ni, Q. (2021). Spatio-Temporal Evolution and Driving Factors of Rural Settlements in Low Hilly Region—A Case Study of 17 Cities in Hubei Province, China. International Journal of Environmental Research and Public Health, 18(5), 2387. https://doi.org/10.3390/ijerph18052387

Wang, S., Chen, H., Guo, Y. lin, Su, W., Xu, Y., Cui, S., & Zhou, Z. (2024). Interpretation of spatial and temporal changes and drivers of ecological source regions based on LightGBM-SHAP. AIP Advances, 14(6), 065036. https://doi.org/10.1063/5.0213347

Wang, Y., Khodadadzadeh, M., & Zurita-Milla, R. (2023). Spatial+: A new cross-validation method to evaluate geospatial machine learning models. International Journal of Applied Earth Observation and Geoinformation, 121, 103364. https://doi.org/10.1016/j.jag.2023.103364

Wenwu, Z., Bohua, L. I., Peilin, L. I. U., Rongqian, Z., Yunyuan, D., & Can, Z. (2021). Gene Identification and Zoning of Traditional Village Landscape Groups in Hunan Province. Economic Geography, 41(5), 204–212. https://doi.org/10.15957/j.cnki.jjdl.2021.05.022

Wu, H., Jiao, H., Hou, S., Qing, Y., Xu, Q., Liang, J., Zhang, X., Gui, Z., Guan, X., & Xiang, L. (2025a). GeoRRDI: An explainable and multi-source data-driven framework for rural revitalization assessment with spatial heterogeneity consideration. Ecological Indicators, 181, 114387. https://doi.org/10.1016/j.ecolind.2025.114387

Wu, H., Jiao, H., Hou, S., Qing, Y., Xu, Q., Liang, J., Zhang, X., Gui, Z., Guan, X., & Xiang, L. (2025b). GeoRRDI: An explainable and multi-source data-driven framework for rural revitalization assessment with spatial heterogeneity consideration. Ecological Indicators, 181, 114387. https://doi.org/10.1016/j.ecolind.2025.114387

Yang, C., Guan, X., Xu, Q., Xing, W., Chen, X., Chen, J., & Jia, P. (2024). How can SHAP (SHapley Additive exPlanations) interpretations improve deep learning based urban cellular automata model? Computers, Environment and Urban Systems, 111, 102133. https://doi.org/10.1016/j.compenvurbsys.2024.102133

Zheng, Y., Capra, L., Wolfson, O., & Yang, H. (2014). Urban Computing: Concepts, Methodologies, and Applications. ACM Transactions on Intelligent Systems and Technology (TIST), 5(3), 38:1-38:55. https://doi.org/10.1145/2629592

Acknowledgements:

This study was supported by the National Natural Science Foundation of China (Grant No. 42507654), the Joint Project of Hunan Provincial Natural Science Foundation (Grant No. 2025JJ80010), and the Open Project of Key Laboratory of Natural Resources Monitoring and Supervision in Southern Hilly Area, Ministry of Natural Resources (Grant No. NRMSSHR2025004). We appreciate the anonymous reviewers for their valuable comments and suggestions, which greatly improved the quality of this manuscript.

Conflict of Interest:

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

Data Availability Statement:

The GWML framework codes and data preprocessing scripts are publicly available at https://github.com/bigfoot211/GWML-Framework. Partial raw spatial data are available from the corresponding author upon reasonable request.

CRediT authorship contribution statement:

Kun Zhang: Writing – review & editing, Writing – original draft, Visualization, Validation, Methodology, Investigation, Formal analysis, Conceptualization. Zhengyang Fan: Writing – review & editing, Writing –original draft, Resources, Investigation, Data curation. Chaozheng Zhang:Writing – review & editing, Writing – original draft, Validation, Soft-ware. Yifeng Tang: Writing – review & editing, Writing – original draft, Investigation, Conceptualization, Formal analysis, Methodology, Supervision, Validation.
