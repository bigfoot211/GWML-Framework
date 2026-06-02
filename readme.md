# GWML Framework — Complete Code & Data Preprocessing Scripts

以下代码基于参考资料中描述的完整GWML框架实现，包含三大模块：**DIC（Pattern）**、**GNID+GWRF（Mechanism）**、**SA-CA-Markov（Prediction）**，以及数据预处理流程。

## 📁 项目目录结构

GWML_Framework/
├── data/
│   ├── raw/                    # 原始数据
│   ├── processed/              # 预处理后数据
│   └── indicators/             # 19-indicator system
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # 数据预处理 + dasymetric mapping
│   ├── module1_dic.py          # Delta-Index Coupling
│   ├── module2_mechanism.py    # GNID + GWRF + SHAP
│   ├── module3_prediction.py   # SA-CA-Markov
│   └── evaluation.py           # 模型对比评估
├── config.yaml
├── main.py
└── requirements.txt

## 📊 关键输出对照表

| 模块         | 核心输出                  | 关键数值                                  |
| ------------ | ------------------------- | ----------------------------------------- |
| DIC          | 4类演化分区               | T1:27.4%, T2:20.1%, T3:37.4%, T4:15.1%    |
| GWRF         | R²=0.812, Moran's I=0.043 | 优于OLS(0.543), GWR(0.671), RF(0.734)     |
| MGWR         | 多尺度带宽                | Road:4.2km, GDP:47.6km, Elevation:128.3km |
| SHAP         | 空间特征重要性            | CZT:GDP(0.31), Xiangxi:Elevation(0.38)    |
| SA-CA-Markov | 2035情景预测              | S3: +6.8%面积, +0.11连通性, -4.3%人口密度 |

**使用方式**: 将以上代码保存到对应文件后，运行 `python main.py` 即可执行完整GWML流程。数据文件需按 `config.yaml` 路径放置。

