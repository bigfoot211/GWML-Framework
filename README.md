# GWML —— 时空地理加权机器学习框架（Python 实现）

> 本程序包依据以下论文所述方法实现：
>
> 1. **张坤，范正阳，邹钰莹，张超正，唐一峰，等.** *多地形下湖南省农村居民点人地耦合分异及情景优化*。
>
> 论文提出并应用了 **GWML（Geographically Weighted Machine Learning）**
> 闭环分析框架，包含三大模块：
>
> | 模块   | 中文名                              | 英文名                         | 功能     | 论文章节 |
> | ------ | ----------------------------------- | ------------------------------ | -------- | -------- |
> | 模块 1 | 时序差分景观指数耦合                | **DIC** (Delta-Index Coupling) | 格局量化 | 2.1      |
> | 模块 2 | 局部因子交互探测 + 地理加权随机森林 | **GNID** + **GWRF**            | 机理解译 | 2.2      |
> | 模块 3 | 机理约束元胞自动机-马尔可夫         | **SA-CA-Markov**               | 情景预测 | 2.3      |

三者构成 **P（Pattern）— M（Mechanism）— P（Prediction）三环闭环**：
格局量化结果作为机理解译的空间分类基础，机理解译输出的多尺度参数与
适宜性/约束表面嵌入情景预测模型，模拟结果再反馈校验格局模块。

---

## 目录

- [1. 功能总览](#1-功能总览)
- [2. 环境要求与安装](#2-环境要求与安装)
- [3. 快速开始（一键演示）](#3-快速开始一键演示)
- [4. 数据准备](#4-数据准备)
- [5. 模块 1：DIC 格局量化](#5-模块-1dic-格局量化)
- [6. 模块 2：GWRF 机理解译](#6-模块-2gwrf-机理解译)
- [7. 模块 2：GNID 交互探测](#7-模块-2gnid-交互探测)
- [8. 模块 3：SA-CA-Markov 情景模拟](#8-模块-3sa-ca-markov-情景模拟)
- [9. 模型评价体系](#9-模型评价体系)
- [10. 命令行工具](#10-命令行工具)
- [11. 输出文件说明](#11-输出文件说明)
- [12. 常见问题（FAQ）](#12-常见问题faq)
- [13. 与论文结果的对应关系](#13-与论文结果的对应关系)
- [14. 引用与致谢](#14-引用与致谢)

---

## 1. 功能总览

```
gwml/
├── spatial_utils.py   空间核函数、带宽寻优、空间交叉验证、Moran's I
├── metrics.py         景观指标（PD / AI / CONN / FRAC）
├── dic.py             模块1：DIC 差分与演化类型划分（规则法 + 聚类法）
├── gwrf.py            模块2：GWRF 地理加权随机森林（单/多尺度带宽）+ GWR 对照
├── gnid.py            模块2：GNID 局地 q 值表面与因子交互探测
├── sa_ca_markov.py    模块3：SA-CA-Markov 情景模拟与回溯检验
├── evaluate.py        R² / RMSE / 残差 Moran's I / Kappa / 空间CV
└── cli.py             命令行入口

examples/
├── make_demo_data.py  合成演示数据生成
└── run_demo.py        端到端演示流水线

tests/
└── test_gwml.py       单元测试（13 项检查）
```

**核心对象一览**

| 类/函数                  | 所在模块             | 说明                     |
| ------------------------ | -------------------- | ------------------------ |
| `DIC`                    | `gwml.dic`           | DIC 差分与演化类型划分   |
| `compute_delta()`        | `gwml.dic`           | 三种差分口径             |
| `classify_by_rules()`    | `gwml.dic`           | 论文 Table 3 规则法分类  |
| `jenks_breaks()`         | `gwml.dic`           | Fisher-Jenks 自然断点    |
| `GWRF`                   | `gwml.gwrf`          | 地理加权随机森林（核心） |
| `GWR`                    | `gwml.gwrf`          | 单带宽 GWR 对照模型      |
| `GNID`                   | `gwml.gnid`          | 局地 q 值与交互探测      |
| `SACAMarkov`             | `gwml.sa_ca_markov`  | 机理约束 CA-Markov       |
| `transition_matrix()`    | `gwml.sa_ca_markov`  | 马尔可夫转移矩阵         |
| `patch_metrics()`        | `gwml.metrics`       | 二值栅格景观指标         |
| `summarize_regression()` | `gwml.evaluate`      | 回归评价汇总             |
| `spatial_cv_score()`     | `gwml.evaluate`      | 空间交叉验证包装         |
| `morans_i()`             | `gwml.spatial_utils` | 残差空间自相关           |

---

## 2. 环境要求与安装

- Python **3.10+**（已在 3.14 上验证）
- 核心依赖（见 `requirements.txt`）：

```
numpy>=1.24
scipy>=1.10
pandas>=1.5
scikit-learn>=1.2
matplotlib>=3.6      # 可选，演示与绘图
```

- 可选依赖（按需安装）：
  - `shap>=0.42`：GWRF 局部 SHAP 重要性（`local_importance(method='shap')`）
  - `geopandas>=0.14`、`rasterio>=1.3`：面矢量栅格化（`rasterize_polygons`）

安装：

```bash
pip install -r requirements.txt
```

无需安装本包本身：将项目目录加入 `sys.path`（或在工作目录下运行）即可
`import gwml`。

---

## 3. 快速开始（一键演示）

用合成数据跑通全部三大模块：

```bash
python examples/run_demo.py --outdir ./output --n 600 --seed 42
```

或通过命令行入口：

```bash
python -m gwml.cli demo --outdir ./output --n 600
```

运行约 **5–10 分钟**（主要耗时为 GWRF 多尺度带宽寻优与局部重要性；
`--skip-slow` 可跳过局部重要性）。完成后 `./output/` 下生成：

```
01_dic_result.csv                 DIC 差分与演化类型
02_model_comparison.csv           OLS / GWR / RF / GWRF 精度对照
03_gwrf_local_importance.csv      逐位置特征重要性
04_gnid_interaction.csv           局地 q 值与交互类型
05_sim_S1_BAU.npy 等              三情景模拟栅格
06_scenario_summary.csv           情景指标汇总
summary.txt                       运行摘要
```

### 最小示例：10 行代码拟合 GWRF

```python
import numpy as np
from gwml import GWRF
from gwml.evaluate import summarize_regression

# 你的数据：X 因子矩阵 (n,p)，y 目标向量 (n,)，coords 坐标 (n,2)
X = np.random.rand(500, 4)
y = X @ np.array([1.5, -0.8, 2.0, 0.5]) + np.random.randn(500) * 0.3
coords = np.random.rand(500, 2) * 100

model = GWRF(multiscale=True, verbose=1)   # 多尺度带宽
model.fit(X, y, coords)

print("逐变量最优带宽:", model.bandwidth_)      # 论文式(6)的 b_k
print("空间CV RMSE:", model.cv_rmse_)
print("评价:", summarize_regression(y, model.predict(), coords))
importance = model.local_importance(method="permutation")   # (n,p)
```

---

## 4. 数据准备

### 4.1 DIC 模块输入（指标表）

每个空间单元（村庄/网格）一行，`pandas.DataFrame` 或 CSV，需包含：

| 列名                             | 说明                   |
| -------------------------------- | ---------------------- |
| `id`                             | 单元编号（任意）       |
| `t1_PD, t1_AI, t1_CONN, t1_FRAC` | 时点 t1 的四项景观指标 |
| `t2_PD, t2_AI, t2_CONN, t2_FRAC` | 时点 t2 的四项景观指标 |

景观指标可从二值栅格（居民点=1）计算：

```python
from gwml.metrics import patch_metrics
m = patch_metrics(binary_raster, cell_size=1.0, conn_threshold=2.0)
# 返回 dict: PD, AI, CONN, FRAC, n_patches, ...
```

`conn_threshold` 为图论连通阈值（像元数），对应论文"基于图论的连通度
（Probability of Connectivity / IIC）"思想。面矢量输入可先经
`rasterize_polygons()` 转栅格（需 rasterio）。

### 4.2 GWRF / GNID 模块输入（样本表）

| 列                   | 说明                                              |
| -------------------- | ------------------------------------------------- |
| `x, y`（或 lon/lat） | 单元空间坐标（单位与带宽一致，如 km）             |
| 因子列 × p           | 驱动因子（连续变量，如高程、路网密度、人均GDP……） |
| 目标列 y             | 连续目标（如居民点面积变化率、ΔCONN）             |

建议：使用前对因子做标准化/归一化（量纲不影响 RF，但影响带宽可解释性
与 GNID 分层）。

### 4.3 SA-CA-Markov 模块输入（栅格）

- `landuse_t1.npy / landuse_t2.npy`：两期土地利用栅格（整型地类编码）
- 可选：`elevation / slope / gdp` 等栅格（情景约束面）
- 可选：GWRF 输出的适宜性表面 `suitability_{class}.npy`、GNID 约束面

---

## 5. 模块 1：DIC 格局量化

### 5.1 原理

静态景观指数只能反映"状态"，无法区分长期稳定与剧烈收缩的村庄。DIC
通过耦合景观指标与其时间变化量，刻画"轨迹"：

```
ΔPD   = (PD_t2 − PD_t1) / |PD_t1|       # 斑块密度变化率
ΔAI   = (AI_t2 − AI_t1) / |AI_t1|       # 集聚指数变化率
ΔCONN = (CONN_t2 − CONN_t1) / |CONN_t1| # 连通度变化率
ΔFRAC = (FRAC_t2 − FRAC_t1) / |FRAC_t1| # 平均分维数变化率
```

（`compute_delta` 还支持 `'absolute'` 绝对差与 `'annualized'` 年均变化率
两种口径。）

### 5.2 演化类型（论文 Table 3）

| 类型        | ΔPD  | ΔAI  | ΔCONN | 解释                     |
| ----------- | ---- | ---- | ----- | ------------------------ |
| T1 集聚强化 | ↓↓   | ↑↑   | ↑↑    | 斑块持续整合、连通性提升 |
| T2 破碎主导 | ↑↑   | ↓↓   | ↓↓    | 斑块持续裂解、隔离加剧   |
| T3 扩张驱动 | ↑    | ↓    | ↑     | 用地持续向外扩张         |
| T4 稳定均衡 | ≈0   | ≈0   | ≈0    | 形态基本稳定             |

### 5.3 使用

```python
from gwml import DIC

dic = DIC(mode="relative", classify_method="rules",   # 或 'cluster'
          pd_thr=0.1, ai_thr=0.05, conn_thr=0.05)
out = dic.fit_transform(df, t1_prefix="t1_", t2_prefix="t2_")
# out 新增列: dPD, dAI, dCONN, dFRAC, type, type_code
```

| 参数                         | 默认              | 说明                                             |
| ---------------------------- | ----------------- | ------------------------------------------------ |
| `mode`                       | `'relative'`      | 差分口径                                         |
| `classify_method`            | `'rules'`         | `'rules'`（论文规则）/ `'cluster'`（聚类）       |
| `pd_thr / ai_thr / conn_thr` | 0.1 / 0.05 / 0.05 | 显著变化阈值                                     |
| `strict`                     | `False`           | 规则是否完全命中（False 时按最近模板匹配，稳健） |
| `n_clusters`                 | 4                 | 聚类法类别数                                     |

> 论文还提到用 **Jenks 自然断点**聚类：`classify_cluster(..., method='jenks')`
> 或 `'kmeans'` 可用；`jenks_breaks()` 提供一维断点实现。

---

## 6. 模块 2：GWRF 机理解译

### 6.1 原理

标准 RF 无空间信息；GWRF 在每个空间位置 u 上，用空间核（论文默认高斯核）
对样本加权，独立训练局部随机森林：

```
w_ij = exp(-0.5 * (d_ij / b)²)          # 论文式(6)，高斯核
ŷ(u) = f_u(X(u))                        # 位置 u 的局部 RF 预测
```

**多尺度突破（MGWR 思想）**：每个驱动变量拥有独立最优带宽 b_k，
通过**黄金分割搜索**（小样本）或**等间隔搜索**（大样本）迭代寻优。
本实现以**坐标下降**方式逐变量寻优（每次固定其余变量带宽），
权重合并支持 `mean`（默认）与 `product` 两种方式。

### 6.2 参数

| 参数               | 默认         | 说明                                        |
| ------------------ | ------------ | ------------------------------------------- |
| `base_estimator`   | RF(100树)    | 任意支持 `sample_weight` 的回归器           |
| `kernel`           | `'gaussian'` | `'gaussian'` / `'bisquare'`                 |
| `bandwidth`        | `None`       | 指定带宽；None 自动寻优                     |
| `multiscale`       | `False`      | True 时逐变量独立带宽                       |
| `bandwidth_search` | `'golden'`   | `'golden'` / `'grid'`（大样本）             |
| `grid_n`           | 20           | grid 候选数                                 |
| `max_iter`         | 10           | 多尺度坐标下降轮次                          |
| `cv_folds`         | 5            | 空间交叉验证折数（论文 5 折）               |
| `cv_strategy`      | `'spatial'`  | `'spatial'`（KMeans 分块）/ `'random'`      |
| `max_focal`        | `None`       | 带宽寻优每折抽样焦点数（大样本加速，如 24） |
| `n_jobs`           | `-1`         | 局部模型并行数                              |

### 6.3 使用

```python
from gwml import GWRF
from gwml.evaluate import summarize_regression

model = GWRF(multiscale=True, bandwidth_search="grid", grid_n=6,
             max_iter=2, cv_folds=5, max_focal=24, n_jobs=-1, verbose=1)
model.fit(X, y, coords)

y_hat   = model.predict()                    # 逐位置预测
bw      = model.bandwidth_                   # 逐变量最优带宽
importance = model.local_importance()        # (n,p) 置换重要性
print(summarize_regression(y, y_hat, coords))  # R²/RMSE/残差Moran's I
```

**局部重要性**：

- `method='permutation'`：局部模型下置换某变量后的加权 MSE 损失增量
  （默认，无需额外依赖）；
- `method='shap'`：逐位置 SHAP 值（需 `pip install shap`），
  输出空间显式的特征重要性图（论文 4.2.3 节）。

> 论文 4.2.2 节结果示例：路网密度带宽 4.2 km（局地）、人均 GDP 47.6 km
> （市域）、平均高程 128.3 km（省域）。本实现会给出类似的逐变量带宽，
> 用于证明多尺度效应的存在。

### 6.4 对照模型

```python
from gwml import GWR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# OLS / RF 用 sklearn 即可；GWR 用本包：
gwr = GWR(kernel="gaussian")          # 单带宽，自动寻优
gwr.fit(X, y, coords)
```

---

## 7. 模块 2：GNID 交互探测

### 7.1 原理

经典地理探测器输出全局 q 值；GNID 在任意位置 u 上计算**局地 q 值表面**：

```
q(u) = 1 − Σ_h [ N_h(u) · σ²_h(u) ] / [ N(u) · σ²(u) ]
```

其中 h 为因子分层（连续因子先做分位数分层，默认 L=5 层）；N_h(u)、σ²_h(u)
为 u 处以带宽 b 的空间核窗口内第 h 层的有效样本量与加权方差；
分母可选 `'global'`（论文口径：全域总方差）或 `'local'`（窗口内方差）。

**交互作用类型**（地理探测器规则，逐位置判定 q(X1∩X2) 与 q1、q2 的关系）：

| 类型                              | 条件              |
| --------------------------------- | ----------------- |
| 非线性减弱 nonlinear_weakening    | q12 < min(q1, q2) |
| 单因子增强 univariate_enhancement | min ≤ q12 < max   |
| 双因子增强 bivariate_enhancement  | max ≤ q12 < q1+q2 |
| 非线性增强 nonlinear_enhancement  | q12 ≥ q1+q2       |
| 相互独立 independent              | q12 ≈ q1 ≈ q2     |

### 7.2 使用

```python
from gwml import GNID

gnid = GNID(bandwidth=None, n_strata=5, denominator="global")
q_gdp = gnid.q_surface(df["gdp"].values, y, coords)          # 单因子 q 表面
res   = gnid.interaction_surface(df["gdp"].values,
                                 df["pop"].values, y, coords)
# res: q1, q2, q12, interaction_type(标签), interaction_code(0-4)
```

| 参数          | 默认         | 说明                                      |
| ------------- | ------------ | ----------------------------------------- |
| `bandwidth`   | None         | 核带宽；None 按 kNN 距离中位数 × 2 启发式 |
| `n_strata`    | 5            | 分层数 L                                  |
| `denominator` | `'global'`   | `'global'`（论文）/ `'local'`             |
| `kernel`      | `'gaussian'` | 核函数                                    |

---

## 8. 模块 3：SA-CA-Markov 情景模拟

### 8.1 原理

传统 CA-Markov 仅按历史转移概率外推；SA-CA-Markov 用 GWML 输出的
机理参数**逐像元校正**转移概率（论文式 (7)）：

```
P'(i→j | u) ∝ P(i→j) × S_j(u) × C_j(u)
```

- P：由历史两期土地利用（t1→t2）统计的转移概率矩阵；
- S_j(u)：GWRF 输出的地类 j 空间适宜性表面；
- C_j(u)：GNID 输出的约束表面（归一化至 [0,1]，可含情景阈值）。

空间分配采用 CA 式"适宜性 × 约束 × 邻域支持"打分（默认 5×5 邻域）。

### 8.2 使用

```python
import numpy as np
from gwml import SACAMarkov, make_ecological_constraint, make_balanced_constraint

model = SACAMarkov(classes=[0, 1, 2], neighborhood=5, alpha=1.0)
model.fit(landuse_t1, landuse_t2)                # 估计转移矩阵

# 回溯检验（论文表5：用 t1、t2 训练并模拟 t2 与实测对比）
retro = model.retrospective_validation(landuse_t1, landuse_t2,
                                       suitability=suitability)
print(retro["kappa"], retro["area_error_pct"])

# 三情景模拟（论文表6）
cons_eco = {1: make_ecological_constraint(elev, slope, 1,
                                          elev_threshold=600,
                                          slope_threshold=25, penalty=0.05)}
cons_bal = {1: np.clip(make_balanced_constraint(elev, slope, gdp, 1), 0, 1)}
sim_bau = model.simulate(landuse_t2, steps=1, suitability=suitability)
sim_eco = model.simulate(landuse_t2, steps=1, suitability=suitability,
                         constraints=cons_eco)
sim_bal = model.simulate(landuse_t2, steps=1, suitability=suitability,
                         constraints=cons_bal)
```

| 参数                        | 默认 | 说明                              |
| --------------------------- | ---- | --------------------------------- |
| `classes`                   | 必填 | 地类编码列表                      |
| `neighborhood`              | 5    | CA 邻域窗口（奇数）               |
| `alpha`                     | 1.0  | 邻域支持系数                      |
| `steps`                     | 1    | 模拟期数                          |
| `suitability / constraints` | None | 地类→栅格的适宜性/约束字典        |
| `demand`                    | None | 目标期末面积；None 由马尔可夫投影 |

**情景锚定需求**：情景模拟时除约束面外，还可对各情景给定不同的期末面积
需求（`demand` 参数），体现"情景锚定"思想——演示中 S2 生态情景将居民点
扩张规模锚定在基准的 70%、S3 均衡情景为 85%，配合约束面形成
"扩张规模 + 空间分配"双维情景差异（对应论文表 6 三情景面积变化
+12.3% / +3.1% / +6.8% 的相对差异模式）。

情景约束面构造函数：

- `make_ecological_constraint(elev, slope, class_target, ...)`：
  高海拔/陡坡像元惩罚（生态保护情景）；
- `make_balanced_constraint(elev, slope, gdp, class_target, ...)`：
  经济适宜区放宽 + 生态敏感区约束（均衡发展情景）。

---

## 9. 模型评价体系

对应论文 2.4 节与表 4 的三维评价体系：

| 维度     | 指标           | 函数                     | 说明                  |
| -------- | -------------- | ------------------------ | --------------------- |
| 拟合精度 | R² / 调整后 R² | `r2()` / `adjusted_r2()` | 数值拟合              |
| 拟合精度 | RMSE           | `rmse()`                 | 数值误差              |
| 空间相关 | 残差 Moran's I | `morans_i()`             | ≈0 说明空间结构被捕获 |
| 格局一致 | Kappa          | `kappa()`                | 分类/格局匹配度       |

```python
from gwml.evaluate import summarize_regression, kappa
met = summarize_regression(y, y_hat, coords, n_params=p+1)
print(met["R2"], met["RMSE"], met["residual_Morans_I"])
print(kappa(obs_grid, sim_grid))
```

**空间交叉验证**（论文采用 5 折空间交叉验证避免空间自相关过拟合）：

```python
from gwml.evaluate import spatial_cv_score
res = spatial_cv_score(lambda: GWRF(n_jobs=1, bandwidth=16.0),
                       X, y, coords, n_folds=5)
print(res["mean_rmse"], res["mean_r2"])
```

`spatial_kfold_indices(coords, 5)` 按坐标 KMeans 分块，保证训练/测试空间分离。

---

## 10. 命令行工具

```
python -m gwml.cli <demo|dic|gwrf|gnid> [参数]
```

| 子命令 | 用途            | 示例                                                         |
| ------ | --------------- | ------------------------------------------------------------ |
| `demo` | 端到端演示      | `python -m gwml.cli demo --outdir ./output --n 600`          |
| `dic`  | DIC 差分与分类  | `python -m gwml.cli dic --csv m.csv --t1-prefix t1_ --t2-prefix t2_ --output r.csv` |
| `gwrf` | GWRF 拟合与对照 | `python -m gwml.cli gwrf --csv s.csv --lon x --lat y --target y --features f1 --features f2 --multiscale --benchmark --outdir ./o` |
| `gnid` | GNID 交互探测   | `python -m gwml.cli gnid --csv s.csv --lon x --lat y --target y --features f1 --features f2 --output r.csv` |

各子命令详细参数用 `--help` 查看。

---

## 11. 输出文件说明

| 文件                      | 内容                                              |
| ------------------------- | ------------------------------------------------- |
| `*_dic_result.csv`        | 原指标表 + ΔPD/ΔAI/ΔCONN/ΔFRAC + type/type_code   |
| `*_model_comparison.csv`  | 各模型 R²/RMSE/残差 Moran's I（OLS/GWR/RF/GWRF…） |
| `*_local_importance.csv`  | (n×p) 逐位置归一化特征重要性                      |
| `*_gnid_interaction.csv`  | 逐位置 q1/q2/q12/交互类型                         |
| `05_sim_*.npy`            | 各情景期末模拟栅格（numpy 数组）                  |
| `06_scenario_summary.csv` | 各情景面积变化                                    |
| `summary.txt`             | 演示运行摘要                                      |

---

## 12. 常见问题（FAQ）

**Q1：GWRF 拟合很慢，怎么办？**

- 减小 `base_estimator` 的 `n_estimators`（如 60~100 即可）；
- 设置 `max_focal=24~40`（带宽寻优抽样加速）；
- `bandwidth_search='grid'` + 较小 `grid_n`；大样本直接 `max_iter=1`；
- 先用单带宽（`multiscale=False`）验证，再开多尺度。

**Q2：为什么带宽寻优用网格而不是黄金分割？**
论文指出大样本用等间隔搜索、小样本用黄金分割。本实现两者都支持：
`bandwidth_search='golden'`（小样本，n≲2000）或 `'grid'`（大样本）。

**Q3：GWRF 与普通 RF 有何区别？**
RF 给出全局重要性；GWRF 在每个位置拟合局部模型，输出逐位置预测、
逐变量最优带宽与逐位置重要性，同时处理**非线性 + 空间非平稳 + 多尺度**。

**Q4：GNID 的 q 值如何解读？**
q∈[0,1]，越大说明该因子在位置 u 对目标变量分异的解释力越强。
交互类型按 q(X1∩X2) 与 q1、q2 的相对大小判定（见 7.1 表）。

**Q5：SA-CA-Markov 与普通 CA-Markov 有何区别？**
普通 CA-Markov 仅外推历史转移；SA-CA-Markov 用 GWRF 适宜性面与 GNID
约束面逐像元校正转移概率，实现"机理引导、数据支撑"的情景模拟。

**Q6：`patch_metrics` 的 CONN 是什么定义？**
斑块为节点、质心距离 ≤ `conn_threshold` 的斑块视为连通，
CONN = 最大连通成分面积 / 地类总面积 ∈ [0,1]。这是论文
"基于图论的连通度指标"的实用实现。

**Q7：可以用自己的面矢量（GeoJSON/Shapefile）吗？**
可以：`rasterize_polygons(gdf, ...)`（需 rasterio）转为栅格后，
用 `patch_metrics` 计算指标；或直接在样本表中提供指标列给 DIC。

**Q8：报 `ModuleNotFoundError: No module named 'examples'`？**
从项目根目录运行，或先 `sys.path.insert(0, <项目根目录>)`。

---

## 13. 与论文结果的对应关系

以下为论文报告的关键数值（供比对参考，本包演示数据为合成数据，
数值会不同）：

| 指标                 | 论文值（表3）                                             |
| -------------------- | --------------------------------------------------------- |
| GWRF 调整后 R²       | 0.812                                                     |
| GWRF RMSE            | 1.156                                                     |
| GWRF 残差 Moran's I  | 0.043                                                     |
| GWRF Kappa           | 0.789                                                     |
| 对照：OLS / GWR / RF | 0.543 / 0.671 / 0.734（R²）                               |
| 逐变量带宽           | 路网 4.2km，到县中心 12.8km，人均GDP 47.6km，高程 128.3km |
| SA-CA-Markov 回溯    | 面积误差 0.54%，Kappa 0.71                                |
| 三情景 2035          | BAU +12.3%，生态 +3.1%，均衡 +6.8%                        |

> 论文 GitHub 仓库：https://github.com/bigfoot211/GWML-Framework

---

## 14. 引用与致谢

如使用本程序包开展研究，请引用：

方法学基础（部分）：Fotheringham et al. (2017) MGWR；Georganos et al. (2021) GWRF；王劲峰等 (2017) 地理探测器；Ghosh et al. (2017) CA-Markov 综述；Lundberg & Lee (2017) SHAP。

本实现为论文方法的独立复现，供科研与教学使用；合成演示数据不代表真实区域数据，应用于实际问题时请使用真实数据并注意数据质量与空间尺度。
