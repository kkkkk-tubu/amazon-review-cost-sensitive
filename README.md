# Amazon 评论情感分类 · 类别不平衡与代价敏感学习

在 Amazon Fine Food Reviews 上做二分类情感分析，重点考察**类别不平衡**问题下
代价敏感学习（加权交叉熵 / Focal Loss）是否真的有效。

《信息检索与数据挖掘》课程实践项目（4 人小组）· 本人负责「类别不平衡与代价敏感学习」
方向，并承担全组共享的数据清洗、划分表与评价协议。

## 项目简介

数据集正负类比例 5.38 : 1，负面样本只占 15.68%。多数类基线直接全判正面就有
Accuracy 0.8432，但负类 F1 是 0 —— 也就是一条负面都捞不回来。这正是代价敏感学习
声称要解决的问题。

本项目把这个方向做完整：不调用现成的损失实现，从损失函数与梯度开始自己写，
再系统地与基线、阈值调整、重采样、跨算法做对照。

**主要结论是一个负面结果，但它有明确价值**：允许各配置在验证集上自选判决阈值之后，
**普通的交叉熵基线反而最好**（macro F1 0.8962），全部加权与 Focal 变体都没追上。
这与 Elkan (2001) 关于代价敏感学习的分析一致——改变训练的样本平衡对分类器本身影响很小，
正确做法是在训练后按概率显式确定决策阈值。

## 数据集

[Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)（Kaggle，公开数据集）

| 环节 | 数量 |
|---|---|
| 原始记录 | 568,454 条 |
| 去重（`UserId + ProfileName + Time + Text`） | 剔除 174,521 条（30.7%） |
| 清洗后记录 | 393,932 条 |
| 二分类建模集 | **364,163** 条（正类 307,055 / 负类 57,108） |
| 中性（Score=3） | 29,769 条，剔除并单独保留 |

几点处理上的取舍：

- 该数据集 30.7% 的记录是「同一用户、同一时刻、同一段文本」的重复爬取。
  不去重的话同一段文本会同时落进训练集与测试集，造成严重信息泄漏。
- 去重口径做过敏感性验证：换成 `UserId + Time + Text` 多删 41 条，最终样本量只差
  0.01%，说明结论对键的选取不敏感。
- 残留的跨用户重复文本（743 组）经核查是真实的长文信息分享，不删除，
  改用 `dup_group`（规范化文本 MD5 前 10 位）标记，切分时保证同组不跨集合。
- 标签来自星级而非人工标注：`Score ≤ 2 → 0`、`Score ≥ 4 → 1`。
  星级与文本不一致的噪声无法靠损失函数的重加权消除，这是精度的天然上限。

划分 70/15/15，按 `dup_group` 分组，三个集合的正负比一致，随机种子 42。

## 方法

### 自己实现的部分

模型是自己写的线性分类器，**没有调用任何现成的优化器或损失实现**
（`scripts/07_focal_loss.py`）：

| 模块 | 说明 |
|---|---|
| `loss_grad(z, y, alpha, gamma)` | 统一实现 CE / 加权 CE / Focal Loss 及其梯度 |
| `class Adam` | 一阶、二阶动量与偏差修正逐行写出 |
| `CostSensitiveLinear.fit` | 小批量切片、稀疏梯度、偏置更新 |

损失统一形式（$\gamma=0$ 退化为加权交叉熵，$\alpha_t \equiv 1$ 且 $\gamma=0$ 退化为标准交叉熵）：

$$L = -\alpha_t (1-p_t)^{\gamma} \log(p_t)$$

对 logit 的梯度（$\partial p_t/\partial z = (2y-1)\,p(1-p)$）：

$$\frac{\partial L}{\partial z} = \left[\alpha_t \gamma (1-p_t)^{\gamma-1}\log(p_t) - \alpha_t\frac{(1-p_t)^{\gamma}}{p_t}\right](2y-1)\,p(1-p)$$

$\gamma=0$、$\alpha_t=1$ 时退化为 $p-y$，与标准 BCE 完全一致。

### 实验设置

- 特征：TF-IDF（1–2gram，10 万维，`min_df=2`，`sublinear_tf`），**只在训练集拟合**
- 优化：Adam，`lr=0.05`，`batch=8192`，20 epoch，`l2=1e-6`
- 指标：Accuracy、正负类 P/R/F1、macro F1、混淆矩阵、训练与预测耗时
- 单次训练 3–4 秒，预测约 0.003 秒

## 主要结果

### 1. 阈值调整比加权损失更划算

固定不带权重的模型，只在**验证集**上挑阈值，再用于测试集：

| 做法 | Accuracy | F1(负) | macro F1 |
|---|---|---|---|
| 默认阈值 0.5 | 0.9406 | 0.7892 | 0.8773 |
| `class_weight='balanced'` | 0.9265 | 0.7933 | 0.8743 |
| **验证集选阈值（t=0.65）** | **0.9429** | **0.8172** | **0.8917** |

三项指标全面超过按类别频率加权的做法。加权把负类召回推高 19.0 个百分点，
但精度同时跌 18.0 个百分点，净收益为负。

### 2. 自实现的加权 CE 与 Focal Loss（阈值固定 0.5）

| 配置 | Accuracy | F1(负) | macro F1 |
|---|---|---|---|
| FL γ=2 无 α（仅调制项） | 0.9451 | 0.8125 | **0.8902** |
| FL α=0.25 γ=2 | 0.9382 | 0.8135 | 0.8882 |
| WCE α=0.25（仅类别加权） | 0.9371 | 0.8101 | 0.8862 |
| CE 基线（自实现） | 0.9432 | 0.8015 | 0.8842 |

固定阈值时调制项确实带来小幅提升。但**各配置在验证集自选阈值后**：

| 配置 | 阈值 | macro F1 |
|---|---|---|
| **CE 基线（自实现）** | 0.65 | **0.8962** |
| FL γ=2 无 α | 0.55 | 0.8959 |
| FL α=0.25 γ=1 | 0.45 | 0.8930 |
| WCE α=0.25 | 0.40 | 0.8895 |

差距基本被抹平，普通交叉熵反超。

### 3. 跨算法对照（检验是否只是逻辑回归的巧合）

| 模型 | 默认 macro F1 | 调阈值后 |
|---|---|---|
| LogReg TF-IDF | 0.8773 | 0.8917 |
| CE 基线（自实现） | 0.8842 | **0.8962** |
| FL γ=2 无 α（自实现） | 0.8902 | 0.8959 |
| LinearSVC | 0.8901 | 0.8925 |
| LinearSVC（balanced） | 0.8828 | 0.8887 |
| ComplementNB | 0.8466 | 0.8669 |
| MultinomialNB | 0.8253 | 0.8669 |

线性 SVM 上加 `class_weight='balanced'` 后 macro F1 同样下降（0.8901 → 0.8828），
与逻辑回归上的观察一致，说明不是某个模型的巧合。

### 4. 重采样对照（ROS / RUS / SMOTE）

| 配置 | 调阈值后 macro F1 |
|---|---|
| 无采样（对照） | 0.8917 |
| 随机过采样 ROS | **0.8945** |
| SMOTE (k=5) | 0.8918 |
| 随机欠采样 RUS | 0.8805 |

ROS 是唯一略微超过基线的，但幅度不到 0.3 个百分点，仍不及 CE 的 0.8962。
**最朴素的 ROS 胜过精心设计的 SMOTE** —— 稀疏 TF-IDF 上的插值语义薄弱，
合成样本是两篇评论词向量的加权平均，语义上往往是拼凑的混合文本，反而引入噪声。

## 消融实验

### α 与 γ 的贡献

| 配置 | α | γ | 阈值0.5 macro F1 | 自选阈值 macro F1 |
|---|---|---|---|---|
| CE 基线 | 无 | 0 | 0.8842 | **0.8962** |
| WCE（仅 α） | 0.25 | 0 | 0.8862 | 0.8895 |
| FL（α+γ） | 0.25 | 2 | 0.8882 | 0.8917 |
| FL（仅 γ） | 无 | 2 | 0.8902 | 0.8959 |

调制项 γ 是主要贡献者，**类别因子 α 的贡献为负**。这与 Focal 原文
「γ=2、α=0.25 最佳」的推荐不一致——原文的 α 用于抵消 1:1000 的前景背景失衡，
本任务的失衡程度远小于此。

### 其他

- **γ 取值**：γ=1 与 γ=2 接近，γ=5 明显变差（0.8833）。γ 过大会过度压制中等难度样本的梯度。
- **先验初始化偏置** $b_0=\log(\pi/(1-\pi))$：macro F1 变化 +0.0006，基本无影响。
- **Class-Balanced Loss (Cui 2019)**：最好的一组 macro F1 0.8940，仍低于基线。
  有效样本数 $E_n=(1-\beta^n)/(1-\beta)$ 在 n 达到万级时已完全饱和，加权项退化——
  该方法面向的是「尾部类只有个位数样本」的长尾情形。
- **多随机种子**：额外两个划分上 CE 均优于 Focal（极差 0.0023 / 0.0032），
  结论不是随机波动造成的。
- **收益随长度的分布**：Focal 的收益集中在 21–50 与 101–200 词段（+0.008 左右），
  但在 400 词以上的长评上明显变差（−0.031）——长评的「难」来自文本内部的混合情感，
  不是类别频率问题。

## 局限性

1. **核心改进没有跑赢基线**，这是本次最主要的结果，如实报告。
2. Focal 论文推荐的最佳设置在本任务上不是最优，α 的贡献为负。
3. Class-Balanced Loss 完全无效（有效样本数在万级样本上饱和）。
4. 只在线性模型 + TF-IDF 上验证（已补充 NB 与线性 SVM 对照），未在神经网络上验证。
5. 已覆盖「改损失」与「改数据」两条路径，但**未尝试集成方法**（EasyEnsemble、
   BalancedRandomForest 等），方法覆盖面有缺口。
6. 标签来自星级而非人工标注，构成不可逾越的误差下限。

## 项目结构

```
.
├── README.md
├── requirements.txt
├── scripts/
│   ├── 00_diagnose.py           # 数据质量诊断
│   ├── 01_data_cleaning.py      # 数据清洗（全组共享）
│   ├── 02_explore_stats.py      # 探索性统计
│   ├── 03_validate_output.py    # 清洗结果校验
│   ├── 04_make_split.py         # 70/15/15 分组划分表（全组共享）
│   ├── 05_baseline.py           # 逻辑回归基线
│   ├── 06_threshold_scan.py     # 阈值扫描
│   ├── 07_focal_loss.py         # ★ 自实现 WCE / Focal Loss + 消融 + 多种子
│   ├── 08_class_balanced.py     # Class-Balanced Loss 对照
│   ├── 09_classic_baselines.py  # NB / 线性 SVM 跨算法对照
│   ├── 10_resampling.py         # ROS / RUS / SMOTE 重采样对照
│   └── eval_utils.py            # 统一评价协议（全组共享）
├── reports/                     # 全部实验报告与运行日志
│   └── figures/                 # 类别分布、阈值曲线等 6 张图
└── data/                        # 数据集（不上传，需自行下载）
```

## 复现方式

数据集需自行下载，解压后把 `Reviews.csv` 放在仓库根目录（脚本用
`Path(__file__).parent.parent` 定位根目录）：

```bash
pip install -r requirements.txt

python scripts/01_data_cleaning.py      # 清洗 -> data/Reviews_clean.csv
python scripts/04_make_split.py         # 划分 -> data/splits.csv
python scripts/05_baseline.py           # 基线
python scripts/06_threshold_scan.py     # 阈值扫描
python scripts/07_focal_loss.py         # 自实现 WCE/FL + 消融 + 多种子
python scripts/08_class_balanced.py     # Class-Balanced Loss 对照
python scripts/09_classic_baselines.py  # 跨算法对照
python scripts/10_resampling.py         # 重采样对照
```

各脚本的输出写到 `reports/`，与仓库中已保存的报告一一对应。

## 技术栈

Python · scikit-learn · NumPy · Pandas · SciPy · Matplotlib

## 参考

1. Lin et al. *Focal Loss for Dense Object Detection*. ICCV 2017.
2. Cui et al. *Class-Balanced Loss Based on Effective Number of Samples*. CVPR 2019.
3. Elkan. *The Foundations of Cost-Sensitive Learning*. IJCAI 2001.
4. McAuley & Leskovec. *From amateurs to connoisseurs: modeling the evolution of user expertise through online reviews*. WWW 2013.（数据集来源）
