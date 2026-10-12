# 论文阅读笔记（成员 B：类别不平衡与代价敏感学习）

精读 1 篇 + 补充 2 篇。三篇都已核对原文或会议页面，引文信息见下。

---

## 论文一（核心）：Focal Loss for Dense Object Detection

**完整引文**
> Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár. Focal Loss for Dense Object Detection. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 2017, pp. 2980–2988. arXiv:1708.02002.

**研究问题**
一阶段目标检测器（SSD、YOLO）的精度长期落后于两阶段检测器（Faster R-CNN）。作者诊断其根因是训练时的**极端前景/背景类别不平衡**：一张图要枚举约 10 万个候选位置，其中绝大多数是易分的背景，这些易分负样本的损失累加起来淹没了真正有信息量的难例梯度。

**核心方法**
不改动采样策略，而是**重塑损失函数本身**：给标准交叉熵乘一个随置信度衰减的调制因子，让易分样本的损失自动趋近于 0。

**关键公式**

定义样本对自身类别的预测概率：

$$p_t = \begin{cases} p & y = 1 \\ 1 - p & y = 0 \end{cases}$$

标准交叉熵：

$$\mathrm{CE}(p_t) = -\log(p_t)$$

α-balanced 交叉熵（按类别频率加权）：

$$\mathrm{CE}(p_t) = -\alpha_t \log(p_t), \quad \alpha_t = \begin{cases} \alpha & y = 1 \\ 1 - \alpha & y = 0 \end{cases}$$

Focal Loss：

$$\mathrm{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \log(p_t)$$

含义：当 $p_t \to 1$（样本已被分对），$(1-p_t)^\gamma \to 0$，该样本的损失被压到接近 0；当 $p_t$ 小（难例），因子接近 1，损失几乎不变。原文最佳设置为 $\gamma = 2, \alpha = 0.25$。

**原文的一个重要实现细节**
作者指出，用 Focal Loss 训练时，最后一层卷积的偏置应按先验初始化：

$$b_0 = -\log\frac{1-\pi}{\pi}$$

其中 $\pi$ 是正类先验，使训练初始时刻的 $p$ 就等于 $\pi$，避免初期大量易分样本产生的大梯度把训练带偏。

**伪代码（原文 Algorithm 的等价形式，本项目的实现见 `scripts/07_focal_loss.py`）**

```
for each mini-batch (x, y):
    z = w·x + b
    p = sigmoid(z)
    p_t = p if y == 1 else 1 - p
    a_t = alpha if y == 1 else 1 - alpha          # alpha=None 时恒为 1
    L   = -a_t * (1 - p_t)^gamma * log(p_t)       # 逐样本损失
    g_z = [a_t*gamma*(1-p_t)^(gamma-1)*log(p_t) - a_t*(1-p_t)^gamma/p_t] * (2y-1)*p*(1-p)
    g_w = g_z · x
    g_b = sum(g_z)
    (w, b) ← Adam((w, b), (g_w, g_b))             # 优化器同样自实现
```

**我在本项目中验证到什么**

| 原文结论 | 本数据集（正负 5.38:1）的实测 |
|---|---|
| γ=2、α=0.25 是最佳设置 | γ=2 有效，但加上 α 反而略差；只用调制项（无 α）的 macroF1 最高（0.8902 @ 阈值 0.5） |
| Focal Loss 显著优于 CE | 阈值固定 0.5 时略优（+0.0060）；各配置自选阈值后**反而不如普通 CE**（0.8959 vs 0.8962） |
| 先验初始化偏置很关键 | 实测仅 +0.0006，基本无影响 |

---

## 论文二（补充）：Class-Balanced Loss Based on Effective Number of Samples

**完整引文**
> Yin Cui, Menglin Jia, Tsung-Yi Lin, Yang Song, Serge Belongie. Class-Balanced Loss Based on Effective Number of Samples. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2019, pp. 9268–9277. DOI: 10.1109/CVPR.2019.00949. arXiv:1901.05555.

**研究问题**
长尾分布下，按样本数反比加权会过度补偿尾部类。作者认为：**随着样本数增加，新加入一个样本带来的边际信息是递减的**（数据之间存在重叠），因此应当用"有效样本数"而不是原始样本数来定权重。

**关键公式**

有效样本数（把每个样本看作一个小邻域而非孤立点）：

$$E_n = \frac{1 - \beta^n}{1 - \beta}, \quad \beta \in [0, 1)$$

当 $\beta \to 0$ 时 $E_n \to n$（无衰减）；$\beta \to 1$ 时 $E_n \to 1/(1-\beta)$（完全饱和）。

Class-Balanced 权重：$\alpha_t \propto 1/E_{n_y}$，与 Focal Loss 组合为

$$\mathrm{CB\text{-}focal}(\mathbf{z}, y) = -\frac{1-\beta}{1-\beta^{n_y}} (1-p_t)^{\gamma} \log(p_t)$$

原文指出：该式等价于把 Focal Loss 的 $\alpha_t$ 显式地取为 $(1-\beta)/(1-\beta^{n_y})$。

**我在本项目中验证到什么**

本训练集正类 214,927 条、负类 39,975 条，$E_n$ 在 $n$ 达到万级时对任意 $\beta < 1$ 都已饱和（$E_n \to 1/(1-\beta)$），于是两类的 $1/E_n$ 比值趋近 1，加权项退化。实测：

- β=0.9 ~ 0.999 时，α 权重比仅为 1.00~1.02，等同不加权；
- 只有 β 取到 0.99999 这种极端值，权重比才接近频率反比 5.38；
- 12 组 CB 实验中最好的 macroF1 = 0.8940，仍低于 CE 基线的 0.8962。

结论：Class-Balanced 是为**长尾、尾部类只有个位数样本**设计的，本任务两类都在万级、不平衡度仅 5.38:1，不在其适用区间。

---

## 论文三（补充）：The Foundations of Cost-Sensitive Learning

**完整引文**
> Charles Elkan. The Foundations of Cost-Sensitive Learning. *Proceedings of the 17th International Joint Conference on Artificial Intelligence (IJCAI'01)*, Volume 2, pp. 973–978, Seattle, Washington: Morgan Kaufmann, 2001.

**研究问题**
不同误分类代价不同时，如何做最优决策？作者给出两类路径：改变训练数据的分布（重采样/重加权），或在训练后**显式利用概率估计计算最优决策**（移阈值）。

**核心结论（与本实验直接呼应）**

原文摘要明确指出：

> ...changing the balance of negative and positive training examples has little effect on the classifiers produced by standard Bayesian and decision tree learning methods. Accordingly, the recommended way ... is to learn a classifier from the training set as given, and then to compute optimal decisions explicitly using the probability estimates given by the classifier.

即：**改变训练集中正负样本的平衡，对标准学习方法得到的分类器影响很小；推荐做法是照原样训练，然后用分类器给出的概率显式地算出最优决策。**

最优决策阈值可由代价矩阵直接导出。记把负类误判为正类的代价为 $C_{FP}$、把正类误判为负类的代价为 $C_{FN}$，则最优阈值为

$$t^* = \frac{C_{FP}}{C_{FP} + C_{FN}}$$

**我在本项目中验证到什么**

这条 2001 年的理论预言，在本数据集上被完整复现：

- `class_weight='balanced'` 让负类召回 +19.0 个百分点、精度 −18.0 个百分点，F1 几乎不动（0.7892 → 0.7933），macroF1 反而微降；
- 固定模型、只在**验证集**上挑一个阈值（0.65），测试集 F1负 达到 0.8172、macroF1 0.8917，**在 F1负、macroF1、准确率三项上全部超过 balanced 加权**。

也就是说，加权不但等价于移阈值，而且**移的位置不对**（过于激进）。这一点在实验之前就能从 Elkan 的分析中预期到，本次实验给出了在文本情感分类上的实证。

---

## 研究问题 — 论文方法 — 自实现模块 对应表

| 研究问题 | 对应论文方法 | 自实现模块（文件:函数/类） | 实验产出 |
|---|---|---|---|
| 不平衡下如何提升少数类召回 | α-balanced 交叉熵（Lin 2017 §3） | `07_focal_loss.py: loss_grad(alpha=..., gamma=0)` | WCE α=0.25，macroF1 0.8895 |
| 能否让模型关注难例而非类别频率 | Focal Loss 调制项 $(1-p_t)^\gamma$ | `07_focal_loss.py: loss_grad(gamma>0)` | FL γ=2 无α，macroF1 0.8959 |
| 损失梯度是否与标准 BCE 一致 | —（自证） | `07_focal_loss.py: loss_grad` 中 γ=0 退化校验 | 与 $p-y$ 解析解一致 |
| 优化过程是否可控可检查 | Adam（Kingma & Ba 2015） | `07_focal_loss.py: class Adam` | 全部配置训练耗时约 3s |
| 先验初始化是否关键 | Lin 2017 §4 初始化技巧 | `07_focal_loss.py: CostSensitiveLinear(init_prior=)` | +0.0006，无影响 |
| 有效样本数加权是否优于频率反比 | Cui 2019 式 (13) | `08_class_balanced.py: cb_alpha()` | 最优 0.8940，仍不及 CE |
| 加权是否优于移阈值 | Elkan 2001 定理 | `06_threshold_scan.py: pick_threshold` | 调阈值全面胜出 |
| 收益是否集中在特定长度 | （本项目拓展） | `07_focal_loss.py: length_bucket()` | 短/中评 +0.003~+0.009，长评 −0.031 |

---

## 与题目推荐文献的对照说明

题目建议「Lin et al. (2017), Focal Loss（迁移损失思想须说明）」。按此要求说明迁移的正当性与边界：

Focal Loss 的调制项作用于**每个样本自身预测正确的程度** $(1-p_t)$，这一量与任务无关，因此在文本分类上形式完全成立，不需要任何改写。
但它的设计动机是 1:1000 量级的前景/背景不平衡，而本任务是 5.38:1 的轻度不平衡，且难例主要来自星级标签噪声与反讽表达，而非"少数类样本太少"。
因此迁移后收益有限，这一点在实验中得到了量化的确认，也是本次工作最有价值的结论之一。
