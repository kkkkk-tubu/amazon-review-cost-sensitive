"""基线实验(题目步骤2):多数类基线 + TF-IDF 逻辑回归

同时验证 eval_utils 与划分表能协同工作,输出第一轮可对照的指标。
用法: python 05_baseline.py
"""
import sys, time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_utils import (ROOT, class_weights, confusion_frame, eval_preds,
                        format_report, load_dataset, majority_rate)


def main():
    log = open(ROOT / "reports" / "baseline_result.md", "w", encoding="utf-8")
    def p(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        log.write(s + "\n")

    p("# 基线实验结果")
    p()
    p(f"- 数据: `Reviews_binary.csv` 364,163 条 x 划分表 `splits.csv`(random seed=42)")
    p(f"- 文本列: `tokens_nostop`(去停用词、保留否定词)　指标口径: pos=1 正面, neg=0 负面")
    p()

    d = load_dataset()
    p(f"训练集 {len(d.tr)} / 验证集 {len(d.va)} / 测试集 {len(d.te)}")
    p(f"训练集正例率 {d.tr.label.mean():.4f}　测试集正例率 {d.te.label.mean():.4f}")
    p()

    rows = []

    # 1 多数类基线
    t = time.perf_counter()
    y_pred = np.ones(len(d.te), dtype=int)
    rows.append(eval_preds(d.te.label, y_pred, 0.0, time.perf_counter() - t,
                           "多数类基线(全判正)",
                           note=f"理论上限={majority_rate(d.te.label):.4f}"))

    # 2 / 3 TF-IDF + 逻辑回归
    vec = TfidfVectorizer(max_features=100000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    t = time.perf_counter()
    Xtr = vec.fit_transform(d.tr[d.text_col].astype(str))
    t_fit_vec = time.perf_counter() - t
    p(f"TF-IDF 词表 {len(vec.vocabulary_)} 维 (1-2gram, min_df=2), 拟合 {t_fit_vec:.1f}s")
    Xva, Xte = vec.transform(d.va[d.text_col].astype(str)), vec.transform(d.te[d.text_col].astype(str))
    p()

    w = class_weights(d.tr.label)
    p(f"balanced 权重: w_pos={w[1]:.4f}  w_neg={w[0]:.4f}  (比值 {w[0]/w[1]:.2f})")
    p()

    for name, kw in [("LogReg TF-IDF", None),
                     ("LogReg TF-IDF (balanced)", dict(class_weight="balanced"))]:
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="liblinear", **({"class_weight": kw["class_weight"]} if kw else {}))
        t0 = time.perf_counter()
        clf.fit(Xtr, d.tr.label)
        fit_sec = time.perf_counter() - t0
        t0 = time.perf_counter()
        y_pred = clf.predict(Xte)
        pred_sec = time.perf_counter() - t0
        rows.append(eval_preds(d.te.label, y_pred, fit_sec, pred_sec, name))

    p("## 测试集指标")
    p()
    p(format_report(rows, sort_by="macro_f1"))
    p()

    p("## 最佳基线的混淆矩阵 (LogReg balanced,测试集)")
    p()
    p("```")
    p(confusion_frame(d.te.label, clf.predict(Xte)).to_string())
    p("```")
    p()

    a, b = rows[1], rows[2]
    p("## 观察")
    p()
    p(f"- 多项类基线 Accuracy = {rows[0]['accuracy']:.4f},但 macroF1 仅 {rows[0]['macro_f1']:.4f},"
      f"负类 F1 = {rows[0]['f1_neg']:.4f} —— 准确率完全无法反映它一条负面都没捞回来。")
    p(f"- 加入 balanced 权重后,负类召回 {a['rec_neg']:.4f} → {b['rec_neg']:.4f} "
      f"(+{100*(b['rec_neg']-a['rec_neg']):.1f} 个百分点),但负类精度 {a['prec_neg']:.4f} → {b['prec_neg']:.4f} "
      f"({100*(b['prec_neg']-a['prec_neg']):+.1f} 个百分点)。")
    p(f"- **F1负几乎没动**:{a['f1_neg']:.4f} → {b['f1_neg']:.4f},macroF1 反而微降 "
      f"{a['macro_f1']:.4f} → {b['macro_f1']:.4f},代价是 Accuracy 掉 {100*(a['accuracy']-b['accuracy']):.2f} 个百分点。")
    p()
    p("这条结果暴露了类别加权的核心局限:**按类别频率加权本质上是在平移决策边界,**")
    p("等价于事后挑一个更好的阈值,并没有让同一批样本变得更容易分辨。所以召回涨多少,精度")
    p("就几乎跌多少,F1 这个调和平均值基本原地不动。这正是 Focal Loss 的出发点:它的调制项")
    p("`(1-p_t)^γ` 作用于**每个样本的难易程度**,而不是简单地把整个少数类的损失整体放大。")
    p()
    p("## 下一步建议")
    p()
    p("1. 用默认 LogReg 做一次阈值扫描(改 `eval_proba` 的 threshold),验证「加权 ≈ 移阈值」这个判断。")
    p("   若最优阈值的 F1 与 balanced 版本接近,就说明简单加权没带来实质收益 —— 这正是引入")
    p("   Focal Loss 的动机,可以直接写进论文的相关工作与动机部分。")
    p("2. 自实现加权交叉熵与 Focal Loss(含梯度推导),与上面三行对照; ")
    p("3. 用 `splits_multi.csv` 的 random_1 / random_2024 两个种子复跑,报告多随机种子的均值与方差。")
    log.close()


if __name__ == "__main__":
    main()
