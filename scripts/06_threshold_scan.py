"""阈值扫描:验证「类别加权约等于平移决策阈值」这一判断

思路:把一个不带权重的 LogReg 训练出来,然后手动扫描判决阈值。
如果扫描得到的最优 F1 与 class_weight='balanced' 的结果差不多,
就说明按类别频率加权并没有带来超出"挑个更好的阈值"的收益,
这正是引入 Focal Loss(对样本难易加权)的动机。

用法: python 06_threshold_scan.py
"""
import sys, time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_utils import (ROOT, eval_preds, eval_proba, format_report, load_dataset,
                        confusion_frame)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

VEC_KW = dict(max_features=100000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)


def main():
    log = open(ROOT / "reports" / "threshold_scan.md", "w", encoding="utf-8")
    def p(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        log.write(s + "\n")

    p("# 阈值扫描:类别加权是否只是「换了个阈值」")
    p()
    p("训练一个**不带权重**的 TF-IDF 逻辑回归,固定模型不动,只扫描判决阈值。")
    p("若扫描出的最优 F1 与 `class_weight='balanced'` 的结果基本持平,")
    p("则说明类别加权没有带来超出阈值平移的收益。")
    p()

    d = load_dataset()
    vec = TfidfVectorizer(**VEC_KW)
    t = time.perf_counter()
    Xtr = vec.fit_transform(d.tr[d.text_col].astype(str))
    Xva = vec.transform(d.va[d.text_col].astype(str))
    Xte = vec.transform(d.te[d.text_col].astype(str))
    p(f"TF-IDF {len(vec.vocabulary_)} 维,拟合 {time.perf_counter()-t:.1f}s")
    p()

    clf = LogisticRegression(max_iter=1000, C=1.0, solver="liblinear")
    clf.fit(Xtr, d.tr.label)
    prob = clf.predict_proba(Xte)[:, 1]
    prob_va = clf.predict_proba(Xva)[:, 1]

    p(f"模型在测试集上的概率分布: 均值 {prob.mean():.4f}, "
      f"分位 [p5={np.percentile(prob,5):.3f}, p25={np.percentile(prob,25):.3f}, "
      f"中位={np.median(prob):.3f}, p75={np.percentile(prob,75):.3f}]")
    p()

    rows = []
    for th in np.round(np.arange(0.05, 0.96, 0.05), 2):
        rows.append(eval_proba(d.te.label, prob, threshold=th, name=f"阈值 {th:.2f}"))
    rows.append(eval_preds(d.te.label, clf.predict(Xte), name="默认阈值 0.50"))

    p("## 阈值扫描结果(测试集,按 macroF1 排序)")
    p()
    p(format_report(rows, sort_by="macro_f1"))
    p()

    best_mac = max(rows, key=lambda r: r["macro_f1"])
    best_f1n = max(rows, key=lambda r: r["f1_neg"])
    best_th = float(best_mac["name"].split()[-1])
    p(f"- 最优 macroF1 出现在阈值 **{best_mac['name']}**:macroF1={best_mac['macro_f1']:.4f}, "
      f"F1负={best_mac['f1_neg']:.4f}, Accuracy={best_mac['accuracy']:.4f}")
    p(f"- 最优 F1负 出现在阈值 **{best_f1n['name']}**:F1负={best_f1n['f1_neg']:.4f}, "
      f"macroF1={best_f1n['macro_f1']:.4f}")
    p()
    p("> 上面是在**测试集**上扫描,只能当该模型的能力上界(oracle),不能当方法结论 ——")
    p("> 阈值是超参数,必须在验证集上选。下面的对照才是有效结果。")
    p()

    va_rows = [eval_proba(d.va.label, prob_va, threshold=th, name=f"{th:.2f}")
               for th in np.round(np.arange(0.05, 0.96, 0.05), 2)]
    best_va = max(va_rows, key=lambda r: r["macro_f1"])
    th_va = float(best_va["name"])
    p("## 合法做法:在验证集上选阈值")
    p()
    p(f"验证集最优阈值(macroF1 准则) = **{th_va:.2f}**,对应验证集 macroF1 = {best_va['macro_f1']:.4f}")
    p("把该阈值原样用到测试集:")
    p()

    # 与 balanced 版本对照
    clf_b = LogisticRegression(max_iter=1000, C=1.0, solver="liblinear", class_weight="balanced")
    clf_b.fit(Xtr, d.tr.label)
    row_b = eval_preds(d.te.label, clf_b.predict(Xte), name="balanced (阈值 0.5)")
    at_va = eval_proba(d.te.label, prob, threshold=th_va, name=f"验证集选阈值 {th_va:.2f}")
    oracle = eval_proba(d.te.label, prob, threshold=best_th, name=f"oracle 上界 {best_th:.2f}")

    p("## 关键对照(全部为测试集结果)")
    p()
    p(format_report([rows[-1], row_b, at_va, oracle]))
    p()
    p(f"- 默认阈值 0.5          F1负 {rows[-1]['f1_neg']:.4f}   macroF1 {rows[-1]['macro_f1']:.4f}   Acc {rows[-1]['accuracy']:.4f}")
    p(f"- balanced 加权        F1负 {row_b['f1_neg']:.4f}   macroF1 {row_b['macro_f1']:.4f}   Acc {row_b['accuracy']:.4f}")
    p(f"- 验证集选阈值 {th_va:.2f}    F1负 {at_va['f1_neg']:.4f}   macroF1 {at_va['macro_f1']:.4f}   Acc {at_va['accuracy']:.4f}")
    p(f"- oracle 上界 {best_th:.2f}      F1负 {oracle['f1_neg']:.4f}   macroF1 {oracle['macro_f1']:.4f}   Acc {oracle['accuracy']:.4f}  (仅供参照,非合法方法)")
    p()
    p(f"**在验证集上挑一个阈值,测试集 F1负 比 balanced 加权高 {at_va['f1_neg']-row_b['f1_neg']:+.4f},**")
    p(f"macroF1 高 {at_va['macro_f1']-row_b['macro_f1']:+.4f},准确率也高 {at_va['accuracy']-row_b['accuracy']:+.4f}。")
    p("这说明按类别频率加权不只等价于移阈值,而且**移的位置偏激进** —— 它把判决边界推过了头,")
    p("召回涨上去了,精度却跌得更多,连一个在验证集上随手挑的阈值都比不过。")
    p()

    p("## 验证集阈值在测试集上的混淆矩阵")
    p()
    p("```")
    p(confusion_frame(d.te.label, (prob >= th_va).astype(int)).to_string())
    p("```")

    # 图
    ths = [float(r["name"].split()[-1]) for r in rows[:-1]]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ths, [r["f1_pos"] for r in rows[:-1]], marker="o", ms=3, color="#55a868", label="F1 正")
    ax[0].plot(ths, [r["f1_neg"] for r in rows[:-1]], marker="s", ms=3, color="#c44e52", label="F1 负")
    ax[0].plot(ths, [r["macro_f1"] for r in rows[:-1]], marker="^", ms=3, color="#4c72b0", label="macro F1")
    ax[0].axvline(0.5, ls="--", color="#888780", lw=1, label="默认 0.5")
    ax[0].axvline(float(best_mac["name"].split()[-1]), ls=":", color="#dd8452", lw=1.5)
    ax[0].set_xlabel("判决阈值")
    ax[0].set_ylabel("F1")
    ax[0].set_title("阈值与 F1 的关系")
    ax[0].legend(fontsize=8)

    ax[1].plot([r["rec_neg"] for r in rows[:-1]], [r["prec_neg"] for r in rows[:-1]],
               marker="o", ms=3, color="#c44e52")
    for r in rows[::4][:-1]:
        ax[1].annotate(r["name"].split()[-1], (r["rec_neg"], r["prec_neg"]), fontsize=7)
    ax[1].scatter([row_b["rec_neg"]], [row_b["prec_neg"]], marker="*", s=140, color="#dd8452",
                  label="balanced 加权", zorder=5)
    ax[1].set_xlabel("负类 Recall")
    ax[1].set_ylabel("负类 Precision")
    ax[1].set_title("负类的 PR 权衡(沿曲线移动等价于调阈值)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "figures" / "fig6_threshold_curve.png", dpi=140, bbox_inches="tight")
    p()
    p("图: `reports/figures/fig6_threshold_curve.png`")

    log.close()
    print("\n完成 ->", ROOT / "reports" / "threshold_scan.md")


if __name__ == "__main__":
    main()
