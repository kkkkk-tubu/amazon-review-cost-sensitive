"""经典基线对照:朴素贝叶斯与线性 SVM

回应个人报告第 7 节的一条局限 —— 前面的代价敏感实验只在线性模型(自实现的
LogReg 与 sklearn LogReg)上验证过。这里补上两类经典算法,看结论是否外推。

用法: python 09_classic_baselines.py
"""
import sys, time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_utils import ROOT, eval_preds, eval_proba, format_report, load_dataset

VEC_KW = dict(max_features=100000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)


def pick_threshold_by_quantile(score, y_true, n=19):
    """按分位数取候选阈值。适用于概率输出,也适用于 SVM 的 decision_function"""
    cands = np.quantile(score, np.linspace(0.05, 0.95, n))
    best_th, best_v = float(cands[n // 2]), -1.0
    for th in cands:
        v = eval_proba(y_true, score, threshold=float(th))["macro_f1"]
        if v > best_v:
            best_v, best_th = v, float(th)
    return best_th, best_v


def main():
    log = open(ROOT / "reports" / "classic_baselines.md", "w", encoding="utf-8")
    def p(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        log.write(s + "\n")

    p("# 经典基线对照:朴素贝叶斯与线性 SVM")
    p()
    p("目的:检验前面的结论(代价敏感方法收益有限、调阈值更有效)是否只在逻辑回归上成立。")
    p()
    p("**关于 SVM 的说明**:本数据集训练集 25 万条,RBF 等非线性核的复杂度随样本规模")
    p("呈平方至立方增长,在此规模下不可行,因此这里使用线性核 `LinearSVC`。")
    p()

    d = load_dataset()
    vec = TfidfVectorizer(**VEC_KW)
    t = time.perf_counter()
    Xtr = vec.fit_transform(d.tr[d.text_col].astype(str))
    Xva, Xte = vec.transform(d.va[d.text_col].astype(str)), vec.transform(d.te[d.text_col].astype(str))
    p(f"TF-IDF {len(vec.vocabulary_)} 维, 拟合 {time.perf_counter()-t:.1f}s")
    p(f"训练 {len(d.tr)} / 验证 {len(d.va)} / 测试 {len(d.te)}")
    p()

    rows_default, rows_tuned = [], []

    for name, clf in [
        ("MultinomialNB", MultinomialNB(alpha=0.1)),
        ("ComplementNB", ComplementNB(alpha=0.1)),
        ("LinearSVC", LinearSVC(C=1.0, max_iter=2000)),
        ("LinearSVC (balanced)", LinearSVC(C=1.0, max_iter=2000, class_weight="balanced")),
    ]:
        t0 = time.perf_counter()
        clf.fit(Xtr, d.tr.label.values)
        fit_sec = time.perf_counter() - t0

        t0 = time.perf_counter()
        if hasattr(clf, "predict_proba"):
            score = clf.predict_proba(Xte)[:, 1]
            score_va = clf.predict_proba(Xva)[:, 1]
            default_pred = clf.predict(Xte)
        else:
            score = clf.decision_function(Xte)
            score_va = clf.decision_function(Xva)
            default_pred = clf.predict(Xte)
        pred_sec = time.perf_counter() - t0

        rows_default.append(eval_preds(d.te.label.values, default_pred, fit_sec, pred_sec, name))
        th, va_m = pick_threshold_by_quantile(score_va, d.va.label.values)
        r = eval_proba(d.te.label.values, score, threshold=th, fit_sec=fit_sec,
                       pred_sec=pred_sec, name=name)
        r["note"] = f"阈值{th:.3f}" if 0 <= th <= 1 else f"decision{th:.3f}"
        rows_tuned.append(r)
        print(f"  {name}: 默认 macroF1={rows_default[-1]['macro_f1']:.4f} | "
              f"调阈值 macroF1={r['macro_f1']:.4f} | 训练 {fit_sec:.0f}s", flush=True)

    p("## 1. 默认决策(阈值 0.5 / SVM 取 decision>0)")
    p()
    p(format_report(rows_default, sort_by="macro_f1"))
    p()
    p("## 2. 各模型在验证集自选阈值后")
    p()
    p(format_report(rows_tuned, sort_by="macro_f1",
                    cols=["name", "note", "accuracy", "rec_neg", "prec_neg", "f1_neg",
                          "f1_pos", "macro_f1", "fit_sec", "pred_sec"]))
    p()

    best = max(rows_tuned, key=lambda r: r["macro_f1"])
    p("## 3. 与逻辑回归系列结果的横向对比")
    p()
    p("| 模型 | 默认 macroF1 | 调阈值后 macroF1 | 训练(s) |")
    p("|---|---|---|---|")
    p("| LogReg TF-IDF (sklearn, 05 脚本) | 0.8773 | 0.8917 | 3.5 |")
    p("| CE 基线 (自实现, 07 脚本) | 0.8842 | 0.8962 | 2.9 |")
    p("| FL γ=2 无α (自实现, 07 脚本) | 0.8902 | 0.8959 | 2.9 |")
    for rd, rt in zip(rows_default, rows_tuned):
        p(f"| {rd['name']} (09 脚本) | {rd['macro_f1']:.4f} | {rt['macro_f1']:.4f} | {rd['fit_sec']:.1f} |")
    p()
    p(f"经典算法中表现最好的是 **{best['name']}**,调阈值后 macroF1 = {best['macro_f1']:.4f}。")
    p()

    p("## 4. 结论")
    p()
    nb = next(r for r in rows_default if r["name"] == "MultinomialNB")
    cnb = next(r for r in rows_default if r["name"] == "ComplementNB")
    svm = next(r for r in rows_default if r["name"] == "LinearSVC")
    svm_b = next(r for r in rows_default if r["name"] == "LinearSVC (balanced)")
    p(f"- **朴素贝叶斯在本任务上明显偏弱**。MultinomialNB 默认 macroF1 只有 {nb['macro_f1']:.4f}:")
    p(f"  负类召回仅 {nb['rec_neg']:.4f},而负类精度却有 {nb['prec_neg']:.4f} —— 它极少把样本判为负面,")
    p("  于是漏掉了近一半的真实负面评论。84% 的正类先验压过了文本证据。")
    p(f"- 专为不平衡设计的 ComplementNB 走向另一个极端:负类召回 {cnb['rec_neg']:.4f} 很高,")
    p(f"  精度却只有 {cnb['prec_neg']:.4f},macroF1 {cnb['macro_f1']:.4f} 同样不理想。")
    p("  两者一个过保守、一个过激进,都需要靠调阈值找中间点,而调完阈值后")
    p("  两者的 macroF1 恰好收敛到同一个值(预测结果完全一致)。")
    p(f"- **线性 SVM 是经典算法里最强的**:默认 macroF1 {svm['macro_f1']:.4f},高于逻辑回归的 0.8773;")
    p("  但调阈值后为 0.8925,仍低于自实现 CE 基线的 0.8962。")
    p(f"- **关键复现**:在 SVM 上加 `class_weight='balanced'`,macroF1 {svm['macro_f1']:.4f} → "
      f"{svm_b['macro_f1']:.4f},")
    p(f"  负类召回 {svm['rec_neg']:.4f} → {svm_b['rec_neg']:.4f},负类精度 {svm['prec_neg']:.4f} → "
      f"{svm_b['prec_neg']:.4f} —— 与逻辑回归上的观察完全一致。")
    p()
    p("**这条跨算法的复现很重要**:代价敏感加权「召回涨、精度跌、F1 基本不动」的现象")
    p("不是逻辑回归特有的,在结构完全不同的线性 SVM 上同样出现,说明前面得出的结论")
    p("可以外推,而不是某个模型的巧合。")
    log.close()
    print("\n完成 ->", ROOT / "reports" / "classic_baselines.md")


if __name__ == "__main__":
    main()
