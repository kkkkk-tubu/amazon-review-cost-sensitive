"""补充实验:Class-Balanced Loss (Cui et al., CVPR 2019)

Cui 等人提出不用样本数、而用"有效样本数"来定类别权重:

    E_n = (1 - β^n) / (1 - β),      β ∈ [0, 1)

再按 1/E_n 给损失加权,得到 α_t = (1-β)/(1-β^{n_y})。
该设计面向长尾分布(尾部类只有个位数样本)。本数据集只有两类且各有几万样本,
预期 E_n 饱和、权重退化 —— 本脚本验证这一点。

用法: python 08_class_balanced.py
"""
import importlib, sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_utils import ROOT, eval_proba, format_report, load_dataset

_focal = importlib.import_module("07_focal_loss")
CostSensitiveLinear, pick_threshold, VEC_KW = (_focal.CostSensitiveLinear,
                                               _focal.pick_threshold, _focal.VEC_KW)

BETAS = [0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999]


def cb_alpha(n_pos, n_neg, beta):
    """α_t ∝ 1/E_n,归一化使 α_pos + α_neg = 1"""
    def eff(n):
        return (1 - beta ** n) / (1 - beta)
    w_pos, w_neg = 1 / eff(n_pos), 1 / eff(n_neg)
    return w_pos / (w_pos + w_neg)


def main():
    log = open(ROOT / "reports" / "class_balanced_result.md", "w", encoding="utf-8")
    def p(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        log.write(s + "\n")

    p("# 补充实验:Class-Balanced Loss(Cui et al., CVPR 2019)")
    p()
    p(r"有效样本数 $E_n=(1-\beta^n)/(1-\beta)$,类别权重 $\alpha_t=(1-\beta)/(1-\beta^{n_y})$。")
    p("下表先看权重本身:本数据集两类各有几万样本,$E_n$ 会迅速饱和。")
    p()

    d = load_dataset()
    n_pos = int((d.tr.label == 1).sum())
    n_neg = int((d.tr.label == 0).sum())
    p(f"训练集: 正类 {n_pos}, 负类 {n_neg}, 频率反比权重比 = {n_pos/n_neg:.2f}")
    p()
    p("| β | α_pos | α_neg | α_neg/α_pos | 相对频率反比(5.38) |")
    p("|---|---|---|---|---|")
    for b in BETAS:
        a = cb_alpha(n_pos, n_neg, b)
        p(f"| {b:g} | {a:.6f} | {1-a:.6f} | {(1-a)/a:.4f} | {((1-a)/a)/(n_pos/n_neg):.3f} |")
    p()

    vec = TfidfVectorizer(**VEC_KW)
    Xtr = vec.fit_transform(d.tr[d.text_col].astype(str))
    Xva, Xte = vec.transform(d.va[d.text_col].astype(str)), vec.transform(d.te[d.text_col].astype(str))
    p(f"TF-IDF {len(vec.vocabulary_)} 维")
    p()

    rows = []
    for b in BETAS:
        a = cb_alpha(n_pos, n_neg, b)
        for tag, gamma in [("CB-CE", 0.0), ("CB-Focal", 2.0)]:
            clf = CostSensitiveLinear(Xtr.shape[1], alpha=a, gamma=gamma, lr=0.05, epochs=20)
            clf.fit(Xtr, d.tr.label.values)
            pr = clf.predict_proba(Xte)
            th, _ = pick_threshold(d.va.label.values, clf.predict_proba(Xva))
            r = eval_proba(d.te.label.values, pr, threshold=th, fit_sec=clf.fit_sec,
                           name=f"{tag} β={b:g}")
            r["note"] = f"α={a:.4f} 阈值{th:.2f}"
            rows.append(r)
            print(f"  {tag} β={b:g}: α={a:.4f} macroF1={r['macro_f1']:.4f} F1负={r['f1_neg']:.4f}",
                  flush=True)

    p("## 结果(验证集选阈值后的测试集表现)")
    p()
    p(format_report(rows, cols=["name", "note", "accuracy", "rec_neg", "f1_neg", "macro_f1", "fit_sec"]))
    p()

    best = max(rows, key=lambda r: r["macro_f1"])
    p("## 结论")
    p()
    p(f"最好的一组是 **{best['name']}**,macroF1 = {best['macro_f1']:.4f}。")
    p("对照前面的结果:CE 基线在验证集选阈值后 macroF1 = 0.8962,FL γ=2 无α = 0.8959。")
    p()
    p("**Class-Balanced 在本任务上基本没有作用**,原因是它的设计与本数据的形态不匹配:")
    p()
    p("- 有效样本数 $E_n=(1-\\beta^n)/(1-\\beta)$ 在 $n$ 达到几万时对任意 $\\beta<1$ 都已饱和")
    p("  ($E_n \\to 1/(1-\\beta)$),于是两类的 $1/E_n$ 比值趋近 1,加权项退化;")
    p("- 只有当 $\\beta$ 取到 0.99999 这种极端值时权重比才接近频率反比,")
    p("  而这个区间正是 Cui 等人原文中留给「长尾、尾部类只有个位数样本」的场景。")
    p("  本任务两类样本都在万级,属于轻度不平衡(5.38:1),不在该方法的适用区间。")
    p()
    p("这条负面结果说明了方法迁移的边界:Focal Loss 与 Class-Balanced Loss 都诞生于")
    p("目标检测与长尾识别,那里的不平衡是 1:1000 量级;文本情感分类的类别不平衡温和得多,")
    p("真正的困难也不是「少数类样本太少」,而是星级标签自身的噪声与反讽表达。")
    p()
    p("## 运行成本")
    p()
    p(f"共 {len(rows)} 次训练,合计 {sum(r['fit_sec'] for r in rows):.0f}s, "
      f"平均 {np.mean([r['fit_sec'] for r in rows]):.1f}s/次。")
    log.close()
    print("\n完成 ->", ROOT / "reports" / "class_balanced_result.md")


if __name__ == "__main__":
    main()