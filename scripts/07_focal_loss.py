"""成员 B 核心实现:加权交叉熵与 Focal Loss(自写损失、梯度与优化器)

================================================================================
一、符号与损失定义(对齐 Lin et al. 2017, Focal Loss for Dense Object Detection)
================================================================================

线性打分        z = w·x + b
正类概率        p = σ(z) = 1 / (1 + exp(-z))
样本"自身类别"的概率
                p_t =  p        if y = 1
                       1 - p    if y = 0
类别平衡因子
                α_t =  α        if y = 1
                       1 - α    if y = 0        (α = None 时取 α_t = 1)

统一形式(γ=0 退化为加权交叉熵,α_t=1 且 γ=0 退化为标准交叉熵)

                L = - α_t · (1 - p_t)^γ · log(p_t)

================================================================================
二、梯度推导(本文件 loss_grad() 的实现依据)
================================================================================

对 p_t 求导:
    ∂L/∂p_t = -α_t · [ ∂(1-p_t)^γ/∂p_t · log p_t + (1-p_t)^γ · (1/p_t) ]
            = -α_t · [ -γ(1-p_t)^(γ-1) log p_t + (1-p_t)^γ / p_t ]
            = α_t·γ·(1-p_t)^(γ-1)·log(p_t)  -  α_t·(1-p_t)^γ / p_t

p_t 对 z 求导:
    ∂p_t/∂z = (2y - 1) · p · (1 - p)          (y=1 时为 p(1-p),y=0 时为其相反数)

链式法则:
    ∂L/∂z = [ α_t·γ·(1-p_t)^(γ-1)·log(p_t) - α_t·(1-p_t)^γ/p_t ] · (2y-1)·p·(1-p)

退化校验:
    取 γ=0、α_t=1,则 ∂L/∂z = -(1/p_t)·(2y-1)·p(1-p)
      y=1: -(1/p)·p(1-p) = -(1-p) = p - y      ✓ 与标准 BCE 一致
      y=0: -(1/(1-p))·(-1)·p(1-p) = p = p - y  ✓

参数梯度:
    ∂L/∂w = (∂L/∂z) · x        ∂L/∂b = ∂L/∂z

================================================================================
三、优化器:Adam(自实现,见 class Adam)
================================================================================

    m_t = β1·m_{t-1} + (1-β1)·g_t
    v_t = β2·v_{t-1} + (1-β2)·g_t²
    m̂ = m_t/(1-β1^t)      v̂ = v_t/(1-β2^t)
    w ← w - η · m̂ / (√v̂ + ε)

================================================================================
四、Focal Loss 论文提到的初始化技巧(本文件作为消融项)
================================================================================

用先验初始化偏置 b0 = log(π/(1-π)),π 为正类先验,使初始 p 就等于先验概率,
避免训练初期大量易分样本产生的大梯度把模型带偏。对应 init_prior=True。

用法:
    python 07_focal_loss.py                 # 全量实验
    python 07_focal_loss.py --quick         # 冒烟:只跑 2 个配置、3 个 epoch
"""
import argparse, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_utils import (ROOT, eval_preds, eval_proba, format_report, load_dataset,
                        confusion_frame)

EPS = 1e-7
VEC_KW = dict(max_features=100000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)


def sigmoid(z):
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def loss_grad(z, y, alpha=None, gamma=0.0):
    """返回 (逐样本损失, dL/dz)。alpha=None 表示不做类别加权(α_t = 1)"""
    p = np.clip(sigmoid(z), EPS, 1 - EPS)
    p_t = np.where(y == 1, p, 1 - p)
    a_t = np.ones_like(p) if alpha is None else np.where(y == 1, alpha, 1 - alpha)
    log_pt = np.log(p_t)

    if gamma == 0.0:
        loss = -a_t * log_pt
        dlz = -a_t / p_t
    else:
        one_minus = 1 - p_t
        loss = -a_t * (one_minus ** gamma) * log_pt
        dlz = a_t * gamma * (one_minus ** (gamma - 1)) * log_pt - a_t * (one_minus ** gamma) / p_t

    dlz = dlz * (2 * y - 1) * p * (1 - p)
    return loss, dlz


class Adam:
    def __init__(self, n, lr=0.05, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m, self.v, self.t = np.zeros(n), np.zeros(n), 0

    def step(self, w, g):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mh = self.m / (1 - self.b1 ** self.t)
        vh = self.v / (1 - self.b2 ** self.t)
        return w - self.lr * mh / (np.sqrt(vh) + self.eps)


class CostSensitiveLinear:
    """线性模型 + 自实现的损失/梯度/Adam,支持 CE / 加权 CE / Focal Loss"""

    def __init__(self, n_features, alpha=None, gamma=0.0, lr=0.05, epochs=20,
                 batch_size=8192, l2=1e-6, init_prior=False, seed=42):
        self.n_features = n_features
        self.alpha, self.gamma = alpha, gamma
        self.lr, self.epochs, self.bs, self.l2 = lr, epochs, batch_size, l2
        self.init_prior, self.seed = init_prior, seed
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.history = []

    def fit(self, X, y, verbose=False):
        rng = np.random.default_rng(self.seed)
        n = X.shape[0]
        y = y.astype(np.float64)
        if self.init_prior:
            pi = float(np.clip(y.mean(), EPS, 1 - EPS))
            self.b = float(np.log(pi / (1 - pi)))
        opt_w = Adam(self.n_features, self.lr)
        opt_b = Adam(1, self.lr)
        t0 = time.perf_counter()

        for ep in range(self.epochs):
            perm = rng.permutation(n)
            tot = 0.0
            for s in range(0, n, self.bs):
                idx = perm[s:s + self.bs]
                Xb = X[idx]
                yb = y[idx]
                z = Xb @ self.w + self.b
                loss, dlz = loss_grad(z, yb, self.alpha, self.gamma)
                tot += float(loss.sum())

                gw = Xb.T @ dlz
                gb = float(dlz.sum())
                m = len(yb)
                gw = gw / m + 2 * self.l2 * self.w
                gb = gb / m

                self.w = opt_w.step(self.w, gw)
                self.b = float(opt_b.step(np.array([self.b]), np.array([gb]))[0])

            self.history.append(tot / n)
            if verbose:
                print(f"      epoch {ep+1}/{self.epochs}  loss={tot/n:.5f}  "
                      f"{time.perf_counter()-t0:.0f}s", flush=True)
        self.fit_sec = time.perf_counter() - t0
        return self

    def decision(self, X):
        return X @ self.w + self.b

    def predict_proba(self, X):
        return sigmoid(self.decision(X))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


def pick_threshold(y_true, prob, metric="macro_f1"):
    """在验证集上扫描阈值,按指定指标选最优"""
    best_th, best_v = 0.5, -1
    for th in np.round(np.arange(0.05, 0.96, 0.05), 2):
        r = eval_proba(y_true, prob, threshold=th)
        if r[metric] > best_v:
            best_v, best_th = r[metric], th
    return float(best_th), best_v


CONFIGS = [
    ("CE 基线 (自实现)",            dict(alpha=None, gamma=0.0)),
    ("WCE α=0.25 (仅类别加权)",      dict(alpha=0.25, gamma=0.0)),
    ("FL α=0.25 γ=1",              dict(alpha=0.25, gamma=1.0)),
    ("FL α=0.25 γ=2",              dict(alpha=0.25, gamma=2.0)),
    ("FL α=0.25 γ=5",              dict(alpha=0.25, gamma=5.0)),
    ("FL γ=2 无α (仅调制项)",        dict(alpha=None, gamma=2.0)),
    ("FL γ=5 无α (仅调制项)",        dict(alpha=None, gamma=5.0)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.05)
    args = ap.parse_args()

    if args.quick:
        args.epochs = 3

    log = open(ROOT / "reports" / "focal_loss_result.md", "w", encoding="utf-8")
    def p(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        log.write(s + "\n")

    p("# 成员 B:加权交叉熵与 Focal Loss 的自实现与对照")
    p()
    p("模型是**自己写**的线性分类器:损失函数、梯度、Adam 更新全部逐行实现")
    p("(`scripts/07_focal_loss.py` 的 `loss_grad` 与 `Adam`),没有调用任何现成优化器。")
    p("特征统一为 TF-IDF(1-2gram, 10 万维),只在训练集拟合。")
    p()
    p(f"训练设置: Adam lr={args.lr}, epochs={args.epochs}, batch=8192, l2=1e-6, 种子 42")
    p()

    d = load_dataset()
    vec = TfidfVectorizer(**VEC_KW)
    t = time.perf_counter()
    Xtr = vec.fit_transform(d.tr[d.text_col].astype(str))
    Xva, Xte = vec.transform(d.va[d.text_col].astype(str)), vec.transform(d.te[d.text_col].astype(str))
    p(f"TF-IDF {len(vec.vocabulary_)} 维, 拟合 {time.perf_counter()-t:.1f}s")
    p(f"训练 {len(d.tr)} / 验证 {len(d.va)} / 测试 {len(d.te)}")
    p()

    configs = CONFIGS[:2] if args.quick else CONFIGS
    rows05, rows_tuned, models = [], [], {}

    p("## 1. 各配置的训练与测试集结果")
    p()
    for name, kw in configs:
        print(f"  >> {name}", flush=True)
        clf = CostSensitiveLinear(Xtr.shape[1], **kw, lr=args.lr, epochs=args.epochs)
        clf.fit(Xtr, d.tr.label.values, verbose=not args.quick)
        models[name] = clf

        t1 = time.perf_counter()
        pte = clf.predict_proba(Xte)
        pred_sec = time.perf_counter() - t1

        th, va_m = pick_threshold(d.va.label.values, clf.predict_proba(Xva))
        rows05.append(eval_proba(d.te.label.values, pte, threshold=0.5,
                                 fit_sec=clf.fit_sec, pred_sec=pred_sec, name=name))
        r_tuned = eval_proba(d.te.label.values, pte, threshold=th,
                             fit_sec=clf.fit_sec, pred_sec=pred_sec, name=name)
        r_tuned["note"] = f"阈值{th:.2f}"
        rows_tuned.append(r_tuned)
        p_row = rows05[-1]
        print(f"     阈值0.5: macroF1={p_row['macro_f1']:.4f} F1负={p_row['f1_neg']:.4f} | "
              f"验证集阈值{th:.2f}: macroF1={r_tuned['macro_f1']:.4f} F1负={r_tuned['f1_neg']:.4f} "
              f"| 训练 {clf.fit_sec:.0f}s", flush=True)

    p("### 1.1 固定阈值 0.5")
    p()
    p(format_report(rows05, sort_by="macro_f1",
                    cols=["name", "accuracy", "rec_neg", "prec_neg", "f1_neg",
                          "f1_pos", "macro_f1", "fit_sec", "pred_sec"]))
    p()
    p("### 1.2 各配置在验证集上自选阈值后再评估(公平比较)")
    p()
    p(format_report(rows_tuned, sort_by="macro_f1",
                    cols=["name", "note", "accuracy", "rec_neg", "prec_neg", "f1_neg",
                          "f1_pos", "macro_f1", "fit_sec", "pred_sec"]))
    p()

    base05 = next(r for r in rows05 if r["name"].startswith("CE"))
    best05 = max(rows05, key=lambda r: r["macro_f1"])
    best_t = max(rows_tuned, key=lambda r: r["macro_f1"])
    base_t = next(r for r in rows_tuned if r["name"].startswith("CE"))

    p("## 2. 消融:α 与 γ 各自贡献多少")
    p()
    p("| 配置 | 类别因子 α | 调制因子 γ | 阈值0.5 macroF1 | 阈值0.5 F1负 | 自选阈值 macroF1 |")
    p("|---|---|---|---|---|---|")
    for r05, rt in zip(rows05, rows_tuned):
        a = next(k.get("alpha") for n, k in configs if n == r05["name"])
        g = next(k["gamma"] for n, k in configs if n == r05["name"])
        p(f"| {r05['name']} | {'无' if a is None else a} | {g:g} | "
          f"{r05['macro_f1']:.4f} | {r05['f1_neg']:.4f} | {rt['macro_f1']:.4f} |")
    p()

    p("## 3. 结论(先说结论:提升不显著)")
    p()
    p(f"- 阈值固定 0.5 时,最好的配置是 **{best05['name']}**,macroF1={best05['macro_f1']:.4f},"
      f"相比 CE 基线 {base05['macro_f1']:.4f} 的变化为 {best05['macro_f1']-base05['macro_f1']:+.4f}。")
    p(f"- 各配置各自选最优阈值后,最好的是 **{best_t['name']}**,macroF1={best_t['macro_f1']:.4f},"
      f"相比 CE 基线 {base_t['macro_f1']:.4f} 的变化为 {best_t['macro_f1']-base_t['macro_f1']:+.4f}。")
    p()

    if not args.quick:
        ablation_prior(len(d.tr), Xtr, d, vec, p, args)
        multi_seed(d, vec, p, args)
        length_bucket(models, d, Xte, p)

    p()
    p("## 附:运行成本")
    p()
    p("| 配置 | 训练耗时(s) | 预测耗时(s) |")
    p("|---|---|---|")
    for r in rows05:
        p(f"| {r['name']} | {r['fit_sec']:.1f} | {r['pred_sec']:.3f} |")

    log.close()
    print("\n完成 ->", ROOT / "reports" / "focal_loss_result.md")


def ablation_prior(n_tr, Xtr, d, vec, p, args):
    """Focal Loss 论文的 bias 初始化技巧,单独做一次对照"""
    p("## 4. 消融:用先验初始化偏置 (init_prior)")
    p()
    p("Focal Loss 论文提到用先验概率初始化最后一层的偏置,使训练初期 p 就等于正类先验。")
    p("这里把它单独开关,观察是否有用。")
    p()
    rows = []
    for name, kw in [("FL α=0.25 γ=2", dict(alpha=0.25, gamma=2.0))]:
        for init in [False, True]:
            clf = CostSensitiveLinear(Xtr.shape[1], **kw, lr=args.lr, epochs=args.epochs,
                                      init_prior=init)
            clf.fit(Xtr, d.tr.label.values)
            pr = clf.predict_proba(vec.transform(d.te[d.text_col].astype(str)))
            th, _ = pick_threshold(d.va.label.values,
                                   clf.predict_proba(vec.transform(d.va[d.text_col].astype(str))))
            r = eval_proba(d.te.label.values, pr, threshold=th, fit_sec=clf.fit_sec,
                           name=f"{name} | init_prior={init}")
            r["note"] = f"阈值{th:.2f}"
            rows.append(r)
    p(format_report(rows, cols=["name", "note", "accuracy", "rec_neg", "f1_neg", "macro_f1", "fit_sec"]))
    p()
    delta = rows[1]["macro_f1"] - rows[0]["macro_f1"]
    p(f"开启先验初始化后 macroF1 变化 {delta:+.4f} —— "
      f"{'有一定作用' if abs(delta) > 0.002 else '基本无影响'}。")
    p()


def multi_seed(d_def, vec, p, args):
    """用随机种子 1 与 2024 复跑 CE 基线与 Focal,报告波动"""
    p("## 5. 多随机种子")
    p()
    p("切换 `splits_multi.csv` 里的 random_1 / random_2024 划分,CE 基线与 Focal 各复跑一次。")
    p()
    out = []
    for seed_mode in ["random_1", "random_2024"]:
        d = load_dataset(mode=seed_mode)
        Xtr = vec.fit_transform(d.tr[d.text_col].astype(str))
        Xva = vec.transform(d.va[d.text_col].astype(str))
        Xte = vec.transform(d.te[d.text_col].astype(str))
        for name, kw in [("CE 基线 (自实现)", dict(alpha=None, gamma=0.0)),
                         ("FL α=0.25 γ=2", dict(alpha=0.25, gamma=2.0))]:
            clf = CostSensitiveLinear(Xtr.shape[1], **kw, lr=args.lr, epochs=args.epochs)
            clf.fit(Xtr, d.tr.label.values)
            pr = clf.predict_proba(Xte)
            th, _ = pick_threshold(d.va.label.values, clf.predict_proba(Xva))
            r = eval_proba(d.te.label.values, pr, threshold=th, fit_sec=clf.fit_sec,
                           name=f"{name} [{seed_mode}]")
            r["note"] = f"阈值{th:.2f}"
            out.append(r)
            print(f"     {seed_mode} {name}: macroF1={r['macro_f1']:.4f} F1负={r['f1_neg']:.4f}", flush=True)
    p(format_report(out, cols=["name", "note", "accuracy", "rec_neg", "f1_neg", "macro_f1"]))
    p()
    for tag in ["CE 基线 (自实现)", "FL α=0.25 γ=2"]:
        vs = [r["macro_f1"] for r in out if r["name"].startswith(tag)]
        if len(vs) > 1:
            p(f"- {tag} 在不同划分上的 macroF1: {[f'{v:.4f}' for v in vs]}, "
              f"极差 {max(vs)-min(vs):.4f}")
    p()


def length_bucket(models, d, Xte, p):
    """按评论长度分桶,看收益集中在哪"""
    p("## 6. 拓展分析:收益随评论长度的分布")
    p()
    p("清洗阶段发现长度与情感极性不是单调关系(短评正面率 0.905,101-200 词最低 0.817)。")
    p("这里看 Focal Loss 的收益是否集中在某些长度段。")
    p()
    bins = [0, 20, 50, 100, 200, 400, 10 ** 7]
    labels = ["<=20", "21-50", "51-100", "101-200", "201-400", ">400"]
    te = d.te.reset_index(drop=True)
    te["bucket"] = pd.cut(te.n_words, bins=bins, labels=labels)
    p("| 长度段 | 样本量 | 正面率 | CE macroF1 | FL macroF1 | 差值 |")
    p("|---|---|---|---|---|---|")
    for lab in labels:
        sub = te[te.bucket == lab]
        if len(sub) < 200:
            continue
        idx = sub.index.values
        y = sub.label.values
        r_ce = eval_preds(y, (models["CE 基线 (自实现)"].predict_proba(Xte[idx]) >= 0.5).astype(int))
        r_fl = eval_preds(y, (models["FL α=0.25 γ=2"].predict_proba(Xte[idx]) >= 0.5).astype(int))
        p(f"| {lab} | {len(sub)} | {y.mean():.3f} | {r_ce['macro_f1']:.4f} | "
          f"{r_fl['macro_f1']:.4f} | {r_fl['macro_f1']-r_ce['macro_f1']:+.4f} |")
    p()


if __name__ == "__main__":
    main()
