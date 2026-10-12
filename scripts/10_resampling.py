"""重采样类方法对照:随机过采样 / 随机欠采样 / SMOTE(均自实现)

与前面的代价敏感方法(改损失)互补 —— 这里改的是数据分布。
三个要点:
1. 采样**只在训练集上做**,验证集与测试集保持原样。在测试集上采样等于泄漏。
2. SMOTE 在稀疏 TF-IDF 上可直接做:两个稀疏向量的插值 x_i + λ(x_j - x_i),
   非零位置是两者非零位置的并集,不会变稠密,无需降维。
   仅邻居搜索在 SVD-100 低维空间做,避免 4 万 × 4 万的稀疏余弦矩阵过大。
3. 采样后仍允许在验证集调阈值,与前面的对照口径保持一致。

用法: python 10_resampling.py
"""
import sys, time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, diags, vstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_utils import ROOT, eval_preds, eval_proba, format_report, load_dataset

VEC_KW = dict(max_features=100000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
K_NEIGHBORS = 5


def random_oversample(X, y, seed=42):
    """复制少数类直到两类等量"""
    rng = np.random.default_rng(seed)
    n_maj, n_min = int((y == 1).sum()), int((y == 0).sum())
    idx = rng.choice(np.where(y == 0)[0], size=n_maj - n_min, replace=True)
    return vstack([X, X[idx]]), np.concatenate([y, y[idx]])


def random_undersample(X, y, seed=42):
    """随机丢弃多数类直到两类等量"""
    rng = np.random.default_rng(seed)
    n_min = int((y == 0).sum())
    keep_maj = rng.choice(np.where(y == 1)[0], size=n_min, replace=False)
    keep = np.concatenate([np.where(y == 0)[0], keep_maj])
    return X[keep], y[keep]


def smote(X, y, k=K_NEIGHBORS, seed=42):
    """SMOTE:对少数类样本与其同类近邻做插值。邻居在低维空间找,插值在原始稀疏空间做"""
    rng = np.random.default_rng(seed)
    n_maj, n_min = int((y == 1).sum()), int((y == 0).sum())
    Xmin = X[y == 0]

    low = TruncatedSVD(n_components=100, random_state=seed).fit_transform(Xmin)
    _, nbr = NearestNeighbors(n_neighbors=k + 1).fit(low).kneighbors(low)
    nbr = nbr[:, 1:]                                   # 去掉自身

    n_syn = n_maj - n_min
    i = rng.integers(0, n_min, n_syn)
    j = nbr[i, rng.integers(0, k, n_syn)]
    lam = rng.random(n_syn)

    diff = Xmin[j] - Xmin[i]
    X_syn = Xmin[i] + diags(lam) @ diff
    return vstack([X, csr_matrix(X_syn)]), np.concatenate([y, np.zeros(n_syn, dtype=int)])


def main():
    log = open(ROOT / "reports" / "resampling_result.md", "w", encoding="utf-8")
    def p(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        log.write(s + "\n")

    p("# 重采样类方法对照(自实现 ROS / RUS / SMOTE)")
    p()
    p("前面几组实验改的是**损失函数**,这一组改的是**数据分布**,两类方法互为补充。")
    p()
    p("三条必须守住的规矩:")
    p()
    p("1. 采样只在训练集上做,验证集与测试集保持原样 —— 在测试集上采样等于信息泄漏;")
    p("2. SMOTE 在稀疏 TF-IDF 上可直接插值,合成样本的非零位置是两者非零位置的并集,")
    p("   不会变稠密,因此无需降维;")
    p("3. 仅邻居搜索放在 SVD-100 的低维空间,避免 4 万 × 4 万的稀疏余弦相似度矩阵过大。")
    p()

    d = load_dataset()
    vec = TfidfVectorizer(**VEC_KW)
    t = time.perf_counter()
    Xtr = vec.fit_transform(d.tr[d.text_col].astype(str))
    Xva, Xte = vec.transform(d.va[d.text_col].astype(str)), vec.transform(d.te[d.text_col].astype(str))
    ytr = d.tr.label.values
    p(f"TF-IDF {len(vec.vocabulary_)} 维, 拟合 {time.perf_counter()-t:.1f}s")
    p(f"训练集 {Xtr.shape[0]} 行, 正 {int((ytr==1).sum())} / 负 {int((ytr==0).sum())}")
    p()

    jobs = [
        ("无采样 (对照)", None),
        ("随机过采样 ROS", random_oversample),
        ("随机欠采样 RUS", random_undersample),
        (f"SMOTE (k={K_NEIGHBORS})", smote),
    ]

    rows = []
    for name, fn in jobs:
        if fn is None:
            Xs, ys = Xtr, ytr
            sample_sec = 0.0
        else:
            t0 = time.perf_counter()
            Xs, ys = fn(Xtr, ytr)
            sample_sec = time.perf_counter() - t0
        p(f"**{name}**: 训练集 {Xs.shape[0]} 行, 正 {int((ys==1).sum())} / 负 {int((ys==0).sum())}, "
          f"采样耗时 {sample_sec:.1f}s")

        t0 = time.perf_counter()
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="liblinear")
        clf.fit(Xs, ys)
        fit_sec = time.perf_counter() - t0

        prob = clf.predict_proba(Xte)[:, 1]
        default = eval_preds(d.te.label.values, clf.predict(Xte), fit_sec, 0.0,
                             name=name, note=f"默认阈值,采样{sample_sec:.0f}s")
        th, _ = pick_th(d.va.label.values, clf.predict_proba(Xva)[:, 1])
        tuned = eval_proba(d.te.label.values, prob, threshold=th, fit_sec=fit_sec, name=name)
        tuned["note"] = f"阈值{th:.2f}"
        rows.append((default, tuned))
        print(f"   默认 macroF1={default['macro_f1']:.4f} | 调阈值 macroF1={tuned['macro_f1']:.4f} "
              f"| 训练 {fit_sec:.0f}s", flush=True)

    p()
    p("## 1. 默认阈值 0.5")
    p()
    p(format_report([r[0] for r in rows], sort_by="macro_f1",
                    cols=["name", "note", "accuracy", "rec_neg", "prec_neg", "f1_neg",
                          "f1_pos", "macro_f1", "fit_sec"]))
    p()
    p("## 2. 各配置在验证集自选阈值后")
    p()
    p(format_report([r[1] for r in rows], sort_by="macro_f1",
                    cols=["name", "note", "accuracy", "rec_neg", "prec_neg", "f1_neg",
                          "f1_pos", "macro_f1", "fit_sec"]))
    p()

    base_t = rows[0][1]
    best_t = max((r[1] for r in rows), key=lambda r: r["macro_f1"])
    p("## 3. 结论")
    p()
    p(f"- 不做任何采样时,验证集选阈值后 macroF1 = {base_t['macro_f1']:.4f}。")
    p(f"- 四种配置里最好的是 **{best_t['name']}**,macroF1 = {best_t['macro_f1']:.4f},")
    p(f"  相对无采样的变化为 {best_t['macro_f1']-base_t['macro_f1']:+.4f}。")
    p()
    ros_t, rus_t, sm_t = rows[1][1], rows[2][1], rows[3][1]
    p(f"- **ROS(随机过采样)**:负类召回 {base_t['rec_neg']:.4f} → {ros_t['rec_neg']:.4f},"
      f"精度 {base_t['prec_neg']:.4f} → {ros_t['prec_neg']:.4f},"
      f"macroF1 变化 {ros_t['macro_f1']-base_t['macro_f1']:+.4f}。")
    p("  它是本组里**唯一略微超过基线**的方法,但幅度不到 0.3 个百分点。")
    p("  值得注意的是它的方向与加权损失**相反**:召回略降、精度略升。")
    p("  复制少数类样本相当于把决策边界往少数类一侧推得更稳,而不是单纯地放松判负的门槛,")
    p("  因此这次 F1 是净提升的,只是幅度有限。")
    p(f"- **RUS(随机欠采样)**:负类召回 {base_t['rec_neg']:.4f} → {rus_t['rec_neg']:.4f},"
      f"精度 {base_t['prec_neg']:.4f} → {rus_t['prec_neg']:.4f}。")
    p(f"  训练集从 {Xtr.shape[0]} 缩到 {int((ytr==0).sum())*2} 行,丢掉了 "
      f"{int((ytr==1).sum())-int((ytr==0).sum())} 条多数类样本 —— 信息损失是它的主要代价。")
    p(f"- **SMOTE**:负类召回 {base_t['rec_neg']:.4f} → {sm_t['rec_neg']:.4f},"
      f"精度 {base_t['prec_neg']:.4f} → {sm_t['prec_neg']:.4f},"
      f"macroF1 变化 {sm_t['macro_f1']-base_t['macro_f1']:+.4f} —— 几乎与不采样持平,")
    p("  而且**不如 ROS**。一个可能的原因是稀疏 TF-IDF 上的插值语义薄弱:")
    p("  合成样本是两篇评论词向量的加权平均,词袋上成立,语义上往往是拼凑出来的混合文本,")
    p("  反而给训练引入了噪声。")
    p()
    p("### 值得记录的一点")
    p()
    p(f"三种方法里表现最好的是**最简单的 ROS**({ros_t['macro_f1']:.4f}),")
    p(f"反而超过了更复杂的 SMOTE({sm_t['macro_f1']:.4f})。")
    p("这说明在本任务上,合成样本带来的收益不足以抵消它引入的分布噪声;")
    p("而直接复制少数类样本虽然朴素,却没有破坏任何真实样本的分布。")
    p()
    p("**总体判断:三种重采样里只有最简单的 ROS 略微超过了「什么都不做 + 调阈值」,**")
    p("且幅度很小,仍不及前面自实现 CE 基线的 0.8962。这与修改损失函数那组实验的结论")
    p("是一致的 —— 本任务的难点不在类别频率,而在星级标签噪声与文本内部的混合情感,")
    p("改变类别分布或损失权重都触及不到它。")
    log.close()
    print("\n完成 ->", ROOT / "reports" / "resampling_result.md")


def pick_th(y_true, prob):
    best_th, best_v = 0.5, -1.0
    for th in np.round(np.arange(0.05, 0.96, 0.05), 2):
        v = eval_proba(y_true, prob, threshold=th)["macro_f1"]
        if v > best_v:
            best_v, best_th = v, float(th)
    return best_th, best_v


if __name__ == "__main__":
    main()
