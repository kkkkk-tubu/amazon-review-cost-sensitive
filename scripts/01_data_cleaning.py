"""Amazon Fine Food Reviews 数据诊断与清洗 (成员B 依赖的共享清洗流程)

输入: Reviews.csv
输出: Reviews_clean.csv / Reviews_binary.csv / cleaning_report.md / figures/*.png

用法:
    python 01_data_cleaning.py
    python 01_data_cleaning.py --dedup none|default|aggressive
"""
import argparse, hashlib, html, re, time, datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "Reviews.csv"
DATA = ROOT / "data"
FIG = ROOT / "reports" / "figures"
SEED = 42

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------- 文本规范化
# 否定词既不进停用词表，也不做任何删除：它们是情感极性反转的关键信号
NEGATIONS = {
    "no", "nor", "not", "never", "none", "nobody", "nothing",
    "neither", "nowhere", "without", "hardly", "barely", "rarely", "seldom",
}
NEG_CONTRACTIONS = {
    "can't": "can not", "cannot": "can not", "won't": "will not",
    "shan't": "shall not", "ain't": "is not", "don't": "do not",
    "doesn't": "does not", "didn't": "did not", "isn't": "is not",
    "wasn't": "was not", "aren't": "are not", "weren't": "were not",
    "hasn't": "has not", "haven't": "have not", "hadn't": "had not",
    "shouldn't": "should not", "wouldn't": "would not",
    "couldn't": "could not", "mustn't": "must not", "needn't": "need not",
}
NEG_RE = re.compile("|".join(re.escape(k) for k in sorted(NEG_CONTRACTIONS, key=len, reverse=True)))

BASE_STOPWORDS = set("""
a about above after again against all am an and any are as at be because been before being below
between both but by could did do does doing down during each few for from further had has have
having he her here hers herself him himself his how i if in into is it its itself just me more
most my myself of off on once only or other our ours ourselves out over own same she should so
some such than that the their theirs them themselves then there these they this those through to
too under until up very was we were what when where which while who whom why will with would you
your yours yourself yourselves very really quite also however therefore thus
""".split())
STOPWORDS = BASE_STOPWORDS - NEGATIONS

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
NEG_WINDOW = 3  # 否定词后多少个词进入否定作用域
NEG_TRIGGERS = {"not", "no", "never", "nor", "cannot", "nothing", "none", "neither", "hardly", "barely", "rarely"}


def strip_html(s: str) -> str:
    s = html.unescape(s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    return re.sub(r"&[a-zA-Z#0-9]*;?", " ", s)


def expand_negations(s: str) -> str:
    return NEG_RE.sub(lambda m: NEG_CONTRACTIONS[m.group(0)], s)


def collapse(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def clean_text(raw, lower=True, keep_punct=False):
    """HTML/URL 剥离 + 缩写否定还原 + 小写 + 标点归一。保留 not/no/never 等否定词。"""
    s = strip_html(str(raw))
    s = expand_negations(s.lower() if lower else s)
    s = re.sub(r"(!+|\?+)", r" \1 ", s)          # 情感强度信号：保留重复 ! 和 ?
    if keep_punct:
        s = re.sub(r"[^a-z0-9!?.,;:'\-]", " ", s)
    else:
        s = re.sub(r"[^a-z0-9!?]", " ", s)
    return collapse(s)


def tokenize(s: str, drop_stopwords=False):
    toks = TOKEN_RE.findall(s)
    if drop_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return toks


def mark_negation(tokens: list) -> list:
    """否定作用域标注：否定词后 NEG_WINDOW 个词加 NEG_ 前缀，供后续加权方案使用。"""
    out, scope = [], 0
    for t in tokens:
        if t in NEG_TRIGGERS:
            scope = NEG_WINDOW
            out.append(t)
        elif scope > 0:
            out.append("NEG_" + t)
            scope -= 1
        else:
            out.append(t)
    return out


def text_hash(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dedup", default="default", choices=["none", "default", "aggressive"])
    ap.add_argument("--max-pairs", type=int, default=3, help="同一(UserId,ProductId)最多保留的评论数")
    args = ap.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    log = open(ROOT / "reports" / "cleaning_log.txt", "w", encoding="utf-8")

    def p(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        log.write(s + "\n")

    t0 = time.time()
    df = pd.read_csv(RAW)
    p(f"读入原始文件: {RAW.name}  shape={df.shape}")

    steps = []
    steps.append(("原始行数", len(df)))

    # ---- 1. 结构合法性 ----
    bad_help = df.HelpfulnessNumerator > df.HelpfulnessDenominator
    p(f"\n[1] 结构合法性: Helpfullness 分子>分母 {int(bad_help.sum())} 行 -> 修正为 min(分子,分母)")
    df.loc[bad_help, "HelpfulnessNumerator"] = df.loc[bad_help, ["HelpfulnessNumerator", "HelpfulnessDenominator"]].min(axis=1)

    bad_time = (df.Time < 0) | (df.Time > time.time())
    p(f"    越界时间戳 {int(bad_time.sum())} 行")
    df["Datetime"] = pd.to_datetime(df.Time, unit="s", utc=True)
    df["Year"] = df.Datetime.dt.year

    naive_summary = df.Summary.isna()
    p(f"    Summary 缺失 {int(naive_summary.sum())} 行, ProfileName 缺失 {int(df.ProfileName.isna().sum())} 行 -> 补空串,不删样本")
    df["Summary"] = df.Summary.fillna("")
    df["ProfileName"] = df.ProfileName.fillna("")

    # ---- 2. 去重口径对比 ----
    key_def = ["UserId", "ProfileName", "Time", "Text"]
    cnt = {
        "none": 0,
        "default": int(df.duplicated(subset=key_def).sum()),
        "aggressive_default": int(df.duplicated(subset=key_def).sum()),
    }
    p("\n[2] 去重口径对比(保留首条):")
    for name, sub in [
        ("除 Id 外完全一致", ["ProductId", "UserId", "ProfileName", "HelpfulnessNumerator", "HelpfulnessDenominator", "Score", "Time", "Summary", "Text"]),
        ("UserId+ProfileName+Time+Text (默认口径)", key_def),
        ("UserId+Time+Text", ["UserId", "Time", "Text"]),
        ("UserId+ProductId+Time+Text", ["UserId", "ProductId", "Time", "Text"]),
        ("Text 全局去重 (激进)", ["Text"]),
        ("Summary+Text", ["Summary", "Text"]),
    ]:
        k = int(df.duplicated(subset=sub).sum())
        p(f"    {name:42s} 重复 {k:7d} 行 -> 余 {len(df)-k:7d}")
        if name.startswith("Text 全局"):
            cnt["aggressive"] = k

    dedup_key = key_def
    if args.dedup == "none":
        p("\n    --dedup none: 跳过去重")
    elif args.dedup == "aggressive":
        dedup_key = ["UserId", "ProfileName", "Time", "Text"]
        df = df.drop_duplicates(subset=dedup_key, keep="first")
        before = len(df)
        df = df.drop_duplicates(subset=["Text"], keep="first")
        p(f"\n    激进去重: 先按 {dedup_key} 删 {cnt['default']}, 再按 Text 全局删 {before-len(df)}")

    if args.dedup == "default":
        df = df.drop_duplicates(subset=dedup_key, keep="first")
        p(f"\n    采用口径 {dedup_key} -> 删除 {cnt['default']} 行")
    steps.append(("去重", len(df)))

    # ---- 3. 文本规范化 ----
    p("\n[3] 文本规范化: HTML unescape / 去标签 / 去URL / 缩写否定还原 / 小写 / 标点归一")
    df["text_clean"] = df.Text.map(clean_text)
    df["summary_clean"] = df.Summary.map(clean_text)
    empty = df.text_clean.str.len() < 3
    p(f"    规范化后长度<3字符的行 {int(empty.sum())} 行 -> 剔除")
    df = df[~empty]
    steps.append(("去空文本", len(df)))

    df["n_words"] = df.text_clean.map(lambda s: len(tokenize(s)))
    df["n_words_nostop"] = df.text_clean.map(lambda s: len(tokenize(s, drop_stopwords=True)))
    df["tokens"] = df.text_clean.map(lambda s: " ".join(tokenize(s)))
    df["tokens_nostop"] = df.text_clean.map(lambda s: " ".join(tokenize(s, drop_stopwords=True)))
    df["tokens_neg"] = df.text_clean.map(lambda s: " ".join(mark_negation(tokenize(s))))

    # ---- 4. 标签 ----
    df["label"] = np.select([df.Score <= 2, df.Score >= 4], [0, 1], default=2)  # 0负 1正 2中性
    df["label_name"] = df.label.map({0: "neg", 1: "pos", 2: "neutral"})

    # ---- 5. 防泄漏 / 噪声标记 ----
    df["dup_group"] = df.text_clean.map(text_hash)
    conflict = df.groupby("dup_group").label.nunique()
    df["label_conflict"] = df.dup_group.map(conflict) > 1
    p(f"\n[5] 相同规范化文本但二分类标签不同(评分噪声)的行: {int(df.label_conflict.sum())}")
    p(f"    同-text 组数 {df.dup_group.nunique()}, 用于分组切分防止跨集泄漏")

    # ---- 6. 输出 ----
    clean_cols = ["Id", "ProductId", "UserId", "ProfileName", "Score", "Time", "Datetime", "Year",
                  "HelpfulnessNumerator", "HelpfulnessDenominator", "label", "label_name",
                  "text_clean", "summary_clean", "n_words", "n_words_nostop",
                  "dup_group", "label_conflict"]
    df[clean_cols].to_csv(DATA / "Reviews_clean.csv", index=False, encoding="utf-8-sig")

    bin_df = df[df.label != 2].copy()
    steps.append(("剔除 Score=3", len(bin_df)))
    bin_df[clean_cols].to_csv(DATA / "Reviews_binary.csv", index=False, encoding="utf-8-sig")
    p(f"\n[6] 输出 Reviews_clean.csv ({len(df)} 行, 含中性) / Reviews_binary.csv ({len(bin_df)} 行)")

    tok_cols = ["Id", "label", "tokens", "tokens_nostop", "tokens_neg", "n_words", "dup_group", "Year", "label_conflict"]
    bin_df[tok_cols].to_csv(DATA / "Reviews_tokens.csv", index=False, encoding="utf-8-sig")

    # ---- 7. 统计与图表 ----
    raw_df = pd.read_csv(RAW, usecols=["Score", "Text", "Time"])
    write_stats(p, df, bin_df, raw_df, steps)
    draw_plots(p, df, bin_df, raw_df)

    p(f"\n总耗时 {time.time()-t0:.1f}s | 随机种子 {SEED}")
    log.close()
    print("\n完成 ->", DATA)


def write_stats(p, df, bin_df, raw_df, steps):
    p("\n" + "=" * 60)
    p("清洗阶段行数变化:")
    for name, n in steps:
        p(f"    {name:14s} {n:7d}")

    vc = df.Score.value_counts().sort_index()
    p("\n清洗后评分分布:")
    for s, c in vc.items():
        p(f"    Score={s}: {c:7d} ({c/len(df):.2%})")

    neg, pos, neu = (bin_df.label == 0).sum(), (bin_df.label == 1).sum(), (df.label == 2).sum()
    p(f"\n二分类样本: 负(1,2)={neg}  正(4,5)={pos}  中性(3)={neu} (已剔除并记录)")
    p(f"    正负比 {pos/neg:.2f} : 1   少数类占比 {neg/(neg+pos):.2%}  <- 成员B 的类别不平衡基数")
    p(f"    多数类基线 Accuracy = {max(neg,pos)/(neg+pos):.4f}")

    p("\n文本长度词数 (清洗后):")
    for c in ["n_words", "n_words_nostop"]:
        q = df[c].quantile([0, .05, .25, .5, .75, .95, 1])
        p(f"    {c:16s} min={q[0]:.0f} p5={q[.05]:.0f} p25={q[.25]:.0f} med={q[.5]:.0f} "
          f"p75={q[.75]:.0f} p95={q[.95]:.0f} max={q[1]:.0f}")

    p("\n正负类的文本长度差异:")
    for lab, g in bin_df.groupby("label"):
        p(f"    label={lab} n={len(g):6d} mean_words={g.n_words.mean():6.1f} median={g.n_words.median():5.0f}")

    p("\n按年份:")
    yr = df.groupby("Year").agg(n=("Id", "size"), pos_ratio=("label", lambda s: (s == 1).mean()))
    yr["n"] = yr.n.astype(int)
    for y, r in yr.iterrows():
        if r.n >= 100:
            p(f"    {y}: n={int(r.n):6d}  pos%={r.pos_ratio:.1%}")

    p(f"\n产品数 {df.ProductId.nunique()}  用户数 {df.UserId.nunique()}")
    p(f"词表规模(全tokens) {len(set(' '.join(bin_df.tokens.sample(min(200000,len(bin_df)), random_state=SEED)).split()))} (抽样20万)")


def draw_plots(p, df, bin_df, raw_df, ):
    # 1 评分分布 前后
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    raw_df.Score.value_counts().sort_index().plot(kind="bar", ax=ax[0], color="#4c72b0")
    ax[0].set_title("清洗前评分分布")
    df.Score.value_counts().sort_index().plot(kind="bar", ax=ax[1], color="#55a868")
    ax[1].set_title("清洗后评分分布")
    for a in ax:
        a.set_xlabel("Score")
        a.set_ylabel("评论数")
        a.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_score_dist.png", dpi=140, bbox_inches="tight")

    # 2 类别不平衡
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    cls = df.label_name.value_counts()
    ax[0].bar(cls.index, cls.values, color=["#c44e52", "#55a868", "#8c8c8c"])
    ax[0].set_title("三分类标签分布 (Score=3 为中性)")
    for i, v in enumerate(cls.values):
        ax[0].text(i, v * 1.01, f"{v:,}", ha="center", fontsize=9)
    b = bin_df.label.map({0: "负 (1,2)", 1: "正 (4,5)"}).value_counts()
    ax[1].pie(b.values, labels=[f"{k}\n{v:,}\n{v/b.sum():.1%}" for k, v in b.items()],
              colors=["#c44e52", "#55a868"], startangle=90, textprops={"fontsize": 9})
    ax[1].set_title("二分类正负占比")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_class_imbalance.png", dpi=140, bbox_inches="tight")

    # 3 文本长度
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(bin_df.n_words, bins=100, range=(0, 600), color="#4c72b0")
    ax[0].set_title("评论词数分布 (截断至600)")
    ax[0].set_xlabel("词数")
    ax[0].set_ylabel("评论数")
    for lab, c, n in [(0, "#c44e52", "负面"), (1, "#55a868", "正面")]:
        s = bin_df[bin_df.label == lab].n_words
        ax[1].hist(np.log1p(s), bins=60, alpha=.6, label=f"{n} n={len(s):,}", color=c)
    ax[1].set_title("正负类词数分布 (log1p)")
    ax[1].set_xlabel("log(1+词数)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(FIG / "fig3_text_length.png", dpi=140, bbox_inches="tight")

    # 4 时间分布
    yr = df.groupby("Year").agg(n=("Id", "size"), pos=("label", lambda s: (s == 1).mean()))
    yr = yr[yr.n >= 200]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(yr.index, yr.n, color="#4c72b0")
    ax.set_ylabel("评论数")
    ax2 = ax.twinx()
    ax2.plot(yr.index, yr.pos, color="#c44e52", marker="o")
    ax2.set_ylabel("正面占比", color="#c44e52")
    ax.set_title("逐年评论量与正面占比 (>=200条/年)")
    ax.set_xlabel("年份")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_time_trend.png", dpi=140, bbox_inches="tight")

    # 5 长度 vs 情感
    bin_df2 = bin_df.copy()
    bins = [0, 20, 50, 100, 200, 400, 10**6]
    labels = ["<=20", "21-50", "51-100", "101-200", "201-400", ">400"]
    bin_df2["bucket"] = pd.cut(bin_df2.n_words, bins=bins, labels=labels, right=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    agg = bin_df2.groupby("bucket").agg(n=("label", "size"), pos_rate=("label", "mean"))
    ax.bar(agg.index.astype(str), agg.pos_rate, color="#8172b2")
    ax.axhline(bin_df.label.mean(), ls="--", color="#c44e52", label=f"整体正例率 {bin_df.label.mean():.3f}")
    for i, (r, n) in enumerate(zip(agg.pos_rate, agg.n)):
        ax.text(i, r + .01, f"{r:.2f}\nn={n:,}", ha="center", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("评论词数区间")
    ax.set_ylabel("正面比例")
    ax.set_title("评论长度与情感极性的关系")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "fig5_length_vs_polarity.png", dpi=140, bbox_inches="tight")

    p("\n图表已输出: " + ", ".join(sorted(x.name for x in FIG.glob("*.png"))))


if __name__ == "__main__":
    main()
