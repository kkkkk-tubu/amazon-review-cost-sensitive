import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "Reviews_binary.csv",
                 usecols=["Id", "Score", "label", "n_words", "n_words_nostop", "dup_group", "Year", "label_conflict", "text_clean"])
out = []
def p(*a): out.append(" ".join(str(x) for x in a))

p("=== 长度分桶 x 正例率 ===")
bins = [0, 20, 50, 100, 200, 400, 10**7]
labels = ["<=20", "21-50", "51-100", "101-200", "201-400", ">400"]
df["bucket"] = pd.cut(df.n_words, bins=bins, labels=labels)
g = df.groupby("bucket", observed=True).agg(n=("label", "size"), pos_rate=("label", "mean"), med_w=("n_words", "median"))
for k, r in g.iterrows():
    p(f"  {k:8s} n={int(r.n):7d} ({r.n/len(df):5.1%})  pos_rate={r.pos_rate:.4f}  median_words={r.med_w:.0f}")
p(f"  整体 pos_rate={df.label.mean():.4f}")

p("\n=== 极端长度 ===")
p("  n_words>=1000 的条数:", int((df.n_words >= 1000).sum()), " 其中正例率 %.3f" % df[df.n_words >= 1000].label.mean())
p("  n_words<=10  的条数:", int((df.n_words <= 10).sum()), " 其中正例率 %.3f" % df[df.n_words <= 10].label.mean())

p("\n=== dup_group 重复度 ===")
gs = df.groupby("dup_group").size()
p("  唯一文本组:", len(gs))
p("  仅出现1次的组:", int((gs == 1).sum()), f"({(gs==1).mean():.1%})")
p("  出现>=2次的组:", int((gs >= 2).sum()), " 覆盖行数:", int(gs[gs >= 2].sum()))
p("  出现>=5次的组:", int((gs >= 5).sum()), " 覆盖行数:", int(gs[gs >= 5].sum()))
p("  最大组大小:", int(gs.max()))
big = gs.idxmax()
p("  最大组的前2条文本:")
for _, r in df[df.dup_group == big].head(2).iterrows():
    p("    ", str(r.text_clean)[:120])
p("\n  标签冲突组数:", int(df[df.label_conflict].dup_group.nunique()), " 涉及行数:", int(df.label_conflict.sum()))
cf = df[df.label_conflict].groupby("dup_group").head(2)
for gid, sub in list(cf.groupby("dup_group"))[:3]:
    p(f"    dup_group={gid}")
    for _, r in sub.iterrows():
        p(f"      Score={r.Score} label={r.label}  {str(r.text_clean)[:95]}")

p("\n=== 最高频文本(模板/垃圾评论候选) ===")
top = gs.sort_values(ascending=False).head(5)
for gid, n in top.items():
    txt = df[df.dup_group == gid].text_clean.iloc[0]
    p(f"  n={n:5d}  {txt[:100]}")

p("\n=== 年度正例率 ===")
yr = df.groupby("Year").agg(n=("label", "size"), pos=("label", "mean"))
for y, r in yr[yr.n >= 200].iterrows():
    p(f"  {y}: n={int(r.n):6d}  pos={r.pos:.4f}")

p("\n=== 停用词影响 ===")
p("  含停用词 词数中位数: %.0f  p95: %.0f" % (df.n_words.median(), df.n_words.quantile(.95)))
p("  去停用词 词数中位数: %.0f  p95: %.0f" % (df.n_words_nostop.median(), df.n_words_nostop.quantile(.95)))
p("  平均保留比例: %.3f" % (df.n_words_nostop.sum() / df.n_words.sum()))

open(ROOT / "reports" / "explore_stats.txt", "w", encoding="utf-8").write("\n".join(out))
print("\n".join(out))
