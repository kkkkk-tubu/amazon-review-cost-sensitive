import pandas as pd, html, datetime as dt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
out = open(ROOT / "reports" / "diagnosis_log.txt", "w", encoding="utf-8")
def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    out.write(s + "\n")

df = pd.read_csv(ROOT / "Reviews.csv")
p("=== shape ===", df.shape)
p("\n=== dtypes ===\n", df.dtypes)
p("\n=== na ===\n", df.isna().sum())

p("\n=== Id: unique?", df.Id.is_unique, "| min", df.Id.min(), "max", df.Id.max(),
  "| monotonic", df.Id.is_monotonic_increasing)

p("\n=== Score dist ===\n", df.Score.value_counts().sort_index())
p("score outside 1..5:", int(((df.Score < 1) | (df.Score > 5)).sum()))

tmin, tmax = df.Time.min(), df.Time.max()
p("\n=== Time ===", tmin, tmax)
p("  as date:", dt.datetime.utcfromtimestamp(tmin), "->", dt.datetime.utcfromtimestamp(tmax))

num, den = df.HelpfulnessNumerator, df.HelpfulnessDenominator
p("\n=== Helpfulness ===")
p("  <0 numerator:", int((num < 0).sum()), "| <0 denominator:", int((den < 0).sum()))
p("  numerator > denominator:", int((num > den).sum()))
p("  denominator==0 (no votes):", int((den == 0).sum()), f"({(den==0).mean():.1%})")
p("  numerator==0 & den>0 (unhelpful):", int(((num == 0) & (den > 0)).sum()))
p("  ratio==1 (all helpful):", int(((num == den) & (den > 0)).sum()))

p("\n=== duplicates ===")
full = df.drop(columns=["Id"])
p("  dup ignoring Id:", int(full.duplicated().sum()))
for sub in [["UserId", "ProductId", "Time"], ["UserId", "ProductId", "Time", "Text"],
            ["ProductId", "UserId", "Score", "Summary", "Text"],
            ["UserId", "ProfileName", "Time", "Text"], ["Summary", "Text"], ["Text"],
            ["UserId", "ProductId"]]:
    p(f"  dup on {sub}: {int(df.duplicated(subset=sub).sum())}")

p("\n=== Text / Summary ===")
for c in ["Text", "Summary"]:
    s = df[c].astype(str).str.strip()
    p(f"  {c}: empty/blank={int((s=='').sum())} len_min={s.str.len().min()} "
      f"len_med={int(s.str.len().median())} len_max={s.str.len().max()} "
      f"len<10={int((s.str.len()<10).sum())} len>5000={int((s.str.len()>5000).sum())}")

p("\n=== ProfileName ===")
pn = df.ProfileName.astype(str).str.strip()
p("  nunique:", pn.nunique(), "| empty:", int((pn == "").sum()))
p("  top10:\n", pn.value_counts().head(10))
p("  UserId nunique:", df.UserId.nunique(), "| ProductId nunique:", df.ProductId.nunique())
p("  same UserId -> multiple ProfileName:", int((df.groupby('UserId').ProfileName.nunique() > 1).sum()))

p("\n=== html / encoding artifacts ===")
txt = df.Text.astype(str)
summ = df.Summary.astype(str)
pats = {
    "literal <br />": r"<br\s*/?>",
    "&amp;": r"&amp;",
    "&#34;": r"&#34;",
    "&quot;": r"&quot;",
    "&lt;": r"&lt;",
    "&gt;": r"&gt;",
    "&#39;": r"&#39;",
    "any html tag": r"<[^>]+>",
    "any &entity;": r"&[a-zA-Z#0-9]+;",
    "\\r or \\n": r"[\r\n]",
    "non-ascii": r"[^\x00-\x7f]",
    "url http": r"https?://",
    "repeated space(2+)": r"\s{2,}",
}
for name, pat in pats.items():
    t = int(txt.str.contains(pat, regex=True, na=False).sum())
    s = int(summ.str.contains(pat, regex=True, na=False).sum())
    p(f"  {name:22s} Text={t:7d} Summary={s:7d}")

p("\n=== sample unescaped check ===")
ex = txt[txt.str.contains(r"&amp;", na=False)].head(2)
for e in ex:
    p("  RAW:", e[:160].replace("\n", " "))
    p("  ESC:", html.unescape(e)[:160].replace("\n", " "))

p("\n=== reviews per product ===")
vc = df.ProductId.value_counts()
p("  products:", len(vc), "| reviews/product mean %.2f median %d max %d" % (vc.mean(), vc.median(), vc.max()))
p("  products with 1 review:", int((vc == 1).sum()))
p("\n=== reviews per user ===")
uv = df.UserId.value_counts()
p("  users:", len(uv), "| mean %.2f median %d max %d" % (uv.mean(), uv.median(), uv.max()))
p("  users with >=50 reviews:", int((uv >= 50).sum()))

p("\n=== head ===")
p(df.head(3).to_string())
out.close()
