import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data"
out = []
def p(*a): out.append(" ".join(str(x) for x in a))

tok = pd.read_csv(D / "Reviews_tokens.csv", nrows=50000)
clean = pd.read_csv(D / "Reviews_clean.csv", nrows=50000)

p("tokens cols:", list(tok.columns))
p("clean cols:", list(clean.columns))

p("\n--- 抽样 3 条比较 ---")
for i in [0, 1, 2]:
    r = clean.iloc[i]
    p(f"[Score={r.Score} label={r.label_name}]")
    p("  text_clean:", str(r.text_clean)[:150])
    c = tok[tok.Id == r.Id].iloc[0]
    p("  tokens    :", str(c.tokens)[:150])
    p("  nostop    :", str(c.tokens_nostop)[:130])
    p("  neg       :", str(c.tokens_neg)[:130])

p("\n--- 校验 ---")
p("text_clean 残留 html 标签:", int(clean.text_clean.str.contains(r"<[^>]+>", na=False).sum()))
p("text_clean 残留 URL     :", int(clean.text_clean.str.contains(r"https?://|www\.", na=False).sum()))
p("text_clean 残留 html实体:", int(clean.text_clean.str.contains(r"&[a-z]+;", na=False).sum()))
p("text_clean 含大写       :", int(clean.text_clean.str.contains(r"[A-Z]", na=False).sum()))
p("text_clean 字母/数字/!? 之外字符的行数:",
  int(clean.text_clean.str.contains(r"[^a-z0-9!? ]", regex=True, na=False).sum()))

m = clean.reset_index(drop=True)
p("label 与 Score 一致:", bool(((m.Score <= 2) == (m.label == 0)).all() and ((m.Score >= 4) == (m.label == 1)).all()))

cnt_present = 0
for s in tok.tokens_nostop.head(5000):
    ws = set(str(s).split())
    if "not" in ws or "no" in ws:
        cnt_present += 1
p("\n前5000条 nostop 中含 not/no 的条数:", cnt_present, "(否定词应被保留)")

p("出现 NEG_ 前缀标记的条数:", int(tok.tokens_neg.str.contains("NEG_").sum()), "/", len(tok))
p("n_words 最小值:", int(clean.n_words.min()), " 最大值:", int(clean.n_words.max()))
p("n_words==0 的行数:", int((clean.n_words == 0).sum()))

ex = clean[clean.text_clean.str.contains(r"\bnot\b", na=False)].head(3)
p("\n--- 含 not 的原文截断示例 ---")
for _, r in ex.iterrows():
    p(f"  {str(r.text_clean)[:110]}")

sample_neg = tok[tok.tokens_neg.str.contains("NEG_")].head(3)
p("\n--- NEG_ 标记示例 ---")
for _, r in sample_neg.iterrows():
    p("  ", str(r.tokens_neg)[:160])

open(ROOT / "reports" / "validate_output.txt", "w", encoding="utf-8").write("\n".join(out))
print("\n".join(out))
