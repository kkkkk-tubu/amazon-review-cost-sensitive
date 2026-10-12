"""划分脚本:生成全组共用的 train / val / test 对照表

设计要点
- 在「组」层面划分,而不是逐行:同 dup_group(相同文本)的样本必须整组落在同一集合
- 组层面用 StratifiedShuffleSplit,保证三个集合的正负比一致
- 支持三种模式:random(默认) / user(按用户隔离) / temporal(按时间顺序)

用法:
    python 04_make_split.py
    python 04_make_split.py --mode temporal
    python 04_make_split.py --all      # 生成全部模式 + 多随机种子,用于对照实验
"""
import argparse, time
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SEED = 42
TRAIN_RATIO, VAL_RATIO = 0.70, 0.15   # test 自动为 0.15
USER_SEEDS = [42, 1, 2024]


def group_frame(df, key):
    """把行折叠成组:组内 label 取第一条,用于组层面的分层"""
    return df.groupby(key, sort=False).agg(
        label=("label", "first"),
        tmin=("Time", "min"),
        n=("Id", "size"),
    ).reset_index().rename(columns={key: "gid"})


def split_groups(gdf, seed):
    """组层面两次分层切分 -> {gid: split}。返回后必须做泄漏检查"""
    y = gdf.label.values
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - TRAIN_RATIO, random_state=seed)
    tr_idx, rest_idx = next(sss.split(gdf[["label"]], y))
    rest = gdf.iloc[rest_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    va_sub, te_sub = next(sss2.split(rest[["label"]], rest.label.values))
    va_idx = rest_idx[va_sub]
    te_idx = rest_idx[te_sub]

    mapping = pd.Series(index=gdf.gid.values, data="train", dtype=object)
    mapping.iloc[va_idx] = "val"
    mapping.iloc[te_idx] = "test"
    return mapping


_ORDER = {"train": 0, "val": 1, "test": 2}


def reconcile(df, arr, key, strategy="mode"):
    """同一 key 的样本必须落在同一集合。
    mode  : 取组内多数所在的集合(用于 user 模式)
    latest: 归到组内最晚出现的集合,保证 train 不含任何延伸到未来的组(用于 temporal)
    """
    g = pd.DataFrame({"k": df[key].values, "s": arr})
    cnt = g.groupby(["k", "s"], observed=True).size().rename("n").reset_index()
    cnt["pri"] = cnt.s.map(_ORDER)
    if strategy == "latest":
        pick = cnt.sort_values(["k", "pri"], ascending=[True, False]).drop_duplicates("k")
    else:
        pick = cnt.sort_values(["k", "n", "pri"], ascending=[True, False, True]).drop_duplicates("k")
    new = g.k.map(pick.set_index("k").s).values
    return new, int((new != np.asarray(arr)).sum())


def temporal_split(df, key):
    """按行时间顺序切,再 reconcile:保证 train 完全来自较早的时间片段"""
    order = np.argsort(df.Time.values, kind="mergesort")
    n = len(df)
    n_tr, n_va = int(n * TRAIN_RATIO), int(n * VAL_RATIO)
    arr = np.empty(n, dtype=object)
    arr[order[:n_tr]] = "train"
    arr[order[n_tr:n_tr + n_va]] = "val"
    arr[order[n_tr + n_va:]] = "test"
    arr, changed = reconcile(df, arr, key, strategy="latest")
    return arr, changed


_CACHE = {}


def build(df, key, seed=SEED, mode="random"):
    if mode == "temporal":
        arr, _ = temporal_split(df, key)
        return arr
    if key not in _CACHE:
        _CACHE[key] = group_frame(df, key)
    gdf = _CACHE[key]
    arr = df[key].map(split_groups(gdf, seed)).values
    # UserId 分组无法自动隔离跨用户的重复文本,收敛一次把硬约束补齐
    if key != "dup_group":
        arr, _ = reconcile(df, arr, "dup_group")
    return arr


def leak_check(df, split_col, p):
    """同组样本是否跨集合 / 用户是否跨 train 与 test"""
    dup_cross = df.groupby("dup_group")[split_col].nunique().pipe(lambda s: int((s > 1).sum()))
    tr = set(df.loc[df[split_col] == "train", "UserId"])
    te = set(df.loc[df[split_col] == "test", "UserId"])
    p(f"    同 dup_group 跨集合的组数: {dup_cross}  (必须为 0)")
    p(f"    同时出现在 train 与 test 的用户数: {len(tr & te)}  (user 模式应为 0)")
    return dup_cross


def describe(df, split_col, p, extra_tag=""):
    p(f"\n  [{split_col}]{extra_tag}")
    for s in ["train", "val", "test"]:
        sub = df[df[split_col] == s]
        pos = (sub.label == 1).mean()
        p(f"    {s:5s} n={len(sub):6d} ({len(sub)/len(df):5.1%})  正例率={pos:.4f}  "
          f"正={int((sub.label==1).sum()):6d} 负={int((sub.label==0).sum()):5d}  "
          f"时间 {date(sub.Time.min())} ~ {date(sub.Time.max())}")
    leak_check(df, split_col, p)


def overlap_report(df, split_col, p):
    """时间划分专用:同一个 dup_group 的文本可能横跨多年,组完整性与时间严格单调不能两全"""
    tr_end = df.loc[df[split_col] == "train", "Time"].max()
    va_end = df.loc[df[split_col] == "val", "Time"].max()
    a = int((df.loc[df[split_col] == "val", "Time"] <= tr_end).sum())
    b = int((df.loc[df[split_col] == "test", "Time"] <= va_end).sum())
    p(f"    时间重叠: train 截止 {date(tr_end)},"
      f" val 中早于该时刻 {a} 行, test 中早于 val 截止 {b} 行, 合计 {a+b} 行 ({100*(a+b)/len(df):.3f}%)")
    p(f"    成因: 96 组 dup_group 的文本被不同用户在跨度>1年的时间里重复提交,"
      f"整组锁定在较晚集合后仍带回少量早期行")


def date(t):
    return pd.to_datetime(t, unit="s").date()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="random", choices=["random", "user", "temporal"])
    ap.add_argument("--all", action="store_true", help="生成全部模式与多随机种子")
    args = ap.parse_args()

    t0 = time.time()
    log = open(ROOT / "reports" / "split_log.txt", "w", encoding="utf-8")
    def p(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        log.write(s + "\n")

    df = pd.read_csv(DATA / "Reviews_binary.csv",
                     usecols=["Id", "UserId", "ProductId", "label", "Time", "Year", "dup_group"])
    p(f"建模表 Reviews_binary.csv  shape={df.shape}   种子 {SEED}")
    p(f"总体: 组数(dup_group)={df.dup_group.nunique()}  用户数={df.UserId.nunique()}  正例率={df.label.mean():.4f}")

    if args.all:
        out = df[["Id"]].copy()
        for seed in USER_SEEDS:
            col = f"random_{seed}"
            df[col] = out[col] = build(df, "dup_group", seed, "random")
            describe(df, col, p, f"  random seed={seed}")

        df["user"] = out["user"] = build(df, "UserId", SEED, "random")
        describe(df, "user", p, "  user 模式:按 UserId 隔离")

        df["temporal"] = out["temporal"] = build(df, "dup_group", mode="temporal")
        describe(df, "temporal", p, "  temporal 模式:按时间顺序")
        overlap_report(df, "temporal", p)

        out.to_csv(DATA / "splits_multi.csv", index=False, encoding="utf-8-sig")

        base = out[["Id", f"random_{SEED}"]].rename(columns={f"random_{SEED}": "split"})
        base.to_csv(DATA / "splits.csv", index=False, encoding="utf-8-sig")
        p(f"\n输出 data/splits.csv (默认 split = random seed={SEED})")
        p(f"输出 data/splits_multi.csv (列 {list(out.columns[1:])})")
    else:
        key = "UserId" if args.mode == "user" else "dup_group"
        df["split"] = build(df, key, SEED, args.mode)
        describe(df, "split", p, f"  mode={args.mode}")
        df[["Id", "split"]].to_csv(DATA / "splits.csv", index=False, encoding="utf-8-sig")
        p(f"\n输出 data/splits.csv  mode={args.mode} seed={SEED}")

    p(f"\n总耗时 {time.time()-t0:.1f}s")
    p("\n说明:划分在「组」层面完成,组再根据 key 映射回行;splits.csv 只存 Id->split,")
    p("      不复制原文,建模时按 Id merge 即可,保证全组拿到完全一致的数据。")
    log.close()


if __name__ == "__main__":
    main()
