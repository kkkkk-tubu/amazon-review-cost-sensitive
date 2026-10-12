"""全组统一评价协议

用法:
    from eval_utils import load_dataset, eval_preds, format_report
    d = load_dataset()                       # 自动 merge 划分表
    row = eval_preds(d.te.label, y_pred, fit_sec=2.1, pred_sec=0.3, name="LogReg")
    print(format_report([row]))

指标口径:pos=1 正面, neg=0 负面。所有 member 用同一份,便于横向成表。
"""
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
NEG, POS = 0, 1

METRIC_COLS = ["name", "accuracy", "prec_pos", "rec_pos", "f1_pos",
               "prec_neg", "rec_neg", "f1_neg", "macro_f1", "fit_sec", "pred_sec"]


MULTI_MODES = ("random_42", "random_1", "random_2024", "user", "temporal")


def load_dataset(mode="split", text_col="tokens_nostop", root=DATA):
    """读取建模集并 merge 划分表。

    mode="split"     -> splits.csv 的默认划分(random seed=42),日常实验用这个
    mode="temporal"  -> splits_multi.csv 的时间划分,做时序漂移对照
    mode="user"      -> 按 UserId 隔离的划分
    mode="random_1" / "random_2024" -> 多随机种子

    返回对象带 tr / va / te 三个属性,列名与原始表一致。
    """
    if mode == "split":
        sp = pd.read_csv(root / "splits.csv")
        col = "split"
    elif mode in MULTI_MODES:
        sp = pd.read_csv(root / "splits_multi.csv", usecols=["Id", mode])
        col = mode
    else:
        raise ValueError(f"mode 必须是 'split' 或 {MULTI_MODES} 之一,收到 {mode!r}")

    tok = pd.read_csv(root / "Reviews_tokens.csv",
                      usecols=["Id", "label", text_col, "n_words", "dup_group", "Year"])
    df = tok.merge(sp.rename(columns={col: "split"}), on="Id", how="inner")

    class DS:
        pass
    d = DS()
    d.all = df
    d.text_col = text_col
    for part in ["train", "val", "test"]:
        setattr(d, {"train": "tr", "val": "va", "test": "te"}[part], df[df.split == part])
    return d


def class_weights(y, n_classes=2):
    """sklearn class_weight='balanced' 同款: n / (k * n_c)"""
    cnt = np.bincount(np.asarray(y, dtype=int), minlength=n_classes)
    w = len(y) / (n_classes * np.maximum(cnt, 1))
    return {POS: w[POS], NEG: w[NEG]}


def majority_rate(y_true):
    return float(max(np.mean(np.asarray(y_true) == 0), np.mean(np.asarray(y_true) == 1)))


def eval_preds(y_true, y_pred, fit_sec=None, pred_sec=None, name="model", note=""):
    """返回一行 dict。TN/FP/FN/TP 以 pos=1 为正类"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=[NEG, POS], zero_division=0)
    _, _, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[NEG, POS], average="macro", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[NEG, POS]).ravel()
    return {
        "name": name,
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "prec_pos": prec[POS], "rec_pos": rec[POS], "f1_pos": f1[POS], "sup_pos": int(sup[POS]),
        "prec_neg": prec[NEG], "rec_neg": rec[NEG], "f1_neg": f1[NEG], "sup_neg": int(sup[NEG]),
        "macro_f1": f1_macro,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "fit_sec": fit_sec, "pred_sec": pred_sec,
        "note": note,
    }


def eval_proba(y_true, y_prob, threshold=0.5, **kw):
    """给定 P(正) 后手动定阈值。代价敏感实验里阈值不是 0.5 时用这个"""
    return eval_preds(y_true, (np.asarray(y_prob) >= threshold).astype(int), **kw)


def confusion_frame(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(np.asarray(y_true), np.asarray(y_pred), labels=[NEG, POS]).ravel()
    return pd.DataFrame(
        [[f"真负 {tn}", f"假正 {fp}"], [f"假负 {fn}", f"真正 {tp}"]],
        index=["真实 负面", "真实 正面"], columns=["预测 负面", "预测 正面"])


REPORT_HEADER = {
    "name": "模型", "n": "样本数", "accuracy": "Accuracy",
    "prec_pos": "P正", "rec_pos": "R正", "f1_pos": "F1正",
    "prec_neg": "P负", "rec_neg": "R负", "f1_neg": "F1负",
    "macro_f1": "macroF1", "fit_sec": "训练s", "pred_sec": "预测s",
    "note": "备注",
}


def format_report(rows, cols=None, sort_by=None, float_fmt=".4f"):
    """把多行 dict 渲染成 markdown 表(不依赖 tabulate)"""
    rows = list(rows)
    if sort_by:
        rows = sorted(rows, key=lambda r: r.get(sort_by) or 0, reverse=True)
    cols = [c for c in (cols or METRIC_COLS) if c in rows[0]]

    def cell(v):
        if v is None:
            return "-"
        if isinstance(v, (float, np.floating)):
            return format(v, float_fmt)
        return str(v)

    head = "| " + " | ".join(REPORT_HEADER.get(c, c) for c in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = ["| " + " | ".join(cell(r.get(c)) for c in cols) + " |" for r in rows]
    return "\n".join([head, sep] + body)


class Timer:
    def __init__(self):
        self.fit = self.pred = None

    def __enter__(self):
        self.fit = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.fit = time.perf_counter() - self.fit

    def predict(self):
        self.pred = time.perf_counter()
        return self

    def stop(self):
        self.pred = time.perf_counter() - self.pred


if __name__ == "__main__":
    y_true = [1] * 80 + [0] * 20
    y_pred = [1] * 70 + [0] * 10 + [1] * 5 + [0] * 15
    r = eval_preds(y_true, y_pred, 1.0, 0.1, "自检")
    print(format_report([r]))
    print(confusion_frame(y_true, y_pred))
    print("\n多数类基线:", majority_rate(y_true))
    print("类别权重:", class_weights(y_true))
