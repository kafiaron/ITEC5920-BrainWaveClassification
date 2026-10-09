# Aggregate results/folds.csv.
#
#   python summarize.py --protocol cv                        per-subject table, mean ± SD over repeats
#   python summarize.py --protocol cv --ci                   corrected 95% CI instead of SD
#   python summarize.py --protocol cv --metric auc --latex
#   python summarize.py --protocol cv --compare "fbcsp_lda [all]" "ctnet [all] +EA +PT"
#   python summarize.py --protocol curve                     accuracy vs calibration trials
#   python summarize.py --protocol cv --audit                inflation from each reproduced leak
#   python summarize.py --protocol cv --bestof "ctnet [all] +EA" "ctnet [all] +aug:all +EA"
import argparse

import numpy as np
import pandas as pd
from scipy import stats

import config as cfg


def load(args):
    df = pd.read_csv(cfg.RESULTS_DIR / "folds.csv")
    df = df[(df.protocol == args.protocol) & (df.tag == args.tag)]
    if df.empty:
        raise SystemExit("No matching rows. Check --protocol and --tag.")
    return df.drop_duplicates(subset=["method", "subject", "repeat", "fold", "n_train"], keep="last")


# Corrected resampled t statistic for repeated k-fold CV: the variance is inflated by
# n_test / n_train because training sets overlap across folds.
# References: Nadeau & Bengio (2003); Bouckaert & Frank (2004)
def corrected_t(values, test_train_ratio):
    values = np.asarray(values, dtype=float)
    J = len(values)
    if J < 2:
        return values.mean(), np.nan, np.nan
    se = np.sqrt((1 / J + test_train_ratio) * values.var(ddof=1))
    return values.mean(), se, J - 1


def corrected_ci(values, ratio, level=0.95):
    mean, se, dof = corrected_t(values, ratio)
    if np.isnan(se):
        return mean, np.nan
    return mean, stats.t.ppf(0.5 + level / 2, dof) * se


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def _ratio(g):
    return (g.n_test / (g.n_fit.where(g.augmented == 0, g.n_fit / 4) + g.n_val)).mean()


def _prepare(df):
    df = df.copy()
    df["augmented"] = (df.aug != "none").astype(int)
    return df


def fmt(mean, spread):
    return f"{100 * mean:.1f}" if np.isnan(spread) else f"{100 * mean:.1f} ± {100 * spread:.1f}"


def subject_table(df, metric, use_ci):
    rows = {}
    for method, g in df.groupby("method"):
        row = {}
        for subj in cfg.SUBJECTS:
            s = g[g.subject == subj]
            if s.empty:
                row[subj] = "-"
                continue
            if use_ci:
                row[subj] = fmt(*corrected_ci(s[metric], _ratio(s)))
            else:
                per_repeat = s.groupby("repeat")[metric].mean()
                vals = per_repeat if len(per_repeat) > 1 else s[metric]
                row[subj] = fmt(vals.mean(), vals.std() if len(vals) > 1 else np.nan)
        complete = all(row[s] != "-" for s in cfg.SUBJECTS)
        subj_means = g.groupby("subject")[metric].mean()
        row["Avg"] = fmt(subj_means.mean(), subj_means.std()) if complete else "-"
        rows[method] = row
    return pd.DataFrame(rows).T[cfg.SUBJECTS + ["Avg"]]


def compare(df, a, b, metric):
    keys = ["subject", "repeat", "fold", "n_train"]
    A = df[df.method == a].set_index(keys)
    B = df[df.method == b].set_index(keys)
    common = A.index.intersection(B.index)
    if len(common) < 4:
        raise SystemExit(f"Only {len(common)} paired folds; run both methods on the same subjects/repeats.")
    diff = (B.loc[common, metric] - A.loc[common, metric]).rename("diff").reset_index()
    ratio = _ratio(A.loc[common].reset_index())

    print(f"\n{b}  vs  {a}   ({metric})")
    print("Within-subject: corrected repeated k-fold t-test and Wilcoxon, Holm-adjusted across subjects")
    rows = []
    for subj, g in diff.groupby("subject"):
        mean, se, dof = corrected_t(g["diff"], ratio)
        p_t = 2 * stats.t.sf(abs(mean / se), dof) if se and not np.isnan(se) and se > 0 else np.nan
        p_w = stats.wilcoxon(g["diff"]).pvalue if (g["diff"] != 0).sum() >= 1 else np.nan
        rows.append(dict(subject=subj, folds=len(g), mean_diff=100 * mean, p_corrected_t=p_t, p_wilcoxon=p_w))
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p_corrected_t.fillna(1.0))
    print(out.round(4).to_string(index=False))

    subj_diff = diff.groupby("subject")["diff"].mean()
    improved = int((subj_diff > 0).sum())
    print(f"\nGroup level: mean difference {100 * subj_diff.mean():+.2f} points, "
          f"improved in {improved}/{len(subj_diff)} subjects.")
    if len(subj_diff) >= 5:
        print(f"Wilcoxon on subject means p = {stats.wilcoxon(subj_diff).pvalue:.4g} "
              "(with 5 subjects the smallest possible two-sided p is 0.0625; report descriptively)")


def _clean_name(method):
    return " ".join(p for p in method.split() if not p.startswith("+leak:"))


# Inflation from each reproduced protocol error: leaky method minus its clean counterpart,
# on identical folds and seeds
def audit(df, metric):
    rows = []
    keys = ["subject", "repeat", "fold", "n_train"]
    for method in sorted(m for m in df.method.unique() if "+leak:" in m):
        clean = _clean_name(method)
        if clean not in set(df.method):
            print(f"skipping {method}: clean counterpart '{clean}' has not been run")
            continue
        A = df[df.method == clean].set_index(keys)[metric]
        B = df[df.method == method].set_index(keys)[metric]
        common = A.index.intersection(B.index)
        diff = (B.loc[common] - A.loc[common]).groupby(level="subject").mean()
        row = dict(leaky_method=method, clean=100 * A.loc[common].mean(), leaky=100 * B.loc[common].mean(),
                   inflation=100 * diff.mean(), subjects_inflated=f"{int((diff > 0).sum())}/{len(diff)}")
        row.update({s: round(100 * v, 1) for s, v in diff.items()})
        rows.append(row)
    if rows:
        print(f"\nInflation in {metric} (percentage points), leaky minus clean on identical folds:")
        print(pd.DataFrame(rows).round(2).to_string(index=False))


# Selecting the best of several conditions per subject by its TEST score (the submitted
# Table II practice) versus each pre-specified condition
def bestof(df, methods, metric):
    per = df[df.method.isin(methods)].groupby(["method", "subject", "repeat"])[metric].mean().unstack("method")
    missing = set(methods) - set(per.columns)
    if missing:
        raise SystemExit(f"Not run: {sorted(missing)}")
    oracle = per.max(axis=1).groupby(level="subject").mean()
    print(f"\nPer-subject best-of-{len(methods)} chosen on test data: {100 * oracle.mean():.1f}")
    for m in methods:
        fixed = per[m].groupby(level="subject").mean()
        print(f"  pre-specified {m}: {100 * fixed.mean():.1f}  (inflation {100 * (oracle - fixed).mean():+.1f})")


def curve(df, metric):
    per_repeat = df.groupby(["method", "n_train", "repeat", "subject"])[metric].mean()
    avg = per_repeat.groupby(["method", "n_train", "repeat"]).mean().groupby(["method", "n_train"])
    table = avg.mean().unstack("n_train").mul(100).round(1)
    print(f"\nMean {metric} over subjects vs number of training trials:")
    print(table.to_string())
    print("\nPer subject:")
    print(df.groupby(["method", "subject", "n_train"])[metric].mean().unstack("n_train").mul(100).round(1).to_string())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", default="cv", choices=["cv", "official", "curve"])
    p.add_argument("--tag", default="main")
    p.add_argument("--metric", default="acc", choices=["acc", "bal_acc", "f1_pos", "f1_macro", "auc", "kappa"])
    p.add_argument("--compare", nargs=2, metavar=("BASELINE", "METHOD"))
    p.add_argument("--ci", action="store_true", help="corrected 95%% CI instead of SD over repeats")
    p.add_argument("--latex", action="store_true")
    p.add_argument("--audit", action="store_true", help="inflation from reproduced leaks")
    p.add_argument("--bestof", nargs="+", metavar="METHOD", help="best-of-conditions selection effect")
    args = p.parse_args()

    df = _prepare(load(args))
    if args.audit:
        audit(df, args.metric)
    elif args.bestof:
        bestof(df, args.bestof, args.metric)
    elif args.compare:
        compare(df, *args.compare, args.metric)
    elif args.protocol == "curve":
        curve(df, args.metric)
    else:
        table = subject_table(df, args.metric, args.ci)
        print(table.to_latex() if args.latex else table.to_markdown())


if __name__ == "__main__":
    main()
