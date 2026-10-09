# Build every paper table and figure from results/folds.csv in one go.
#   python report.py              -> results/report/
import subprocess
import sys

import matplotlib
import pandas as pd

import config as cfg

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = cfg.RESULTS_DIR / "report"


def summarize(*args):
    run = subprocess.run([sys.executable, "summarize.py", *args], capture_output=True, text=True)
    return run.stdout + run.stderr


# Accuracy vs number of calibration trials, mean over subjects, error bars = SD over repeats
def curve_figure(df):
    d = df[df.protocol == "curve"]
    if d.empty:
        return
    per_repeat = d.groupby(["method", "n_train", "repeat", "subject"]).acc.mean()
    avg = per_repeat.groupby(["method", "n_train", "repeat"]).mean().groupby(["method", "n_train"])
    mean, sd = avg.mean().unstack("method"), avg.std().unstack("method")
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for m in mean.columns:
        ax.errorbar(mean.index, 100 * mean[m], yerr=100 * sd[m].fillna(0), marker="o", ms=3,
                    capsize=2, lw=1, label=m)
    ax.set_xscale("log")
    ax.set_xticks(mean.index)
    ax.set_xticklabels([str(int(n)) for n in mean.index])
    ax.set_xlabel("Calibration trials")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"learning_curve.{ext}", dpi=300)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(cfg.RESULTS_DIR / "folds.csv")
    df = df[df.tag == "main"]
    for protocol in sorted(df.protocol.unique()):
        base = ["--protocol", protocol]
        if protocol == "curve":
            (OUT / "curve.txt").write_text(summarize(*base))
            continue
        for metric in ("acc", "auc", "kappa"):
            (OUT / f"{protocol}_{metric}.md").write_text(summarize(*base, "--metric", metric))
            (OUT / f"{protocol}_{metric}.tex").write_text(summarize(*base, "--metric", metric, "--latex"))
        if protocol == "cv":
            (OUT / "cv_ci.md").write_text(summarize(*base, "--ci"))
            (OUT / "audit.txt").write_text(summarize(*base, "--audit"))
    curve_figure(df)
    print(f"Wrote {len(list(OUT.iterdir()))} files to {OUT}")


if __name__ == "__main__":
    main()
