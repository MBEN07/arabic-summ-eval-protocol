"""Draw Fig. 1 (bootstrap forest plot) and Fig. 2 (Jais truncation sweep) from the
CSV files written by reproduce_analysis.py.

    python scripts/make_figures.py --out-dir outputs
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

NAMES = {"mT5-small": "mT5", "AraBART": "AraBART", "AraT5v2-base": "AraT5v2"}
METRIC = {"rouge1": "ROUGE-1", "rouge2": "ROUGE-2", "rougeL": "ROUGE-L"}


def forest(out):
    d = pd.read_csv(os.path.join(out, "pairwise_ci.csv"))
    d = d[d.b != "Jais-6.7B (zero-shot)"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for i, r in d.iterrows():
        y = len(d) - 1 - i
        c = "#1f3b73" if r.distinguishable else "#c0392b"
        ax.errorbar(r["diff"], y, xerr=[[r["diff"] - r.lo], [r.hi - r["diff"]]], fmt="o", color=c, capsize=3)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([f"{NAMES[r.a]} − {NAMES[r.b]}  {METRIC[r.metric]}" for _, r in d.iloc[::-1].iterrows()])
    ax.axvline(0, color="grey", ls="--", lw=1)
    ax.set_xlabel("Difference in ROUGE (points)")
    ax.plot([], [], "o", color="#1f3b73", label="CI excludes zero")
    ax.plot([], [], "o", color="#c0392b", label="CI includes zero")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_bootstrap.pdf")); plt.close(fig)


def sweep(out):
    d = pd.read_csv(os.path.join(out, "length_sweep.csv"))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(d.words, d.P, "o-", label="ROUGE-1 precision")
    ax.plot(d.words, d.R, "o-", label="ROUGE-1 recall")
    ax.plot(d.words, d.F1, "o-", label="ROUGE-1 F1")
    ax.axvline(30.3, color="grey", ls=":", lw=1)
    ax.text(31, 3, "reference length\n30.3 words", fontsize=8, color="grey")
    ax.set_xlabel("Mean generated summary length (words)"); ax.set_ylabel("ROUGE-1 (%)")
    ax.set_ylim(0, None); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_length.pdf")); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out-dir", default="outputs")
    a = ap.parse_args(); forest(a.out_dir); sweep(a.out_dir)
    print("wrote fig_bootstrap.pdf and fig_length.pdf")
