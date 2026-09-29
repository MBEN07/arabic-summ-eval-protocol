"""
Score every system with the OFFICIAL XL-Sum multilingual ROUGE scorer
(github.com/csebuetnlp/xl-sum, folder multilingual_rouge_scoring), which is the
scorer used by Kahla et al. for AraSum.

That package installs under the module name `rouge_score` and therefore cannot
share an environment with Google's rouge-score. Run this script in its own
environment (see README, "Environment B"); it writes per-example scores that
`reproduce_analysis.py` then picks up.

    python scripts/score_xlsum_official.py --pred-dir predictions --test-csv data/test.csv --out-dir outputs

Protocols
    XL-Sum scorer, unstemmed                 use_stemmer=False
    XL-Sum scorer, Snowball (Kahla et al.)   use_stemmer=True, lang="arabic"
    XL-Sum scorer + <normaliser>             each normaliser in normalisers.py passed
                                             as callable_stemmer (the scorer applies it
                                             to tokens longer than 3 characters)
"""
import argparse
import os
import sys

# Tashaphyne's root extraction picks from a set of candidates, so its output depends on
# Python's string-hash seed. Pin it (re-executing once if needed) so runs are identical.
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np
import pandas as pd
from rouge_score import rouge_scorer  # XL-Sum's fork

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalisers import NORMALISERS  # noqa: E402
from data_io import load  # noqa: E402

KEYS = ["rouge1", "rouge2", "rougeL"]
def scorers():
    yield "XL-Sum scorer, unstemmed", rouge_scorer.RougeScorer(KEYS, use_stemmer=False, lang="arabic")
    yield "XL-Sum scorer, Snowball (Kahla et al.)", rouge_scorer.RougeScorer(KEYS, use_stemmer=True, lang="arabic")
    for name, fn in NORMALISERS.items():
        if name == "none":
            continue
        # a normaliser returning '' would be dropped by the scorer; keep the token instead
        stem = (lambda f: (lambda w: f(w) or w))(fn)
        yield f"XL-Sum scorer + {name}", rouge_scorer.RougeScorer(
            KEYS, use_stemmer=True, lang="arabic", callable_stemmer=stem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default="predictions")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--test-csv", default="data/test.csv")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    data = load(args.pred_dir, args.test_csv)
    rows = []
    for pname, sc in scorers():
        for k, d in data.items():
            for i, (p, r) in enumerate(zip(d.prediction, d.reference)):
                s = sc.score(r, p)
                rows.append({"protocol": pname, "system": k, "i": i,
                             **{f"{m}_{x}": getattr(s[m], a) * 100 for m in KEYS
                                for x, a in [("p", "precision"), ("r", "recall"), ("f", "fmeasure")]}})
        print(pname, {k: round(np.mean([r["rouge1_f"] for r in rows if r["protocol"] == pname
                                         and r["system"] == k]), 2) for k in data}, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "xlsum_official_per_example.csv.gz"), index=False)


if __name__ == "__main__":
    main()
