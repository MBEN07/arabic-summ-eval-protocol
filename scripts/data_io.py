"""Load the test set and every system's predictions, and check they line up.

The repository does not redistribute article or reference text. `data/test.csv`
(columns: article, summary, source) is rebuilt from AraSum and XL-Sum Arabic by
`notebooks/01_data_preparation.ipynb`; `data/test_split.csv` holds the SHA-256 of
every test article and summary, so a rebuilt file is verified row by row.
"""
import hashlib
import os

import pandas as pd

SYSTEMS = {  # display name -> prediction file under --pred-dir
    "mT5-small": "finetuned/mT5-small_predictions.csv",
    "AraBART": "finetuned/AraBART_predictions.csv",
    "AraT5v2-base": "finetuned/AraT5v2-base_predictions.csv",
    "AraBART (zero-shot)": "zeroshot/AraBART_predictions.csv",
    "Jais-6.7B (zero-shot)": "zeroshot/jais-6p7b-chat_predictions.csv",
}


def _sha(t):
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def load(pred_dir="predictions", test_csv="data/test.csv", split_csv="data/test_split.csv"):
    """Return {system: DataFrame[article, reference, prediction, source]} in test order."""
    if not os.path.exists(test_csv):
        raise FileNotFoundError(
            f"{test_csv} not found. Rebuild it with notebooks/01_data_preparation.ipynb "
            "(it writes canonical_dataset/test.csv) and copy it to data/test.csv.")
    test = pd.read_csv(test_csv).fillna("")
    split = pd.read_csv(split_csv)
    assert len(test) == len(split), f"test set has {len(test)} rows, expected {len(split)}"
    bad = (test.article.map(_sha) != split.article_sha256) | (test.summary.map(_sha) != split.summary_sha256)
    assert not bad.any(), (f"{int(bad.sum())} rows of {test_csv} differ from the published split "
                           "(first: row {}). Re-run notebook 01 unchanged.".format(int(bad.idxmax())))
    out = {}
    for name, rel in SYSTEMS.items():
        p = pd.read_csv(os.path.join(pred_dir, rel)).fillna("").sort_values("idx")
        assert list(p.idx) == list(range(len(test))), f"{rel}: idx column does not match the test set"
        out[name] = pd.DataFrame({"article": test.article.values, "reference": test.summary.values,
                                  "prediction": p.prediction.values, "source": test.source.values})
    return out
