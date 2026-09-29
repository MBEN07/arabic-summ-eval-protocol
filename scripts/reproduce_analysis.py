"""
Reproduce every score-level result in the paper from the saved prediction files.

No GPU and no model weights are needed: all analyses operate on the generated
summaries in `predictions/`, which were produced by the notebooks in
`notebooks/` (see README).

    python scripts/reproduce_analysis.py --pred-dir predictions --test-csv data/test.csv --out-dir outputs

Outputs (CSV + one JSON summary) are written to --out-dir:
    protocols.csv           ROUGE-1/2/L for every system under 8 scoring protocols
    main_with_ci.csv        primary-protocol scores with 95% bootstrap CIs
    pairwise_ci.csv         paired bootstrap CIs on model differences
    normaliser_matrix.csv   per-normaliser diagnostic behaviour (error matrix)
    entities.csv            what each normaliser does to 20 corpus named entities
    extractiveness.csv      coverage / density / novel n-grams vs. the source article
    length_brackets.csv     ROUGE-1 by output-length bracket
    length_correlation.csv  Spearman correlations and regression R^2 for length
    length_sweep.csv        zero-shot Jais ROUGE-1 P/R/F1 when truncated to N words (Fig. 2)
    jais_ci.csv             bootstrap CIs for zero-shot Jais (full and truncated)
    prior_work_arasum.csv   AraSum slice vs. Kahla et al., official and re-implemented protocol
    extractiveness_pairwise_ci.csv  paired bootstrap CIs on extractiveness differences
    per_example_scores.csv.gz  per-example P/R/F1 under the four rouge-score protocols
    summary.json            everything above plus library versions
"""
import argparse
import collections
import importlib.metadata as md
import json
import os
import re
import sys

# Tashaphyne's root extraction picks from a set of candidates, so its output depends on
# Python's string-hash seed. Pin it (re-executing once if needed) so runs are identical.
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np
import pandas as pd
from scipy import stats
from rouge_score import rouge_scorer

import nltk

SEED = 42
N_BOOT = 2000
KEYS = ["rouge1", "rouge2", "rougeL"]

FINETUNED = ["mT5-small", "AraBART", "AraT5v2-base"]

# --------------------------------------------------------------------------
# Tokenisation / normalisation protocols
# --------------------------------------------------------------------------
from normalisers import basic_tokenize, NORMALISERS
from data_io import load, SYSTEMS


SPLITTERS = {
    "whitespace": str.split,
    "punct-kept": basic_tokenize,  # punctuation marks become tokens (and can match)
}


class Protocol:
    """A custom tokeniser for Google's rouge-score. It must be an object with
    .tokenize(); a bare callable passed as tokenizer= is silently ignored.

    empty: what to do when a normaliser returns '' --
        'keep'    leave '' in the token list (every '' then matches every other '')
        'restore' fall back to the original token
    """

    def __init__(self, name, split, norm="none", empty="restore"):
        self.name, self.split, self.norm, self.empty = name, SPLITTERS[split], NORMALISERS[norm], empty

    def tokenize(self, text):
        out = []
        for t in self.split(text):
            v = self.norm(t)
            if v == "" and self.empty == "restore":
                v = t
            out.append(v)
        return out


# The custom-tokeniser protocols of the submitted paper (Google rouge-score 0.1.2).
PROTOCOLS = [
    Protocol("whitespace", "whitespace"),
    Protocol("punct-aware (punct. kept)", "punct-kept"),
    Protocol("punct-aware + Snowball", "punct-kept", "Snowball stemmer", "keep"),
    Protocol("punct-aware + Qalsadi (primary)", "punct-kept", "Qalsadi lemmatiser", "restore"),
]
PRIMARY = "punct-aware + Qalsadi (primary)"
# Produced by score_xlsum_official.py with the official XL-Sum scorer.
OFFICIAL = ["XL-Sum scorer, unstemmed", "XL-Sum scorer, Snowball (Kahla et al.)"]
SWEEP = {n: f"XL-Sum scorer + {n}" for n in NORMALISERS if n != "none"}
TABLE_ORDER = ["whitespace", "XL-Sum scorer, unstemmed", "punct-aware (punct. kept)",
               "XL-Sum scorer, Snowball (Kahla et al.)", "punct-aware + Snowball",
               "punct-aware + Qalsadi (primary)"] + list(SWEEP.values())


def per_example_rouge(preds, refs, proto):
    sc = rouge_scorer.RougeScorer(KEYS, use_stemmer=False, tokenizer=proto)
    out = {f"{k}_{m}": np.zeros(len(preds)) for k in KEYS for m in "prf"}
    for i, (p, r) in enumerate(zip(preds, refs)):
        s = sc.score(r, p)
        for k in KEYS:
            out[f"{k}_p"][i] = s[k].precision * 100
            out[f"{k}_r"][i] = s[k].recall * 100
            out[f"{k}_f"][i] = s[k].fmeasure * 100
    return out


def self_test():
    p = "أعلن متحدث باسم فرق الإطفاء أنّ الحريق الضخم اندلع في كاتدرائية نوتردام"
    r = "أعلنت فرق الإطفاء أنّ الحريق الضخم الذي اندلع في كاتدرائية نوتردام في باريس"
    for proto in PROTOCOLS:
        v = per_example_rouge([p], [r], proto)["rouge1_f"][0]
        assert v > 0, f"{proto.name} scored zero on overlapping Arabic -- tokenizer broken"


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------
def boot_idx(n, rng):
    return rng.integers(0, n, size=(N_BOOT, n))


def ci_mean(x, idx):
    means = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# --------------------------------------------------------------------------
# Extractiveness (Grusky et al., 2018)
# --------------------------------------------------------------------------
def extractive_fragments(article, summary):
    """Greedy longest shared token sequences, as in the Newsroom paper."""
    frags, i, A, S = [], 0, len(article), len(summary)
    pos = collections.defaultdict(list)
    for j, t in enumerate(article):
        pos[t].append(j)
    while i < S:
        best = 0
        for j in pos.get(summary[i], ()):
            k = 0
            while i + k < S and j + k < A and summary[i + k] == article[j + k]:
                k += 1
            best = max(best, k)
        if best:
            frags.append(best)
            i += best
        else:
            i += 1
    return frags


def ngrams(toks, n):
    return set(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))


def novel_rate(article, summary, n):
    s = ngrams(summary, n)
    return np.nan if not s else len(s - ngrams(article, n)) / len(s)


def extractiveness(articles, summaries, per_example=None):
    cov, den, nov = [], [], {1: [], 2: [], 3: []}
    for a, s in zip(articles, summaries):
        at, st = basic_tokenize(a), basic_tokenize(s)
        if not st:
            cov.append(np.nan); den.append(np.nan)
            for n in nov:
                nov[n].append(np.nan)
            continue
        f = extractive_fragments(at, st)
        cov.append(sum(f) / len(st))
        den.append(sum(x * x for x in f) / len(st))
        for n in nov:
            nov[n].append(novel_rate(at, st, n))
    if per_example is not None:
        per_example.update({"density": np.array(den), "novel_3g": np.array(nov[3]) * 100,
                            "coverage": np.array(cov)})
    return {
        "coverage": float(np.nanmean(cov)),
        "density": float(np.nanmean(den)),
        "novel_1g_%": float(np.nanmean(nov[1]) * 100),
        "novel_2g_%": float(np.nanmean(nov[2]) * 100),
        "novel_3g_%": float(np.nanmean(nov[3]) * 100),
    }


# --------------------------------------------------------------------------
# Normaliser diagnostics
# --------------------------------------------------------------------------
VARIANT_PAIRS = [  # pairs a correct normaliser should unify
    ("الرئيس", "رئيس", "definite article"),
    ("الكتاب", "كتاب", "definite article"),
    ("المعارضة", "معارضة", "definite article"),
    ("للنزاع", "النزاع", "preposition clitic"),
    ("بالمدينة", "المدينة", "preposition clitic"),
    ("والحكومة", "الحكومة", "conjunction clitic"),
    ("أعلن", "أعلنت", "verb inflection"),
    ("قال", "قالت", "verb inflection"),
    ("كتب", "يكتب", "verb inflection"),
    ("درس", "يدرس", "verb inflection"),
    ("مدينة", "مدن", "broken plural"),
    ("كتاب", "كتب", "broken plural"),
    ("رجل", "رجال", "broken plural"),
]
# Pairs a correct normaliser must keep APART: distinct lexemes that share a root.
OVERMERGE_PAIRS = [
    ("كتاب", "مكتب"),     # book / office
    ("كاتب", "مكتبة"),    # writer / library
    ("عالم", "علم"),      # world|scholar / science|flag
    ("معارضة", "عرض"),    # opposition / offer
    ("حكومة", "حكم"),     # government / verdict
    ("مدرسة", "دراسة"),   # school / study
    ("عمل", "معمل"),      # work / factory, laboratory
    ("قتل", "مقاتل"),     # killing / fighter
    ("سفير", "سفر"),      # ambassador / travel
    ("مجلس", "جلسة"),     # council / session
]
ENTITIES = ["باريس", "بغداد", "لبنان", "واشنطن", "بريطانيا", "برلين", "فرنسا", "بيروت",
            "دمشق", "القاهرة", "لندن", "موسكو", "طهران", "الرياض", "اسطنبول", "نيويورك",
            "تونس", "الجزائر", "المغرب", "اليمن"]


def normaliser_matrix(reference_vocab):
    rows, ent_rows = [], []
    tot = sum(reference_vocab.values())
    for name, f in NORMALISERS.items():
        if name == "none":
            continue
        by = collections.defaultdict(lambda: [0, 0])
        for a, b, ph in VARIANT_PAIRS:
            by[ph][0] += f(a) == f(b)
            by[ph][1] += 1
        over = sum(f(a) == f(b) for a, b in OVERMERGE_PAIRS)
        ents = [e for e in ENTITIES if reference_vocab.get(e, 0) > 0]
        kept = sum(f(e) == e for e in ents)
        altered = sum(c for w, c in reference_vocab.items() if f(w) != w) / tot * 100
        row = {"normaliser": name}
        for ph in ["definite article", "preposition clitic", "conjunction clitic",
                   "verb inflection", "broken plural"]:
            row[ph] = f"{by[ph][0]}/{by[ph][1]}"
        row["distinct lexemes merged"] = f"{over}/{len(OVERMERGE_PAIRS)}"
        row["entities intact"] = f"{kept}/{len(ents)}"
        row["reference tokens altered %"] = round(altered, 1)
        rows.append(row)
        for e in ents:
            ent_rows.append({"entity": e, "freq": reference_vocab[e], "normaliser": name,
                             "form": f(e), "intact": f(e) == e})
    return pd.DataFrame(rows), pd.DataFrame(ent_rows)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default="predictions")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--test-csv", default="data/test.csv")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)
    self_test()

    data = load(args.pred_dir, args.test_csv)
    first = data[FINETUNED[0]]
    for k, d in data.items():
        assert len(d) == 1000 and (d.reference == first.reference).all(), f"{k}: test set mismatch"
    refs, arts, srcs = first.reference.tolist(), first.article.tolist(), first.source.tolist()
    n = len(refs)
    idx = boot_idx(n, rng)
    summary = {"versions": {p: md.version(p) for p in
                            ["rouge_score", "nltk", "qalsadi", "tashaphyne", "pyarabic",
                             "numpy", "pandas", "scipy"]},
               "python": sys.version.split()[0], "seed": SEED, "n_bootstrap": N_BOOT}

    # ---- 1. all systems x all protocols -------------------------------------
    scores, rows = {}, []
    for proto in PROTOCOLS:
        for k, d in data.items():
            scores[(k, proto.name)] = per_example_rouge(d.prediction.tolist(), refs, proto)
        print(f"scored: {proto.name}")
    pe = []
    for proto in PROTOCOLS:
        for k in data:
            sc = scores[(k, proto.name)]
            pe.append(pd.DataFrame({"protocol": proto.name, "system": k, "i": np.arange(n),
                                    **{c: np.round(v, 4) for c, v in sc.items()}}))
    pd.concat(pe).to_csv(os.path.join(args.out_dir, "per_example_scores.csv.gz"), index=False)
    off_path = os.path.join(args.out_dir, "xlsum_official_per_example.csv.gz")
    assert os.path.exists(off_path), (
        f"{off_path} not found -- run scripts/score_xlsum_official.py first (environment B, see README)")
    off = pd.read_csv(off_path)
    for (pname, k), g in off.groupby(["protocol", "system"]):
        g = g.sort_values("i")
        scores[(k, pname)] = {c: g[c].to_numpy() for c in g.columns if c.startswith("rouge")}
    for (k, pname), s in scores.items():
        rows.append({"system": k, "protocol": pname,
                     **{f"{m.upper()}": round(s[f"{m}_f"].mean(), 2) for m in KEYS}})
    prot = pd.DataFrame(rows)
    prot.to_csv(f"{args.out_dir}/protocols.csv", index=False)
    wide = prot.pivot(index="protocol", columns="system", values="ROUGE1").loc[TABLE_ORDER]
    print("\nROUGE-1 F1 by protocol\n", wide.round(2).to_string())
    summary["protocols_rouge1"] = wide.round(2).to_dict()

    # size of the empty-token artefact in the legacy "punct-aware + Snowball" protocol
    art = {}
    fixed = Protocol("punct-aware + Snowball, empties restored", "punct-kept", "Snowball stemmer", "restore")
    for k in FINETUNED:
        v = per_example_rouge(data[k].prediction.tolist(), refs, fixed)["rouge1_f"].mean()
        art[k] = {"with_empty_tokens": round(scores[(k, "punct-aware + Snowball")]["rouge1_f"].mean(), 2),
                  "empties_restored": round(v, 2)}
    print("\nEmpty-token artefact\n", art)
    summary["empty_token_artefact"] = art

    # prior-work comparison on the AraSum slice, under Kahla et al.'s own scorer
    kahla = {"rouge1": 33.84, "rouge2": 16.05, "rougeL": 26.53}  # mBART-50-rus, their best
    ara = np.array([x == "arasum" for x in srcs])
    prior = []
    for proto_name in ["XL-Sum scorer, Snowball (Kahla et al.)", "punct-aware + Snowball"]:
        for k in FINETUNED:
            s_ = scores[(k, proto_name)]
            row = {"protocol": proto_name, "system": k, "n": int(ara.sum())}
            for m in KEYS:
                row[m] = round(s_[f"{m}_f"][ara].mean(), 2)
                row[f"{m}_vs_kahla"] = round(row[m] - kahla[m], 2)
            prior.append(row)
    prior_df = pd.DataFrame(prior); prior_df.to_csv(f"{args.out_dir}/prior_work_arasum.csv", index=False)
    print("\nAraSum slice vs. Kahla et al. best (33.84/16.05/26.53)\n", prior_df.to_string())
    summary["prior_work_arasum"] = prior

    # ---- 2. primary-protocol table with CIs ---------------------------------
    main_rows = []
    for k in data:
        s = scores[(k, PRIMARY)]
        r = {"system": k}
        for m in KEYS:
            mean, lo, hi = ci_mean(s[f"{m}_f"], idx)
            r[m] = round(mean, 2); r[f"{m}_lo"] = round(lo, 2); r[f"{m}_hi"] = round(hi, 2)
        r["R1_precision"] = round(s["rouge1_p"].mean(), 2)
        r["R1_recall"] = round(s["rouge1_r"].mean(), 2)
        r["words"] = round(np.mean([len(p.split()) for p in data[k].prediction]), 1)
        main_rows.append(r)
    main_df = pd.DataFrame(main_rows)
    main_df.to_csv(f"{args.out_dir}/main_with_ci.csv", index=False)
    print("\nPrimary protocol with 95% CI\n", main_df.to_string())

    pw = []
    for a, b in [("mT5-small", "AraBART"), ("mT5-small", "AraT5v2-base"), ("AraBART", "AraT5v2-base"),
                 ("mT5-small", "Jais-6.7B (zero-shot)")]:
        for m in KEYS:
            dlt = scores[(a, PRIMARY)][f"{m}_f"] - scores[(b, PRIMARY)][f"{m}_f"]
            mean, lo, hi = ci_mean(dlt, idx)
            pw.append({"a": a, "b": b, "metric": m, "diff": round(mean, 2), "lo": round(lo, 2),
                       "hi": round(hi, 2), "distinguishable": not (lo <= 0 <= hi)})
    pd.DataFrame(pw).to_csv(f"{args.out_dir}/pairwise_ci.csv", index=False)

    # protocol spread per system (min..max over all 8 protocols) vs max model gap
    spread = {k: (round(wide[k].min(), 2), round(wide[k].max(), 2)) for k in data}
    summary["protocol_spread_rouge1"] = spread
    ft_r1 = main_df.set_index("system").loc[FINETUNED, "rouge1"]
    summary["max_model_gap_rouge1"] = round(ft_r1.max() - ft_r1.min(), 2)

    # ---- 3. normaliser diagnostics ------------------------------------------
    vocab = collections.Counter()
    for r in refs:
        vocab.update(w for w in basic_tokenize(r) if len(w) > 2 and re.match(r"^[\u0600-\u06FF]+$", w))
    nm, ents = normaliser_matrix(vocab)
    for k in FINETUNED:
        nm[f"ROUGE-1 {k}"] = [round(scores[(k, SWEEP[x])]["rouge1_f"].mean(), 2)
                               for x in nm.normaliser]
    base = {k: scores[(k, "XL-Sum scorer, unstemmed")]["rouge1_f"].mean() for k in FINETUNED}
    nm["gain over unnormalised (mT5)"] = [round(scores[("mT5-small", SWEEP[x])]["rouge1_f"].mean()
                                                - base["mT5-small"], 2) for x in nm.normaliser]
    nm.to_csv(f"{args.out_dir}/normaliser_matrix.csv", index=False)
    ents.to_csv(f"{args.out_dir}/entities.csv", index=False)
    print("\nNormaliser error matrix\n", nm.to_string())
    summary["reference_vocab"] = {"tokens": sum(vocab.values()), "types": len(vocab)}

    # ---- 4. extractiveness ---------------------------------------------------
    ex_rows, ex_pe = [{"system": "Reference (gold)", **extractiveness(arts, refs)}], {}
    for k, d in data.items():
        ex_pe[k] = {}
        ex_rows.append({"system": k, **extractiveness(arts, d.prediction.tolist(), ex_pe[k])})
    ex = pd.DataFrame(ex_rows).round(3)
    ex_pw = []
    for a, b in [("AraBART", "AraT5v2-base"), ("mT5-small", "AraBART"), ("mT5-small", "AraT5v2-base")]:
        for m in ["coverage", "density", "novel_3g"]:
            dlt = ex_pe[a][m] - ex_pe[b][m]
            ok = ~np.isnan(dlt)
            mean, lo, hi = ci_mean(dlt[ok], boot_idx(int(ok.sum()), rng))
            ex_pw.append({"a": a, "b": b, "measure": m, "diff": round(mean, 3), "lo": round(lo, 3),
                          "hi": round(hi, 3), "distinguishable": not (lo <= 0 <= hi)})
    pd.DataFrame(ex_pw).to_csv(f"{args.out_dir}/extractiveness_pairwise_ci.csv", index=False)
    print("\nExtractiveness differences (paired bootstrap)\n", pd.DataFrame(ex_pw).to_string())
    summary["extractiveness_pairwise"] = ex_pw
    ex.to_csv(f"{args.out_dir}/extractiveness.csv", index=False)
    print("\nExtractiveness (vs. source article)\n", ex.to_string())

    # ---- 5. length -----------------------------------------------------------
    ref_len = np.array([len(r.split()) for r in refs])
    br_rows, cor_rows = [], []
    bins = [(0, 19), (20, 29), (30, 39), (40, 10**6)]
    for k, d in data.items():
        L = np.array([len(p.split()) for p in d.prediction])
        s = scores[(k, PRIMARY)]
        f1, p, r = s["rouge1_f"], s["rouge1_p"], s["rouge1_r"]
        for lo, hi in bins:
            m = (L >= lo) & (L <= hi)
            br_rows.append({"system": k, "bracket": f"{lo}-{hi}" if hi < 10**6 else f">={lo}",
                            "n": int(m.sum()),
                            "R1_F1": round(f1[m].mean(), 2) if m.any() else np.nan,
                            "R1_P": round(p[m].mean(), 2) if m.any() else np.nan,
                            "R1_R": round(r[m].mean(), 2) if m.any() else np.nan})
        ratio = np.log((L + 1) / (ref_len + 1))
        rho_len = stats.spearmanr(L, f1)
        rho_dev = stats.spearmanr(np.abs(ratio), f1)
        rho_p = stats.spearmanr(L, p)
        rho_r = stats.spearmanr(L, r)
        X = np.column_stack([np.ones(n), ratio, ratio ** 2])
        beta, *_ = np.linalg.lstsq(X, f1, rcond=None)
        r2 = 1 - ((f1 - X @ beta) ** 2).sum() / ((f1 - f1.mean()) ** 2).sum()
        cor_rows.append({"system": k, "mean_words": round(L.mean(), 1),
                         "rho(len,F1)": round(rho_len.statistic, 3), "p_len": rho_len.pvalue,
                         "rho(|log ratio|,F1)": round(rho_dev.statistic, 3), "p_dev": rho_dev.pvalue,
                         "rho(len,P)": round(rho_p.statistic, 3), "rho(len,R)": round(rho_r.statistic, 3),
                         "R2_length_model": round(r2, 3)})
    # pooled across all five systems: how much of the SYSTEM-level variance is length?
    br = pd.DataFrame(br_rows); br.to_csv(f"{args.out_dir}/length_brackets.csv", index=False)
    cor = pd.DataFrame(cor_rows); cor.to_csv(f"{args.out_dir}/length_correlation.csv", index=False)
    print("\nLength brackets\n", br.to_string())
    print("\nLength correlation\n", cor.to_string())

    # ---- 6. Jais: CI on full output and 30-word truncation -------------------
    jais = data["Jais-6.7B (zero-shot)"].prediction.tolist()
    prim = next(p for p in PROTOCOLS if p.name == PRIMARY)
    jrows = []
    for label, preds in [("full output", jais),
                         ("first 30 words", [" ".join(p.split()[:30]) for p in jais])]:
        s = per_example_rouge(preds, refs, prim)
        row = {"variant": label, "words": round(np.mean([len(p.split()) for p in preds]), 1)}
        for m in KEYS:
            mean, lo, hi = ci_mean(s[f"{m}_f"], idx)
            row.update({m: round(mean, 2), f"{m}_lo": round(lo, 2), f"{m}_hi": round(hi, 2)})
        dlt = scores[("mT5-small", PRIMARY)]["rouge1_f"] - s["rouge1_f"]
        mean, lo, hi = ci_mean(dlt, idx)
        row.update({"mT5_minus_this_R1": round(mean, 2), "lo": round(lo, 2), "hi": round(hi, 2)})
        jrows.append(row)
    jdf = pd.DataFrame(jrows); jdf.to_csv(f"{args.out_dir}/jais_ci.csv", index=False)
    # truncation sweep behind Fig. 2
    sweep = []
    for cap in [5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 85, 100, 150]:
        preds = [" ".join(p.split()[:cap]) for p in jais]
        s = per_example_rouge(preds, refs, prim)
        sweep.append({"cap": cap, "words": np.mean([len(p.split()) for p in preds]),
                      "P": s["rouge1_p"].mean(), "R": s["rouge1_r"].mean(), "F1": s["rouge1_f"].mean()})
    pd.DataFrame(sweep).to_csv(f"{args.out_dir}/length_sweep.csv", index=False)

    print("\nJais with CIs\n", jdf.to_string())

    summary.update({"main": main_rows, "pairwise": pw, "normaliser_matrix": nm.to_dict("records"),
                    "extractiveness": ex.to_dict("records"), "length_correlation": cor.to_dict("records"),
                    "length_brackets": br.to_dict("records"), "jais": jrows})
    with open(f"{args.out_dir}/summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1, default=float)
    print(f"\nWrote results to {args.out_dir}/")


if __name__ == "__main__":
    main()
