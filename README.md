# Evaluation Protocol Effects Exceed Model Differences in Arabic Abstractive Summarization

Code, predictions and per-example scores for:

> M. Bentalb, W. Cherif, O. M. Reda. *Evaluation Protocol Effects Exceed Model Differences
> in Arabic Abstractive Summarization.* ICALP 2026 (Springer CCIS).

The paper fine-tunes mT5-small, AraBART and AraT5v2-base under one protocol, adds zero-shot
AraBART and Jais-6.7B, and shows that the way ROUGE tokenises and normalises Arabic moves the
score more than the choice of model does.

**Every number in the paper's results section can be reproduced from this repository on a
laptop CPU in about five minutes**, without any GPU or model weights. The training
notebooks are included for completeness.

---

## Repository layout

```
data/test_split.csv          test-set order, source and SHA-256 of every article and summary
predictions/finetuned/       generated summaries of the three fine-tuned models (idx, prediction)
predictions/zeroshot/        zero-shot AraBART and Jais-6.7B
scripts/data_io.py           loads the test set + predictions and verifies them against the hashes
scripts/normalisers.py       the tokenisers and the six Arabic normalisers
scripts/score_xlsum_official.py   scoring with the official XL-Sum scorer   (environment B)
scripts/reproduce_analysis.py     every other analysis in the paper           (environment A)
scripts/make_figures.py      Fig. 1 (four panels)
notebooks/                   data preparation, training, generation (Kaggle, one T4 GPU)
outputs/                     the results as produced for the paper, incl. per-example scores
```

## Reproducing the results

### 1. Rebuild the test set

Article and reference texts are not redistributed here. Run
`notebooks/01_data_preparation.ipynb` (it downloads XL-Sum Arabic from Hugging Face and reads
AraSum from the file released by Kahla et al.), then copy the resulting
`canonical_dataset/test.csv` to `data/test.csv`. The pipeline is deterministic (seed 42);
`scripts/data_io.py` checks every row against `data/test_split.csv` and stops if anything differs.

### 2. Environment B: official XL-Sum scorer

The XL-Sum scorer installs under the module name `rouge_score`, the same as Google's
package, so it needs its own environment.

```bash
python -m venv venv-xlsum && source venv-xlsum/bin/activate
pip install -r requirements-xlsum.txt
python -c "import nltk; nltk.download('stopwords')"
python scripts/score_xlsum_official.py      # writes outputs/xlsum_official_per_example.csv.gz
deactivate
```

### 3. Environment A: everything else

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords')"
python scripts/reproduce_analysis.py        # all tables; see outputs/run_log.txt
python scripts/make_figures.py              # Fig. 1 (fig_overview.pdf)
```

### Where each paper result comes from

| Paper | Output file |
|---|---|
| Table 2, main results with 95% CIs; paired differences in Sect. 4.1 | `outputs/main_with_ci.csv`, `outputs/pairwise_ci.csv` |
| Fig. 1a, six scoring protocols (P1–P6) | `outputs/protocols.csv` |
| Fig. 1b, Jais truncation sweep; Jais CIs | `outputs/length_sweep.csv`, `outputs/jais_ci.csv` |
| Fig. 1c, output length vs ROUGE-1; length statistics in Sect. 4.4 | `outputs/per_example_features.csv.gz`, `outputs/length_correlation.csv`, `outputs/length_brackets.csv` |
| Fig. 1d, correlation matrix | `outputs/correlation_matrix.csv` |
| Table 3, normaliser matrix | `outputs/normaliser_matrix.csv` |
| Table 4, named entities | `outputs/entities.csv` |
| Table 5, extractiveness | `outputs/extractiveness.csv`, `outputs/extractiveness_pairwise_ci.csv` |
| Sect. 4.6, comparison with Kahla et al. | `outputs/prior_work_arasum.csv` |
| Per-example scores | `outputs/per_example_scores.csv.gz`, `outputs/xlsum_official_per_example.csv.gz` |

## Scoring protocols

| Name in paper | Implementation |
|---|---|
| `whitespace` | Google `rouge-score` with a tokenizer that splits on whitespace |
| `XL-Sum` | official XL-Sum scorer, `use_stemmer=False, lang="arabic"` (punctuation discarded) |
| `punct-aware` | Google `rouge-score`; lowercase; punctuation split off and **kept** as tokens |
| `XL-Sum + Snowball` | official XL-Sum scorer, `use_stemmer=True, lang="arabic"`: the Kahla et al. protocol |
| `punct-aware + Snowball` | our earlier re-implementation of the line above; **not equivalent** (see below) |
| `punct-aware + Qalsadi` | primary protocol: punct-aware tokens replaced by their Qalsadi lemma |

ROUGE is reported as F-measure (F1) throughout; precision and recall are in the per-example files.

## Failure modes we hit (and how the code guards against them)

1. **Every Arabic ROUGE score is exactly 0.0.** Google `rouge-score`'s default tokenizer keeps
   only `[a-z0-9]` and deletes Arabic. Passing a plain function as `tokenizer=` is silently
   ignored: it must be an object with a `.tokenize()` method. `reproduce_analysis.py` runs a
   self-test on an Arabic sentence pair before scoring.
2. **Tokenizer fails to load** (`TypeError: 'dict' object cannot be converted to 'Sequence'`
   for AraBART, `KeyError: 0` for AraT5) under `transformers` 5.x. Pin `transformers==4.47.1`
   and restart the kernel.
3. **Low training loss but multilingual noise at generation.** Calling `tie_weights()` after
   training overwrites a T5 model's separately trained `lm_head` with the input embedding, and a
   config with `tie_word_embeddings=True` re-ties it on load. Load with
   `tie_word_embeddings=False` and check that `lm_head` differs from `shared`
   (`notebooks/05_compare_all_models.ipynb`, `load_checked`).
4. **A "re-implemented" protocol is about 3 ROUGE-1 points too high.** Scoring punctuation
   tokens and stemming very short words (NLTK's Arabic Snowball stemmer returns an empty string
   for some, e.g. `في`, and every empty string then matches every other) inflates scores
   relative to the official XL-Sum scorer. Use the official scorer when comparing with
   published numbers.

Two smaller reproducibility notes:
- NLTK's `ArabicStemmer` keeps one flag (`suffixes_verb_step1_success`) across calls, so caching
  its output changes scores slightly. The scripts call it per token, as the original runs did.
- Tashaphyne's root extraction picks among candidate roots in set order, which depends on
  Python's hash seed. Both scripts pin `PYTHONHASHSEED=0` (they re-launch themselves if needed).

## Training (optional)

The notebooks ran on Kaggle with one NVIDIA T4 (Python 3.12, PyTorch 2.10.0+cu128,
`transformers` 4.47.1; see `requirements-training.txt`). Kaggle input paths in the notebooks
point at the authors' private datasets/models and must be changed.

| Notebook | Purpose |
|---|---|
| `01_data_preparation` | builds the 77,437-pair corpus and the 75,937 / 500 / 1,000 split |
| `02_finetune_seq2seq` | fine-tunes AraBART, AraT5v2-base or mT5-small (set `MODEL_KEY`) |
| `03_resume_mt5_training` | resumes mT5-small from a Trainer checkpoint |
| `04_jais_zero_shot_and_qlora` | Jais-6.7B; the paper's zero-shot run used `TEST_MODE=True, TEST_MODE_N=1000` (training skipped, full test set) |
| `05_compare_all_models` | generates with every model and scores all protocols |

Hyperparameters: effective batch 16; Adafactor at 5e-4 (T5 family, fp32) or AdamW at 5e-5
(AraBART, fp16); linear decay, no warm-up; at most 8 epochs, early stopping on validation loss
with patience 2, best checkpoint restored; input/target 512/100 tokens; beam 4, maximum
length 150; seed 42.

## Licence and data

Code: MIT (see `LICENSE`). The predictions are model outputs derived from AraSum (Deutsche
Welle) and XL-Sum (BBC, CC BY-NC-SA 4.0) articles and are provided for research only, under
the terms of those datasets.

## Citation

```bibtex
@inproceedings{bentalb2026protocol,
  title     = {Evaluation Protocol Effects Exceed Model Differences in Arabic Abstractive
               Summarization},
  author    = {Bentalb, Mohamed and Cherif, Walid and Reda, Oussama Mohamed},
  booktitle = {Arabic Language Processing (ICALP 2026)},
  series    = {Communications in Computer and Information Science},
  publisher = {Springer},
  year      = {2026}
}
```
