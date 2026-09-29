"""Arabic tokenisers and normalisers shared by both scoring scripts.

Kept free of any `rouge_score` import so it can be used both with Google's
rouge-score package and with the XL-Sum multilingual scorer, which installs
under the same module name.
"""
import re

from nltk.stem import SnowballStemmer
from nltk.stem.isri import ISRIStemmer
import qalsadi.lemmatizer
from tashaphyne.stemming import ArabicLightStemmer

_PUNCT_RE = re.compile(r"([^\w\s])", flags=re.UNICODE)
_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u0640]")  # harakat + tatweel


def basic_tokenize(text):
    """XL-Sum BasicTokenizer style: lowercase, punctuation split off."""
    return [t for t in _PUNCT_RE.sub(r" \1 ", text.lower()).split() if t]


def orth_normalise(tok):
    """Orthographic normalisation commonly applied before Arabic IR/NLP."""
    tok = _DIACRITICS.sub("", tok)
    tok = re.sub("[\u0623\u0625\u0622\u0671]", "\u0627", tok)
    tok = tok.replace("ى", "ي").replace("ة", "ه")
    return tok


class _Cached:
    """Memoise an expensive normaliser. Returns its raw output, including ''."""

    def __init__(self, fn):
        self.fn, self.cache = fn, {}

    def __call__(self, w):
        if w not in self.cache:
            try:
                self.cache[w] = self.fn(w)
            except Exception:
                self.cache[w] = w
        return self.cache[w]


_snow = SnowballStemmer("arabic", ignore_stopwords=True)
_isri = ISRIStemmer()
_qal = qalsadi.lemmatizer.Lemmatizer()
_tash = ArabicLightStemmer()


def _tash_stem(w):
    _tash.light_stem(w)
    return _tash.get_stem()


def _tash_root(w):
    _tash.light_stem(w)
    return _tash.get_root()


NORMALISERS = {  # name -> token -> token (raw output; may be '')
    "none": lambda w: w,
    "orthographic": _Cached(orth_normalise),
    # Snowball is NOT memoised: it is cheap, and NLTK's ArabicStemmer carries
    # one flag (suffixes_verb_step1_success) across calls, so memoising changes
    # results slightly. Calling it per occurrence reproduces the Kaggle runs.
    "Snowball stemmer": _snow.stem,
    "ISRI stemmer": _Cached(_isri.stem),
    "Tashaphyne light stem": _Cached(_tash_stem),
    "Tashaphyne root": _Cached(_tash_root),
    "Qalsadi lemmatiser": _Cached(_qal.lemmatize),
}

