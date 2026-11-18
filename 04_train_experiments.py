"""Step 04 - model experiments with stratified, text-grouped 5-fold CV on the TRAINING set.

Candidates, all scored on the same folds:
  * 2 vectorizers x 4 n-gram ranges x 3 models   (the proposal's 24-config grid)
  * word+char TF-IDF x 3 models                  (extra experiment, reported only;
                                                  from Train_Sentiment.py)
  * AFINN lexicon baseline                       (train_grid.py)
  * AFINN + Tagalog lexicon baseline             (Chen & Skiena 2014, unedited)
  * majority-class baseline
Every vectorizer lives inside the Pipeline and is rebuilt for every fold, so
vocabulary and IDF are learned from the training fold only. Identical texts
stay in the same fold. Scores: macro-F1, macro precision/recall, accuracy,
per-class F1.

The best row of the proposed 24-config grid by macro-F1 is refitted on all
training data.
The test set is not touched here; 05_evaluate.py scores it once.

Outputs: output/04_training/cv_results.csv, best_model.joblib, best_config.json
Run: python 04_train_experiments.py
"""
import json
import re
import time
from functools import partial

import joblib
import numpy as np
import pandas as pd
from afinn import Afinn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from config import (BEST_CONFIG_JSON, BEST_MODEL_PATH, CHAR_MIN_DF, CHAR_NGRAM,
                    CV_RESULTS_CSV, LABELS, LR_C, LR_MAX_ITER, N_FOLDS, NB_ALPHA,
                    NGRAM_RANGES, SEED, SVC_C, TL_NEGATIVE_FILE, TL_POSITIVE_FILE,
                    TRAIN_CSV, WC_LR_C, WC_NB_ALPHA, WC_SVC_C, WORD_MIN_DF, WORD_NGRAM)
from text_prep import normalize, tokenize

VECTORIZERS = {"CountVectorizer": CountVectorizer, "TfidfVectorizer": TfidfVectorizer}
GRID_MODELS = {
    "MultinomialNB": lambda: MultinomialNB(alpha=NB_ALPHA),
    "LogisticRegression": lambda: LogisticRegression(
        class_weight="balanced", max_iter=LR_MAX_ITER, C=LR_C, random_state=SEED),
    "LinearSVC": lambda: LinearSVC(class_weight="balanced", C=SVC_C, random_state=SEED),
}
WORD_CHAR = "Word+Char TF-IDF"
WORD_CHAR_NGRAM = f"word {WORD_NGRAM} + char_wb {CHAR_NGRAM}"
WORD_CHAR_MODELS = {
    "MultinomialNB": lambda: MultinomialNB(alpha=WC_NB_ALPHA),
    "LogisticRegression": lambda: LogisticRegression(
        max_iter=LR_MAX_ITER, class_weight="balanced", C=WC_LR_C, random_state=SEED),
    "LinearSVC": lambda: LinearSVC(class_weight="balanced", C=WC_SVC_C, random_state=SEED),
}
AFINN_NAME = "AFINN lexicon (baseline)"
TL_LEXICON_NAME = "AFINN + Tagalog lexicon (baseline)"
MAJORITY_NAME = "Majority class (baseline)"
BASELINES = {AFINN_NAME, TL_LEXICON_NAME, MAJORITY_NAME}
# Word+Char rows go beyond the 24-configuration grid in the proposal: they are
# scored and reported, but the final model is picked from the proposed grid.
EXTRA_NOTE = "extra experiment (outside the proposed 24-configuration grid)"
AFINN_NOTE = ("AFINN is an English-only lexicon; Tagalog/Taglish words "
              "are unknown to it, so it under-performs on this dataset.")
TL_LEXICON_NOTE = ("AFINN plus the Chen & Skiena (2014) Tagalog lexicon, used unedited "
                   "(+1 positive / -1 negative word). The Tagalog list is machine-built "
                   "and noisy (e.g. 'pagkain' = positive, 'din' = negative).")


def build_pipeline(vec_name, ngram, model_name):
    vectorizer = VECTORIZERS[vec_name](
        preprocessor=normalize,   # lowercase, strip URLs, collapse repeats
        tokenizer=tokenize,       # emoji-aware, stopwords out, negations kept
        token_pattern=None,
        lowercase=False,          # normalize() already lowercases
        ngram_range=ngram,
    )
    return Pipeline([("vec", vectorizer), ("clf", GRID_MODELS[model_name]())])


def build_word_char_pipeline(model_name):
    """Train_Sentiment.py's FeatureUnion, on text_prep's normalize/tokenize."""
    features = FeatureUnion([
        ("word", TfidfVectorizer(
            preprocessor=normalize, tokenizer=tokenize, token_pattern=None,
            lowercase=False, ngram_range=WORD_NGRAM, min_df=WORD_MIN_DF, sublinear_tf=True)),
        ("char", TfidfVectorizer(
            analyzer="char_wb", preprocessor=normalize, ngram_range=CHAR_NGRAM,
            min_df=CHAR_MIN_DF, sublinear_tf=True)),
    ])
    return Pipeline([("features", features), ("clf", WORD_CHAR_MODELS[model_name]())])


def candidates():
    """config name -> (row metadata, pipeline factory)."""
    out = {}
    for vec_name in VECTORIZERS:
        for ngram in NGRAM_RANGES:
            for model_name in GRID_MODELS:
                out[f"{vec_name} {ngram} + {model_name}"] = (
                    dict(vectorizer=vec_name, ngram_range=str(ngram), model=model_name),
                    partial(build_pipeline, vec_name, ngram, model_name))
    for model_name in WORD_CHAR_MODELS:
        out[f"{WORD_CHAR} + {model_name}"] = (
            dict(vectorizer=WORD_CHAR, ngram_range=WORD_CHAR_NGRAM, model=model_name),
            partial(build_word_char_pipeline, model_name))
    out[MAJORITY_NAME] = (
        dict(vectorizer="none", ngram_range="", model="DummyClassifier(most_frequent)"),
        lambda: DummyClassifier(strategy="most_frequent"))
    return out


def fold_scores(y_true, y_pred):
    per_class = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    return {
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, labels=LABELS, average="macro",
                                           zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, labels=LABELS, average="macro",
                                     zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        **{f"f1_{lab.lower()}": s for lab, s in zip(LABELS, per_class)},
    }


def summarize(name, fold_rows, fit_seconds, **meta):
    df = pd.DataFrame(fold_rows)
    row = {"config": name, **meta}
    for col in df.columns:
        row[f"{col}_mean"] = df[col].mean()
        row[f"{col}_std"] = df[col].std(ddof=1)
    row["fit_seconds"] = round(fit_seconds, 2)
    return row


def read_word_list(path):
    return {w.strip().lower() for w in path.read_text(encoding="utf-8").splitlines() if w.strip()}


class AfinnClassifier:
    """Sum of word scores: >0 Positive, <0 Negative, 0 Neutral.

    AFINN scores the English words. With tagalog=True, each word in the Tagalog
    positive/negative list adds +1/-1, unless AFINN already scores it (the
    Tagalog list contains some English words, e.g. "top", "right")."""

    _WORD_RE = re.compile(r"[a-zñ]+")

    def __init__(self, tagalog=False):
        self.afinn = Afinn(language="en", emoticons=False)
        self.tl_pos = read_word_list(TL_POSITIVE_FILE) if tagalog else set()
        self.tl_neg = read_word_list(TL_NEGATIVE_FILE) if tagalog else set()

    def _tagalog_hits(self, norm):
        words = [w for w in self._WORD_RE.findall(norm) if w not in self.afinn._dict]
        return [1 for w in words if w in self.tl_pos] + [-1 for w in words if w in self.tl_neg]

    def score(self, text):
        norm = normalize(text)
        return self.afinn.score(norm) + sum(self._tagalog_hits(norm))

    def predict(self, texts):
        return np.array([self._label(self.score(t)) for t in texts])

    def n_without_known_words(self, texts):
        return sum(1 for t in texts
                   if not self.afinn.find_all(normalize(t)) and not self._tagalog_hits(normalize(t)))

    @staticmethod
    def _label(score):
        return "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"


def run_candidates(configs, X, y, folds):
    rows, total = [], len(configs) + 2  # +2 for the two lexicon baselines
    for i, (name, (meta, make)) in enumerate(configs.items(), 1):
        start, fold_rows = time.perf_counter(), []
        for train_idx, val_idx in folds:
            pipe = make()  # fresh, unfitted per fold
            pipe.fit(X[train_idx], y[train_idx])
            fold_rows.append(fold_scores(y[val_idx], pipe.predict(X[val_idx])))
        row = summarize(name, fold_rows, time.perf_counter() - start, **meta,
                        note="baseline" if name in BASELINES
                        else "" if meta["vectorizer"] in VECTORIZERS else EXTRA_NOTE)
        rows.append(row)
        print(f"  [{i:2d}/{total}] {name:<55} macro-F1 "
              f"{row['macro_f1_mean']:.3f} ± {row['macro_f1_std']:.3f}")
    return rows


def run_lexicon(X, y, folds, i, total, tagalog=False):
    lex = AfinnClassifier(tagalog=tagalog)
    name = TL_LEXICON_NAME if tagalog else AFINN_NAME
    start, fold_rows = time.perf_counter(), []
    for _, val_idx in folds:  # nothing to fit; score the same validation folds
        fold_rows.append(fold_scores(y[val_idx], lex.predict(X[val_idx])))
    row = summarize(name, fold_rows, time.perf_counter() - start,
                    vectorizer="AFINN + Tagalog lexicon" if tagalog else "AFINN lexicon",
                    ngram_range="(1, 1)", model="word-score sum",
                    note=TL_LEXICON_NOTE if tagalog else AFINN_NOTE)
    stats = {"zero_score_total": int(sum(lex.score(t) == 0 for t in X)),
             "zero_no_known_words": lex.n_without_known_words(X), "n": len(X)}
    print(f"  [{i:2d}/{total}] {name:<55} macro-F1 "
          f"{row['macro_f1_mean']:.3f} ± {row['macro_f1_std']:.3f}")
    return row, stats


def main():
    train = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig")
    X, y = train["text_clean"].astype(str).to_numpy(), train["sentiment"].to_numpy()
    # Grouped by identical text so a repeated review ("good") never sits in both
    # the training and the validation part of a fold.
    groups = train["text_key"].fillna("").astype(str).to_numpy()
    folds = list(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED)
                 .split(X, y, groups=groups))

    configs = candidates()
    print(f"Experiments - {len(X):,} training reviews, {N_FOLDS}-fold stratified CV\n")
    rows = run_candidates(configs, X, y, folds)
    total = len(configs) + 2
    afinn_row, afinn_stats = run_lexicon(X, y, folds, total - 1, total)
    tl_row, tl_stats = run_lexicon(X, y, folds, total, total, tagalog=True)
    rows += [afinn_row, tl_row]

    cv = pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)
    cv.insert(0, "rank", range(1, len(cv) + 1))
    cv["in_proposed_grid"] = cv["vectorizer"].isin(list(VECTORIZERS))
    # Final model = best macro-F1 within the proposed grid; baselines and the
    # extra Word+Char experiments are reference points only.
    best = cv[cv["in_proposed_grid"]].iloc[0]
    cv["is_best"] = cv["config"] == best["config"]
    cv.to_csv(CV_RESULTS_CSV, index=False)

    model = configs[best["config"]][1]().fit(X, y)
    joblib.dump(model, BEST_MODEL_PATH)
    BEST_CONFIG_JSON.write_text(json.dumps({
        "config": best["config"], "vectorizer": best["vectorizer"],
        "ngram_range": best["ngram_range"], "model": best["model"],
        "cv_macro_f1_mean": float(best["macro_f1_mean"]),
        "cv_macro_f1_std": float(best["macro_f1_std"]),
        "cv_accuracy_mean": float(best["accuracy_mean"]),
        "cv_macro_precision_mean": float(best["macro_precision_mean"]),
        "cv_macro_recall_mean": float(best["macro_recall_mean"]),
        "selection": "best macro-F1 within the proposed 24-configuration grid",
        "afinn": {**afinn_stats, "note": AFINN_NOTE},
        "afinn_tagalog": {**tl_stats, "note": TL_LEXICON_NOTE},
        "n_train": len(X), "cv_folds": N_FOLDS, "seed": SEED,
    }, indent=2), encoding="utf-8")

    line = "=" * 100
    print(f"\n{line}\nNUMBERS TO PRESENT - CROSS-VALIDATION ({N_FOLDS}-fold, sorted by macro-F1)\n{line}")
    table = cv.rename(columns={"accuracy_mean": "acc", "macro_precision_mean": "prec",
                               "macro_recall_mean": "rec", "f1_negative_mean": "F1 neg",
                               "f1_neutral_mean": "F1 neu", "f1_positive_mean": "F1 pos"})
    table["macro-F1"] = [f"{m:.3f} ± {s:.3f}" for m, s in zip(cv.macro_f1_mean, cv.macro_f1_std)]
    table["config"] = [c + ("  <-- best" if c == best["config"]
                            else "  (extra)" if v == WORD_CHAR else "")
                       for c, v in zip(table["config"], table["vectorizer"])]
    print(table[["rank", "config", "macro-F1", "prec", "rec", "acc", "F1 neg", "F1 neu", "F1 pos"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    wc = cv[cv["vectorizer"] == WORD_CHAR].iloc[0]
    print(f"\nBest model (proposed 24-config grid): {best['config']}  CV macro-F1 "
          f"{best['macro_f1_mean']:.3f} ± {best['macro_f1_std']:.3f}, precision "
          f"{best['macro_precision_mean']:.3f}, recall {best['macro_recall_mean']:.3f}, "
          f"accuracy {best['accuracy_mean']:.1%}")
    print(f"Best extra word+char config (not selectable): {wc['config']}  (rank {wc['rank']}) "
          f"macro-F1 {wc['macro_f1_mean']:.3f} ± {wc['macro_f1_std']:.3f}")
    print(f"AFINN baseline: {afinn_stats['zero_score_total']:,} of {afinn_stats['n']:,} training "
          f"reviews scored 0 (-> Neutral); {afinn_stats['zero_no_known_words']:,} contain no AFINN word.")
    print(f"AFINN + Tagalog: {tl_stats['zero_score_total']:,} scored 0; "
          f"{tl_stats['zero_no_known_words']:,} contain no word from either lexicon.")
    print(f"Saved: {CV_RESULTS_CSV.name}, {BEST_MODEL_PATH.name}, {BEST_CONFIG_JSON.name}")


if __name__ == "__main__":
    main()
