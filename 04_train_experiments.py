"""Step 04 - model experiments with stratified 5-fold CV on the TRAINING set.

Candidates, all scored on the same folds:
  * 2 vectorizers x 4 n-gram ranges x 3 models   (grid from train_grid.py)
  * word+char TF-IDF x 3 models                  (features and model settings
                                                  from Train_Sentiment.py)
  * AFINN lexicon baseline                       (train_grid.py)
  * majority-class baseline
Every vectorizer lives inside the Pipeline and is rebuilt for every fold, so
vocabulary and IDF are learned from the training fold only.

The best non-baseline row by macro-F1 is refitted on all training data.
The test set is not touched here; 05_evaluate.py scores it once.

Outputs: output/cv_results.csv, output/best_model.joblib, output/best_config.json
Run: python 04_train_experiments.py
"""
import json
import time
from functools import partial

import joblib
import numpy as np
import pandas as pd
from afinn import Afinn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from config import (BEST_CONFIG_JSON, BEST_MODEL_PATH, CHAR_MIN_DF, CHAR_NGRAM,
                    CV_RESULTS_CSV, LABELS, LR_C, LR_MAX_ITER, N_FOLDS, NB_ALPHA,
                    NGRAM_RANGES, SEED, SVC_C, TRAIN_CSV, WC_LR_C, WC_NB_ALPHA,
                    WC_SVC_C, WORD_MIN_DF, WORD_NGRAM)
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
MAJORITY_NAME = "Majority class (baseline)"
BASELINES = {AFINN_NAME, MAJORITY_NAME}
AFINN_NOTE = ("AFINN is an English-only lexicon; Tagalog/Bisaya/Taglish words "
              "are unknown to it, so it under-performs on this dataset.")


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


class AfinnClassifier:
    """Sum of AFINN word scores: >0 Positive, <0 Negative, 0 Neutral."""

    def __init__(self):
        self.afinn = Afinn(language="en", emoticons=False)

    def predict(self, texts):
        return np.array([self._label(self.afinn.score(normalize(t))) for t in texts])

    def n_without_known_words(self, texts):
        return sum(1 for t in texts if not self.afinn.find_all(normalize(t)))

    @staticmethod
    def _label(score):
        return "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"


def run_candidates(configs, X, y, folds):
    rows, total = [], len(configs) + 1  # +1 for AFINN
    for i, (name, (meta, make)) in enumerate(configs.items(), 1):
        start, fold_rows = time.perf_counter(), []
        for train_idx, val_idx in folds:
            pipe = make()  # fresh, unfitted per fold
            pipe.fit(X[train_idx], y[train_idx])
            fold_rows.append(fold_scores(y[val_idx], pipe.predict(X[val_idx])))
        row = summarize(name, fold_rows, time.perf_counter() - start, **meta,
                        note="baseline" if name in BASELINES else "")
        rows.append(row)
        print(f"  [{i:2d}/{total}] {name:<55} macro-F1 "
              f"{row['macro_f1_mean']:.3f} ± {row['macro_f1_std']:.3f}")
    return rows


def run_afinn(X, y, folds, total):
    afinn = AfinnClassifier()
    start, fold_rows = time.perf_counter(), []
    for _, val_idx in folds:  # nothing to fit; score the same validation folds
        fold_rows.append(fold_scores(y[val_idx], afinn.predict(X[val_idx])))
    row = summarize(AFINN_NAME, fold_rows, time.perf_counter() - start,
                    vectorizer="AFINN lexicon", ngram_range="(1, 1)",
                    model="AFINN word-score sum", note=AFINN_NOTE)
    stats = {"zero_score_total": int(sum(afinn.afinn.score(normalize(t)) == 0 for t in X)),
             "zero_no_known_words": afinn.n_without_known_words(X), "n": len(X)}
    print(f"  [{total:2d}/{total}] {AFINN_NAME:<55} macro-F1 "
          f"{row['macro_f1_mean']:.3f} ± {row['macro_f1_std']:.3f}")
    return row, stats


def main():
    train = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig")
    X, y = train["text_clean"].astype(str).to_numpy(), train["sentiment"].to_numpy()
    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y))

    configs = candidates()
    print(f"Experiments - {len(X):,} training reviews, {N_FOLDS}-fold stratified CV\n")
    rows = run_candidates(configs, X, y, folds)
    afinn_row, afinn_stats = run_afinn(X, y, folds, len(configs) + 1)
    rows.append(afinn_row)

    cv = pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)
    cv.insert(0, "rank", range(1, len(cv) + 1))
    best = cv[~cv["config"].isin(BASELINES)].iloc[0]  # baselines are reference points
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
        "afinn": {**afinn_stats, "note": AFINN_NOTE},
        "n_train": len(X), "cv_folds": N_FOLDS, "seed": SEED,
    }, indent=2), encoding="utf-8")

    line = "=" * 100
    print(f"\n{line}\nNUMBERS TO PRESENT - CROSS-VALIDATION ({N_FOLDS}-fold, sorted by macro-F1)\n{line}")
    table = cv.rename(columns={"accuracy_mean": "acc", "f1_negative_mean": "F1 neg",
                               "f1_neutral_mean": "F1 neu", "f1_positive_mean": "F1 pos"})
    table["macro-F1"] = [f"{m:.3f} ± {s:.3f}" for m, s in zip(cv.macro_f1_mean, cv.macro_f1_std)]
    table["config"] = [c + ("  <-- best" if c == best["config"] else "") for c in table["config"]]
    print(table[["rank", "config", "macro-F1", "acc", "F1 neg", "F1 neu", "F1 pos"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    wc = cv[cv["vectorizer"] == WORD_CHAR].iloc[0]
    print(f"\nBest model: {best['config']}  CV macro-F1 {best['macro_f1_mean']:.3f} "
          f"± {best['macro_f1_std']:.3f}, CV accuracy {best['accuracy_mean']:.1%}")
    print(f"Best word+char config: {wc['config']}  (rank {wc['rank']}) "
          f"macro-F1 {wc['macro_f1_mean']:.3f} ± {wc['macro_f1_std']:.3f}")
    print(f"AFINN baseline: {afinn_stats['zero_score_total']:,} of {afinn_stats['n']:,} training "
          f"reviews scored 0 (-> Neutral); {afinn_stats['zero_no_known_words']:,} contain no AFINN word.")
    print(f"Saved: {CV_RESULTS_CSV.name}, {BEST_MODEL_PATH.name}, {BEST_CONFIG_JSON.name}")


if __name__ == "__main__":
    main()
