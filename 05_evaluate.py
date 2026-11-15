"""Step 05 - score the held-out test set once with the model saved by step 04.

Outputs: output/predictions_test.csv (schema the dashboard reads),
         output/test_metrics.json
Run: python 05_evaluate.py
"""
import json

import joblib
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

import text_prep  # noqa: F401  - the saved pipeline references its tokenizer
from config import (BEST_CONFIG_JSON, BEST_MODEL_PATH, LABELS, PREDICTIONS_CSV,
                    SHORT_REVIEW_WORDS, TEST_CSV, TEST_METRICS_JSON, TRAIN_CSV)


def group_acc(mask, correct):
    return {"n": int(mask.sum()), "accuracy": float(correct[mask].mean()) if mask.any() else None}


def main():
    model = joblib.load(BEST_MODEL_PATH)
    best = json.loads(BEST_CONFIG_JSON.read_text(encoding="utf-8"))
    train = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig")
    test = pd.read_csv(TEST_CSV, encoding="utf-8-sig")
    X_test = test["text_clean"].astype(str).to_numpy()
    y_test = test["sentiment"].to_numpy()

    pred = model.predict(X_test)
    correct = pred == y_test
    acc = accuracy_score(y_test, pred)
    macro = f1_score(y_test, pred, labels=LABELS, average="macro")
    report = classification_report(y_test, pred, labels=LABELS, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, pred, labels=LABELS)

    majority_label = train["sentiment"].value_counts().idxmax()
    majority_acc = float((y_test == majority_label).mean())
    short = test["is_short"].astype(bool).to_numpy()
    seen = test["text_clean"].astype(str).isin(set(train["text_clean"].astype(str))).to_numpy()

    pred_df = test[["uuid", "Branch", "City", "review_date", "rating", "sentiment",
                    "text_raw", "text_clean", "word_count", "is_short",
                    "rider_blamed", "branch_usable"]].copy()
    pred_df["predicted"] = pred
    pred_df["correct"] = correct
    pred_df.to_csv(PREDICTIONS_CSV, index=False)

    metrics = {
        "config": best["config"], "n_test": len(test),
        "test_accuracy": float(acc), "test_macro_f1": float(macro),
        "test_report": report,
        "confusion_matrix": {"labels": LABELS, "matrix": cm.tolist()},
        "majority_baseline": {"label": majority_label, "accuracy": majority_acc},
        "short_vs_long_accuracy": {"short": group_acc(short, correct),
                                   "long": group_acc(~short, correct)},
        "seen_in_train_accuracy": {"seen": group_acc(seen, correct),
                                   "unseen": group_acc(~seen, correct)},
    }
    TEST_METRICS_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    line = "=" * 78
    print(f"{line}\nNUMBERS TO PRESENT - HELD-OUT TEST SET ({len(test):,} reviews)\n{line}")
    print(f"Model: {best['config']}  (CV macro-F1 {best['cv_macro_f1_mean']:.3f})")
    print(f"Test accuracy {acc:.1%}   Test macro-F1 {macro:.3f}   "
          f"Majority baseline (always '{majority_label}') {majority_acc:.1%}\n")
    print(classification_report(y_test, pred, labels=LABELS, digits=3, zero_division=0))
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(pd.DataFrame(cm, index=[f"actual {l}" for l in LABELS],
                       columns=[f"pred {l}" for l in LABELS]).to_string())
    print(f"\nShort reviews (<{SHORT_REVIEW_WORDS} words): {correct[short].mean():.1%} accuracy "
          f"on {short.sum():,}")
    print(f"Longer reviews:            {correct[~short].mean():.1%} accuracy on {(~short).sum():,}")
    print(f"Test text also in train:   {correct[seen].mean():.1%} accuracy on {seen.sum():,}")
    print(f"Test text not in train:    {correct[~seen].mean():.1%} accuracy on {(~seen).sum():,}")
    print(f"\nSaved: {PREDICTIONS_CSV.name}, {TEST_METRICS_JSON.name}")


if __name__ == "__main__":
    main()
