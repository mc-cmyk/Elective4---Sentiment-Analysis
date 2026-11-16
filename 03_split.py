"""Step 03 - stratified 80/20 train/test split, grouped by review text.

All reviews are kept, short ones included (flagged by is_short). Identical
texts (same text_key, e.g. many customers writing just "good") are kept
together on one side of the split, so the test set never contains a review
the model saw word-for-word in training.

Outputs: output/03_split/<brand>_train.csv, <brand>_test.csv
Run: python 03_split.py
"""
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from config import ALL_CSV, SEED, TEST_CSV, TEST_FRACTION, TRAIN_CSV

df = pd.read_csv(ALL_CSV, encoding="utf-8-sig")
df["text_key"] = df["text_key"].fillna("").astype(str)

# One fold of a stratified, grouped K-fold split = the test set.
splitter = StratifiedGroupKFold(n_splits=round(1 / TEST_FRACTION), shuffle=True, random_state=SEED)
train_idx, test_idx = next(splitter.split(df, df["sentiment"], groups=df["text_key"]))
train, test = df.iloc[train_idx], df.iloc[test_idx]
assert not set(train["text_key"]) & set(test["text_key"]), "text shared across the split"

train.to_csv(TRAIN_CSV, index=False, encoding="utf-8-sig")
test.to_csv(TEST_CSV, index=False, encoding="utf-8-sig")

balance = pd.DataFrame({
    "train_n": train.sentiment.value_counts(),
    "train_pct": train.sentiment.value_counts(normalize=True),
    "test_n": test.sentiment.value_counts(),
    "test_pct": test.sentiment.value_counts(normalize=True),
})
baseline = test.sentiment.value_counts(normalize=True).max()

print("=" * 78 + "\nNUMBERS TO PRESENT\n" + "=" * 78)
print(f"Pool {len(df):,} -> train {len(train):,} | test {len(test):,} "
      f"(stratified by sentiment, grouped by identical text, seed {SEED})")
print("Texts shared between train and test: 0")
print("\nClass balance:")
print(balance.to_string(float_format=lambda v: f"{v:.1%}"))
print(f"\nMajority-class baseline on the test set: {baseline:.1%}")
