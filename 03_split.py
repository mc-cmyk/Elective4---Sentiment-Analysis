"""Step 03 - stratified 80/20 train/test split (from jolibee_extract.py).

All reviews are kept, short ones included (flagged by is_short).
Outputs: output/<brand>_train.csv, output/<brand>_test.csv
Run: python 03_split.py
"""
import pandas as pd

from config import ALL_CSV, SEED, TEST_CSV, TEST_FRACTION, TRAIN_CSV

df = pd.read_csv(ALL_CSV, encoding="utf-8-sig")
test = df.groupby("sentiment", group_keys=False).sample(frac=TEST_FRACTION, random_state=SEED)
train = df.drop(test.index)

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
      f"(stratified by sentiment, seed {SEED})")
print("\nClass balance:")
print(balance.to_string(float_format=lambda v: f"{v:.1%}"))
print(f"\nMajority-class baseline on the test set: {baseline:.1%}")
