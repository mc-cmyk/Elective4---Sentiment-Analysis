"""Step 01 - inspect the raw data (from Elec4dataset_test.py). Writes nothing.

Shape, missing values, duplicates, star-rating distribution, language mix.
Run: python 01_inspect_data.py
"""
import pandas as pd

from config import (LABELS, LEGACY_RESTOS_FILE, NEGATIVE_MAX_STAR, NEUTRAL_STAR,
                    RESTOS_FILE, REVIEWS_FILE, resolve_raw_dir)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def to_sent(x):
    if pd.isna(x):
        return None
    if x <= NEGATIVE_MAX_STAR:
        return LABELS[0]
    if x == NEUTRAL_STAR:
        return LABELS[1]
    return LABELS[2]


raw = resolve_raw_dir()
restos = pd.read_csv(raw / RESTOS_FILE, low_memory=False)
reviews = pd.read_csv(raw / REVIEWS_FILE, low_memory=False)
legacy = pd.read_csv(raw / LEGACY_RESTOS_FILE, low_memory=False)
tables = {REVIEWS_FILE: reviews, RESTOS_FILE: restos, LEGACY_RESTOS_FILE: legacy}

# ------------------------------------------------------------------ shape
rule("SHAPE")
for name, df in tables.items():
    print(f"{name:<22} rows {len(df):>9,}   cols {df.shape[1]}")

# ------------------------------------------------------------------ missing
rule("MISSING VALUES  (columns with any missing)")
for name, df in tables.items():
    miss = df.isna().sum()
    miss = miss[miss > 0]
    print(f"\n{name}:")
    print("  none" if miss.empty else
          miss.to_frame("n").assign(pct=lambda d: d.n / len(df)).to_string(float_format=lambda v: f"{v:.1%}"))

# ------------------------------------------------------------------ duplicates
rule("DUPLICATES")
dup_rows = int(reviews.duplicated().sum())
dup_uuid = int(reviews.duplicated("uuid").sum())
dup_store = int(restos.duplicated("StoreId", keep=False).sum())
print(f"{REVIEWS_FILE}: fully duplicated rows {dup_rows:,}   duplicated uuid {dup_uuid:,}")
print(f"{RESTOS_FILE}: rows sharing a StoreId {dup_store:,}  "
      f"(rows {len(restos):,}, unique StoreId {restos['StoreId'].nunique():,})")

# ------------------------------------------------------------------ ratings
rule("RATING DISTRIBUTION  (column 'overall')")
overall = pd.to_numeric(reviews["overall"], errors="coerce")
print(f"present {overall.notna().sum():,} ({overall.notna().mean():.1%})   mean {overall.mean():.2f}")
print(overall.value_counts().sort_index().to_string())

reviews["_overall"] = overall
dist = reviews["_overall"].map(to_sent).value_counts()
print("\n3-class distribution:")
print(dist.to_frame("n").assign(pct=lambda d: d.n / d.n.sum()).to_string())
baseline = dist.max() / dist.sum()
print(f"\n>> MAJORITY-CLASS BASELINE = {baseline:.1%}  (always predict '{dist.idxmax()}')")

# ------------------------------------------------------------------ language mix
rule("LANGUAGE MIX  (Tagalog vs Bisaya markers)")
txt_ok = reviews["text"].notna() & (reviews["text"].astype(str).str.strip() != "")
stores = restos.drop_duplicates(subset="StoreId", keep="first")
m = reviews[txt_ok].merge(stores[["StoreId", "City"]], on="StoreId", how="inner")

TAGALOG = ["ang", "ng", "mga", "ay", "po", "hindi", "walang", "sobrang", "talaga",
           "lang", "yung", "ako", "namin", "kasi", "masarap", "malamig", "matagal",
           "kulang", "mali", "sana", "grabe", "salamat", "naman", "dito", "kami"]
BISAYA = ["nindot", "kaayo", "kaayong", "kaayug", "daghan", "pud", "gyud", "lami",
          "wala", "naa", "unsa", "pila", "sulod", "ila", "ako", "nimo", "ninyo",
          "maayo", "dili", "pag", "akong", "imong"]

txt = m["text"].astype(str)
has_tag = txt.str.contains(r"\b(?:" + "|".join(TAGALOG) + r")\b", case=False, regex=True)
has_bis = txt.str.contains(r"\b(?:" + "|".join(BISAYA) + r")\b", case=False, regex=True)
english = ~has_tag & ~has_bis
print(f"Reviews with text, joined to a store: {len(m):,}")
print(f"Tagalog markers : {has_tag.sum():,} ({has_tag.mean():.1%})")
print(f"Bisaya markers  : {has_bis.sum():,} ({has_bis.mean():.1%})")
print(f"Both            : {(has_tag & has_bis).sum():,}")
print(f"Pure English    : {english.sum():,} ({english.mean():.1%})")

print("\nBisaya share by city (top 12 cities by volume):")
m["_bis"] = has_bis.values
top_cities = m.City.value_counts().head(12).index
print(m[m.City.isin(top_cities)].groupby("City")
      .agg(n=("text", "size"), bisaya_pct=("_bis", "mean"))
      .sort_values("n", ascending=False).to_string())

# ------------------------------------------------------------------ numbers
rule("NUMBERS TO PRESENT")
print(f"Reviews {len(reviews):,} x {reviews.shape[1]} cols | stores {len(restos):,} "
      f"({restos['StoreId'].nunique():,} unique StoreId)")
print(f"Missing review text: {reviews['text'].isna().sum():,} | duplicated uuid: {dup_uuid:,} | "
      f"StoreId duplicate rows: {dup_store:,}")
print("Sentiment: " + ", ".join(f"{k} {v / dist.sum():.1%}" for k, v in dist.items())
      + f" | majority baseline {baseline:.1%}")
print(f"Language: Tagalog {has_tag.mean():.1%}, Bisaya {has_bis.mean():.1%}, "
      f"pure English {english.mean():.1%}")
