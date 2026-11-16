"""Step 02 - build the brand dataset (from jolibee_extract.py).

Filter to BRAND, label from stars, drop Bisaya reviews (scope is
English/Tagalog/Taglish), flag short and rider-blamed reviews,
flag usable branches, and build the complaint corpus. No split here (03).

Outputs (output/02_dataset/): <brand>_all.csv, <brand>_complaints.csv, <brand>_branch_summary.csv
Run: python 02_build_dataset.py
"""
import re

import pandas as pd

from config import (ALL_CSV, BISAYA_MARKERS, BRANCH_SUMMARY_CSV, BRAND, COMPLAINTS_CSV, LABELS,
                    MIN_REVIEWS_PER_BRANCH, NEGATIVE_MAX_STAR, NEUTRAL_STAR,
                    RESTOS_FILE, REVIEWS_FILE, RIDER_BLAME_MAX_STAR,
                    SHORT_REVIEW_WORDS, SPAM_MIN_WORDS, resolve_raw_dir)
from text_prep import normalize

INVISIBLE = r"[​‌‍‎‏⁠﻿­]"

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 80)


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def clean_text(s):
    """Strip invisible characters and normalise whitespace. Keeps case and emoji."""
    if pd.isna(s):
        return ""
    s = re.sub(INVISIBLE, "", str(s))
    return re.sub(r"\s+", " ", s).strip()


def branch_label(complete_name, brand):
    """'Jollibee - SM Megamall' -> 'SM Megamall'. Tolerates missing/odd separators."""
    c = clean_text(complete_name)
    stripped = re.sub(rf"^\s*{re.escape(brand)}\s*[-–—]?\s*", "", c, flags=re.I).strip()
    return stripped if stripped else c


def to_sentiment(x):
    if x <= NEGATIVE_MAX_STAR:
        return LABELS[0]
    if x == NEUTRAL_STAR:
        return LABELS[1]
    return LABELS[2]


# ------------------------------------------------------------------- load
rule("LOAD")
raw = resolve_raw_dir()
restos = pd.read_csv(raw / RESTOS_FILE, low_memory=False)
reviews = pd.read_csv(raw / REVIEWS_FILE, low_memory=False)
print(f"stores  {restos.shape}")
print(f"reviews {reviews.shape}")
restos["CompleteStoreName"] = restos["CompleteStoreName"].map(clean_text)

# ------------------------------------------------- duplicate StoreId audit
rule("DUPLICATE StoreId AUDIT")
brand_re = rf"^\s*{re.escape(BRAND)}\b"
dup_mask = restos.duplicated("StoreId", keep=False)
brand_prefix = restos["CompleteStoreName"].str.match(brand_re, case=False, na=False)
affected = restos[dup_mask & brand_prefix]
print(f"Duplicate StoreId rows dataset-wide: {dup_mask.sum()}")
print(f"...of which belong to {BRAND}: {len(affected)}")
if len(affected):
    print("\n!! These branches share an id. One will be lost on dedupe -- review manually:")
    print(affected.sort_values("StoreId")[["StoreId", "CompleteStoreName", "City"]].to_string(index=False))
else:
    print(f">> {BRAND} is unaffected. Safe to dedupe.")
restos = restos.drop_duplicates(subset="StoreId", keep="first").copy()

# --------------------------------------------------------------- brand filter
rule(f"FILTER TO {BRAND}")
stores = restos[restos["CompleteStoreName"].str.match(brand_re, case=False, na=False)].copy()
stores["Brand"] = BRAND
stores["Branch"] = stores["CompleteStoreName"].map(lambda c: branch_label(c, BRAND))
print(f"Matched {len(stores)} stores.")
print(stores[["StoreId", "CompleteStoreName", "Branch", "City"]]
      .sort_values("Branch").to_string(index=False))

# ------------------------------------------------------------------- join
rule("JOIN")
df = reviews.merge(
    stores[["StoreId", "Brand", "Branch", "CompleteStoreName", "City", "FoodType"]],
    on="StoreId", how="inner",
)
print(f"Reviews for {BRAND}: {len(df)}")
before = len(df)
df = df.drop_duplicates(subset="uuid", keep="first")
print(f"Duplicate uuids removed: {before - len(df)}")

# ------------------------------------------------------------------ labels
rule("LABELS AND FLAGS")
df["rating"] = pd.to_numeric(df["overall"], errors="coerce")
df = df[df["rating"].notna()].copy()
df["sentiment"] = df["rating"].map(to_sentiment)

# rider is effectively binary (1 or 5), so treat it as a flag, not a scale.
df["rider_rating"] = pd.to_numeric(df["rider"], errors="coerce")
df["rider_blamed"] = df["rider_rating"] <= RIDER_BLAME_MAX_STAR
# True when the complaint is NOT attributable to the rider (missing score counts as branch).
df["branch_attributable"] = ~df["rider_blamed"].fillna(False)

df["text_raw"] = df["text"]
df["text_clean"] = df["text"].map(clean_text)
df["word_count"] = df["text_clean"].str.split().str.len()
df["is_short"] = df["word_count"] < SHORT_REVIEW_WORDS   # kept, flagged
df["review_date"] = pd.to_datetime(df["createdAt"], errors="coerce", utc=True)
df = df[df["text_clean"].str.len() > 0].copy()

print(df["sentiment"].value_counts().to_frame("n").assign(pct=lambda d: d.n / d.n.sum()).to_string())

# ----------------------------------------------------------- language scope
rule("LANGUAGE SCOPE  (drop Bisaya reviews)")
bisaya_re = r"\b(?:" + "|".join(BISAYA_MARKERS) + r")\b"
is_bisaya = df["text_clean"].str.contains(bisaya_re, case=False, regex=True)
n_before_lang = len(df)
print(f"Bisaya reviews dropped: {is_bisaya.sum():,} of {n_before_lang:,} ({is_bisaya.mean():.1%})")
print(df[is_bisaya].groupby("City").size().sort_values(ascending=False).head(10)
      .to_frame("dropped").to_string())
df = df[~is_bisaya].copy()

# ------------------------------------------------------ near-duplicate spam
rule("NEAR-DUPLICATE SPAM")
# Same text after normalisation (case, URLs, repeated letters) and long enough
# that two customers are unlikely to write it independently. Short generic
# texts ("good", "no gravy") are real, separate opinions and are kept; 03/04
# keep identical texts on the same side of every split instead.
df["text_key"] = df["text_clean"].map(normalize).str.strip()
spam = (df["word_count"] >= SPAM_MIN_WORDS) & df.duplicated("text_key", keep="first")
print(f"Repeated texts of >= {SPAM_MIN_WORDS} words dropped as spam: {spam.sum():,}")
df = df[~spam].copy()
print(f"Short texts still repeated (kept, grouped at split time): "
      f"{df.duplicated('text_key').sum():,}")

# ----------------------------------------------------------- branch viability
rule("BRANCH VIABILITY")
branch_stats = df.groupby(["StoreId", "Branch", "City"]).agg(
    n_reviews=("uuid", "size"),
    n_negative=("sentiment", lambda s: (s == "Negative").sum()),
    n_neutral=("sentiment", lambda s: (s == "Neutral").sum()),
    n_positive=("sentiment", lambda s: (s == "Positive").sum()),
    mean_rating=("rating", "mean"),
).reset_index()
branch_stats["pct_negative"] = branch_stats.n_negative / branch_stats.n_reviews
branch_stats["usable"] = branch_stats.n_reviews >= MIN_REVIEWS_PER_BRANCH
branch_stats = branch_stats.sort_values("n_reviews", ascending=False)
df["branch_usable"] = df["StoreId"].isin(branch_stats.loc[branch_stats.usable, "StoreId"])
print(branch_stats.to_string(index=False))

# ------------------------------------------------------- complaint corpus
rule("COMPLAINT CORPUS")
# Negative AND not the rider's fault AND in a usable branch.
complaints = df[(df.sentiment == "Negative") & df.branch_attributable & df.branch_usable].copy()
print(complaints.groupby("Branch").size().sort_values(ascending=False).to_string())

# ---------------------------------------------------------------- write out
rule("WRITING FILES")
cols = ["uuid", "StoreId", "Brand", "Branch", "CompleteStoreName", "City", "FoodType",
        "review_date", "text_raw", "text_clean", "text_key", "word_count", "is_short",
        "rating", "sentiment", "rider_rating", "rider_blamed", "branch_attributable",
        "branch_usable"]
for path, frame in {ALL_CSV: df[cols], COMPLAINTS_CSV: complaints[cols],
                    BRANCH_SUMMARY_CSV: branch_stats}.items():
    frame.to_csv(path, index=False, encoding="utf-8-sig")  # utf-8-sig: Excel-safe
    print(f"  {path.name}  ({len(frame)} rows)")

rule("NUMBERS TO PRESENT")
print(f"{BRAND}: {len(stores)} stores matched, {len(df):,} labelled reviews")
print(f"Date range {df['review_date'].min().date()} -> {df['review_date'].max().date()}")
print("Sentiment: " + ", ".join(f"{k} {v:,} ({v / len(df):.1%})"
                               for k, v in df["sentiment"].value_counts().items()))
print(f"Near-duplicate spam dropped (>= {SPAM_MIN_WORDS} words, repeated text): {int(spam.sum()):,}")
print(f"Bisaya reviews dropped (scope: English/Tagalog/Taglish): "
      f"{n_before_lang - len(df):,} of {n_before_lang:,}")
print(f"Short reviews (<{SHORT_REVIEW_WORDS} words), kept and flagged: "
      f"{df['is_short'].sum():,} ({df['is_short'].mean():.1%})")
print(f"Rider-blamed reviews: {int(df['rider_blamed'].sum()):,}")
print(f"Branches: {len(branch_stats)}, usable (>={MIN_REVIEWS_PER_BRANCH} reviews): "
      f"{int(branch_stats.usable.sum())}")
print(f"Complaint corpus (negative, branch-attributable, usable branch): {len(complaints):,}")
