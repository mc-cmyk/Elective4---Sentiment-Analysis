"""ReviewLens - the whole sentiment analytics pipeline in one file.

Local Food Chain Customer Satisfaction & Review Sentiment Analytics System
(Jollibee, Foodpanda Philippines reviews; scope: English and Tagalog/Taglish).

Runs, in order (same logic as the numbered scripts 01-06):
  STEP 1   Inspect the raw data                       (writes nothing)
  STEP 2   Build the dataset: Jollibee filter, star-rating labels, drop Bisaya
           reviews and near-duplicate spam, flags, complaint corpus
  STEP 2b  Manual proxy-label check (sheet to fill in / agreement + kappa)
  STEP 3   Stratified 80/20 split, identical texts kept on one side
  STEP 4   Experiments: 24-config proposed grid + 3 extra word+char models +
           AFINN, AFINN+Tagalog and majority baselines, text-grouped 5-fold CV;
           final model = best macro-F1 within the proposed grid
  STEP 5   Held-out test evaluation (scored once)
  STEP 6   Complaint extraction: n-gram terms + regex themes, branch priorities

Outputs: output/02_dataset, 03_split, 04_training, 05_evaluation, 06_complaints
Run:     python sentiment_pipeline.py
Needs:   pandas numpy scikit-learn joblib afinn (+ kagglehub if data/raw is empty)
"""
import json
import re
import time
from functools import partial
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from afinn import Afinn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_colwidth", 80)

# =============================================================================
# CONFIGURATION
# =============================================================================
PROJECT_DIR = Path(__file__).resolve().parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "output"
KAGGLE_DATASET = "bwandowando/food-panda-restaurant-reviews"
RESTOS_FILE = "ph_restos_2025.csv"
REVIEWS_FILE = "ph_reviews_2025.csv"
LEGACY_RESTOS_FILE = "restos.csv"
RAW_FILES = (RESTOS_FILE, REVIEWS_FILE, LEGACY_RESTOS_FILE)

# Chen & Skiena (2014) "Sentiment Lexicons for 81 Languages" (Kaggle), Tagalog lists.
LEXICON_DIR = PROJECT_DIR / "81_lang_lexicon" / "sentiment-lexicons"
TL_POSITIVE_FILE = LEXICON_DIR / "positive_words_tl.txt"
TL_NEGATIVE_FILE = LEXICON_DIR / "negative_words_tl.txt"

BRAND = "Jollibee"
SEED = 42
TEST_FRACTION = 0.20
N_FOLDS = 5

NEGATIVE_MAX_STAR = 2           # 1-2 stars -> Negative
NEUTRAL_STAR = 3                # 3 stars   -> Neutral; 4-5 -> Positive
NEGATIVE, NEUTRAL, POSITIVE = LABELS = ["Negative", "Neutral", "Positive"]
RIDER_BLAME_MAX_STAR = 2        # rider score <= this -> complaint blamed on the rider
SHORT_REVIEW_WORDS = 3          # fewer words than this -> is_short (kept, flagged)
MIN_REVIEWS_PER_BRANCH = 30     # branches below this are flagged, not deleted
SPAM_MIN_WORDS = 4              # identical texts this long are copy-paste spam
LABEL_CHECK_N = 100             # reviews sampled for the manual proxy-label check

# Bisaya-only marker words (alphabetical): a review containing one is out of
# scope and dropped. Words shared with Tagalog (wala, ako, pag, akong, nako,
# maayo, ...) are left out so Tagalog reviews are kept.
BISAYA_MARKERS = [
    "bitaw", "bugnaw", "daghan", "didto", "dili", "diri", "dugay", "gamay",
    "ganahan", "gyud", "imong", "jud", "kaau", "kaayo", "kaayong", "kaayu",
    "karon", "kuwang", "lami", "lamia", "naa", "ngano", "nganong", "nimo",
    "nindot", "pud", "unsa", "unya", "walay",
]
BISAYA_RE = re.compile(r"\b(?:" + "|".join(BISAYA_MARKERS) + r")\b", re.IGNORECASE)

# Proposed grid: 2 vectorizers x 4 n-gram ranges x 3 models = 24 configurations.
NGRAM_RANGES = [(1, 1), (2, 2), (3, 3), (1, 3)]
LR_MAX_ITER, NB_ALPHA, LR_C, SVC_C = 2000, 1.0, 1.0, 1.0
# Extra word+char TF-IDF experiment (reported, not selectable).
WORD_NGRAM, WORD_MIN_DF, CHAR_NGRAM, CHAR_MIN_DF = (1, 2), 2, (3, 5), 3
WC_NB_ALPHA, WC_LR_C, WC_SVC_C = 0.5, 4.0, 0.5

# Complaint extraction.
TOP_N_CHAIN, TOP_N_BRANCH = 50, 15
NOTABLE_LIFT = 1.3
MIN_THEME_HITS = 3
CHAIN_MIN_DF, CHAIN_NGRAM = 3, (1, 3)
BRANCH_MIN_DF, BRANCH_NGRAM = 2, (2, 3)

DATASET_DIR = OUTPUT_DIR / "02_dataset"
SPLIT_DIR = OUTPUT_DIR / "03_split"
TRAINING_DIR = OUTPUT_DIR / "04_training"
EVALUATION_DIR = OUTPUT_DIR / "05_evaluation"
COMPLAINTS_DIR = OUTPUT_DIR / "06_complaints"
for _d in (DATASET_DIR, SPLIT_DIR, TRAINING_DIR, EVALUATION_DIR, COMPLAINTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

_B = BRAND.lower()
ALL_CSV = DATASET_DIR / f"{_B}_all.csv"
COMPLAINTS_CSV = DATASET_DIR / f"{_B}_complaints.csv"
BRANCH_SUMMARY_CSV = DATASET_DIR / f"{_B}_branch_summary.csv"
LABEL_SHEET = DATASET_DIR / "label_check_sample.csv"
LABEL_RESULT = DATASET_DIR / "label_check_result.txt"
TRAIN_CSV = SPLIT_DIR / f"{_B}_train.csv"
TEST_CSV = SPLIT_DIR / f"{_B}_test.csv"
CV_RESULTS_CSV = TRAINING_DIR / "cv_results.csv"
BEST_MODEL_PATH = TRAINING_DIR / "best_model.joblib"
BEST_CONFIG_JSON = TRAINING_DIR / "best_config.json"
PREDICTIONS_CSV = EVALUATION_DIR / "predictions_test.csv"
TEST_METRICS_JSON = EVALUATION_DIR / "test_metrics.json"
TERMS_CHAIN_CSV = COMPLAINTS_DIR / "complaint_terms_chain.csv"
TERMS_BRANCH_CSV = COMPLAINTS_DIR / "complaint_terms_branch.csv"
THEMES_BRANCH_CSV = COMPLAINTS_DIR / "complaint_themes_branch.csv"
BRANCH_PRIORITIES_CSV = COMPLAINTS_DIR / "branch_priorities.csv"


def resolve_raw_dir():
    """data/raw/ if it holds the raw CSVs; otherwise kagglehub's cached download."""
    if all((RAW_DIR / f).exists() for f in RAW_FILES):
        return RAW_DIR
    import kagglehub
    return Path(kagglehub.dataset_download(KAGGLE_DATASET))


def rule(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def pct(value):
    return f"{value:.1%}"


def to_sentiment(stars):
    """Proxy label from the star rating."""
    if stars <= NEGATIVE_MAX_STAR:
        return NEGATIVE
    if stars == NEUTRAL_STAR:
        return NEUTRAL
    return POSITIVE


def labelled_confusion(y_true, y_pred, true_name, pred_name):
    """Confusion matrix as a DataFrame with readable row/column names."""
    return pd.DataFrame(confusion_matrix(y_true, y_pred, labels=LABELS),
                        index=[f"{true_name} {lab}" for lab in LABELS],
                        columns=[f"{pred_name} {lab}" for lab in LABELS])


# =============================================================================
# TEXT PREPROCESSING
# - No stemming: English suffix rules mangle Tagalog affixes (na-, ma-, pag-).
# - Negation words are never removed ("not good" / "hindi masarap" flip polarity).
# - Emoji are kept as their own tokens.
# =============================================================================
NEGATIONS = frozenset({
    # English
    "arent", "cannot", "cant", "didnt", "doesnt", "dont", "isnt", "neither",
    "never", "no", "nobody", "none", "nor", "not", "nothing", "nowhere",
    "wasnt", "werent", "without", "wont",
    # Tagalog (hnd / wla: common Taglish spellings of hindi / wala)
    "di", "hindi", "hnd", "huwag", "wag", "wala", "walang", "wla",
})
TAGALOG_STOPWORDS = {
    "akin", "ako", "amin", "ang", "ano", "at", "atin", "ay", "ba", "daw",
    "din", "dito", "diyan", "doon", "dun", "e", "eh", "ikaw", "ito", "itong",
    "iyan", "iyon", "jan", "ka", "kami", "kasi", "kay", "kayo", "kc", "kina",
    "ko", "kung", "lahat", "lang", "lng", "mag", "may", "mga", "mo", "na",
    "naman", "namin", "nang", "natin", "ng", "nga", "ngayon", "ni", "nila",
    "nina", "ninyo", "niya", "nman", "nung", "nya", "nyo", "o", "opo", "pa",
    "pag", "pala", "para", "pero", "po", "raw", "rin", "sa", "sakin", "sayo",
    "si", "sila", "sina", "siya", "sya", "tapos", "tas", "tayo", "un", "ung",
    "yan", "yang", "yon", "yong", "yun", "yung",
}
STOPWORDS = frozenset((set(ENGLISH_STOP_WORDS) | TAGALOG_STOPWORDS) - NEGATIONS)

# Zero-width characters, BOM and soft hyphen.
INVISIBLE = r"[\u200b\u200c\u200d\u200e\u200f\u2060\ufeff\u00ad]"
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_REPEAT_RE = re.compile(r"(.)\1{2,}")
_APOS_RE = re.compile(r"['’`]")   # "didn't" -> "didnt"
# Skin tones, variation selectors and zero-width joiner, dropped so a thumbs-down
# emoji with or without a skin tone becomes the same token.
_EMOJI_MODIFIERS_RE = re.compile(r"[\U0001F3FB-\U0001F3FF\ufe0e\ufe0f\u200d]")
_EMOJI = (
    r"\U0001F300-\U0001F5FF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF"
    r"\U0001F900-\U0001F9FF\U0001FA70-\U0001FAFF\u2600-\u27bf\u2b50\u2b55"
)
_TOKEN_RE = re.compile(rf"[a-z0-9ñ]+|[{_EMOJI}]")


def clean_text(s):
    """Strip invisible characters and normalise whitespace. Keeps case and emoji."""
    if pd.isna(s):
        return ""
    return re.sub(r"\s+", " ", re.sub(INVISIBLE, "", str(s))).strip()


def normalize(text):
    """Lowercase, drop URLs, collapse 3+ repeated characters to 2."""
    text = _URL_RE.sub(" ", str(text).lower())
    return _REPEAT_RE.sub(r"\1\1", text)


def tokenize(text):
    """Split normalized text into word and emoji tokens, dropping stopwords."""
    text = _EMOJI_MODIFIERS_RE.sub("", _APOS_RE.sub("", text))
    return [t for t in _TOKEN_RE.findall(text) if t not in STOPWORDS]


# =============================================================================
# STEP 1 - INSPECT THE RAW DATA
# =============================================================================
def step1_inspect(restos, reviews, legacy):
    rule("STEP 1 - INSPECT RAW DATA")
    tables = {REVIEWS_FILE: reviews, RESTOS_FILE: restos, LEGACY_RESTOS_FILE: legacy}
    for name, df in tables.items():
        miss = df.isna().sum()
        print(f"{name:<22} rows {len(df):>9,}  cols {df.shape[1]:>3}  "
              f"columns with missing values: {int((miss > 0).sum())}")
    print(f"\nDuplicated review rows {int(reviews.duplicated().sum()):,} | duplicated uuid "
          f"{int(reviews.duplicated('uuid').sum()):,} | store rows sharing a StoreId "
          f"{int(restos.duplicated('StoreId', keep=False).sum()):,}")

    overall = pd.to_numeric(reviews["overall"], errors="coerce")
    print(f"\nStar ratings present {pct(overall.notna().mean())}, mean {overall.mean():.2f}")
    print(overall.value_counts().sort_index().to_string())
    dist = overall.dropna().map(to_sentiment).value_counts(normalize=True)
    print("\n3-class distribution: " + ", ".join(f"{k} {pct(v)}" for k, v in dist.items()))
    print(f">> Majority-class baseline (all reviews) = {pct(dist.max())}")

    texts = reviews["text"].dropna().astype(str)
    has_bisaya = texts.str.contains(BISAYA_RE)
    print(f"\nReviews with a Bisaya-only marker word: {has_bisaya.sum():,} of {len(texts):,} "
          f"({pct(has_bisaya.mean())}) - out of scope, dropped in step 2")


# =============================================================================
# STEP 2 - BUILD THE DATASET
# =============================================================================
def branch_label(complete_name):
    """'Jollibee - SM Megamall' -> 'SM Megamall'."""
    c = clean_text(complete_name)
    stripped = re.sub(rf"^\s*{re.escape(BRAND)}\s*[-–—]?\s*", "", c, flags=re.I).strip()
    return stripped or c


def step2_build(restos, reviews):
    rule(f"STEP 2 - BUILD THE {BRAND.upper()} DATASET")
    restos = restos.copy()
    restos["CompleteStoreName"] = restos["CompleteStoreName"].map(clean_text)
    brand_re = re.compile(rf"^\s*{re.escape(BRAND)}\b", re.IGNORECASE)
    is_brand = restos["CompleteStoreName"].str.match(brand_re, na=False)
    n_shared = int((restos.duplicated("StoreId", keep=False) & is_brand).sum())
    print(f"{BRAND} stores sharing a StoreId with another row: {n_shared}")

    restos = restos.drop_duplicates(subset="StoreId", keep="first")   # dedupe, then filter
    stores = restos[restos["CompleteStoreName"].str.match(brand_re, na=False)].copy()
    stores["Brand"] = BRAND
    stores["Branch"] = stores["CompleteStoreName"].map(branch_label)

    store_cols = ["StoreId", "Brand", "Branch", "CompleteStoreName", "City", "FoodType"]
    df = reviews.merge(stores[store_cols], on="StoreId", how="inner")
    n_joined = len(df)
    df = df.drop_duplicates(subset="uuid", keep="first")
    print(f"Stores matched {len(stores)} | reviews {n_joined:,} | duplicate uuids removed "
          f"{n_joined - len(df):,}")

    # Labels from stars; rider score as a blame flag.
    df["rating"] = pd.to_numeric(df["overall"], errors="coerce")
    df = df[df["rating"].notna()].copy()
    df["sentiment"] = df["rating"].map(to_sentiment)
    df["rider_rating"] = pd.to_numeric(df["rider"], errors="coerce")
    df["rider_blamed"] = df["rider_rating"] <= RIDER_BLAME_MAX_STAR
    df["branch_attributable"] = ~df["rider_blamed"].fillna(False)
    df["text_raw"] = df["text"]
    df["text_clean"] = df["text"].map(clean_text)
    df["word_count"] = df["text_clean"].str.split().str.len()
    df["is_short"] = df["word_count"] < SHORT_REVIEW_WORDS
    df["review_date"] = pd.to_datetime(df["createdAt"], errors="coerce", utc=True)
    df = df[df["text_clean"].str.len() > 0].copy()

    # Language scope: drop Bisaya reviews.
    is_bisaya = df["text_clean"].str.contains(BISAYA_RE)
    print(f"Bisaya reviews dropped: {int(is_bisaya.sum()):,} of {len(df):,}")
    df = df[~is_bisaya].copy()

    # Near-duplicate spam: identical (normalised) long texts. Short generic texts
    # ("good") are separate real opinions and are kept, grouped at split time.
    df["text_key"] = df["text_clean"].map(normalize).str.strip()
    spam = (df["word_count"] >= SPAM_MIN_WORDS) & df.duplicated("text_key", keep="first")
    df = df[~spam].copy()
    print(f"Near-duplicate spam dropped (>= {SPAM_MIN_WORDS} words): {int(spam.sum()):,}")

    branch_stats = df.groupby(["StoreId", "Branch", "City"]).agg(
        n_reviews=("uuid", "size"),
        n_negative=("sentiment", lambda s: (s == NEGATIVE).sum()),
        n_neutral=("sentiment", lambda s: (s == NEUTRAL).sum()),
        n_positive=("sentiment", lambda s: (s == POSITIVE).sum()),
        mean_rating=("rating", "mean"),
    ).reset_index()
    branch_stats["pct_negative"] = branch_stats["n_negative"] / branch_stats["n_reviews"]
    branch_stats["usable"] = branch_stats["n_reviews"] >= MIN_REVIEWS_PER_BRANCH
    branch_stats = branch_stats.sort_values("n_reviews", ascending=False)
    usable_ids = branch_stats.loc[branch_stats["usable"], "StoreId"]
    df["branch_usable"] = df["StoreId"].isin(usable_ids)

    # Complaint corpus: negative, not the rider's fault, in a usable branch.
    is_complaint = (df["sentiment"] == NEGATIVE) & df["branch_attributable"] & df["branch_usable"]
    complaints = df[is_complaint].copy()

    cols = ["uuid", "StoreId", "Brand", "Branch", "CompleteStoreName", "City", "FoodType",
            "review_date", "text_raw", "text_clean", "text_key", "word_count", "is_short",
            "rating", "sentiment", "rider_rating", "rider_blamed", "branch_attributable",
            "branch_usable"]
    df[cols].to_csv(ALL_CSV, index=False, encoding="utf-8-sig")
    complaints[cols].to_csv(COMPLAINTS_CSV, index=False, encoding="utf-8-sig")
    branch_stats.to_csv(BRANCH_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    print(f"\n{len(df):,} labelled reviews, {df['review_date'].min().date()} -> "
          f"{df['review_date'].max().date()}")
    print("Sentiment: " + ", ".join(f"{k} {v:,} ({pct(v / len(df))})"
                                   for k, v in df["sentiment"].value_counts().items()))
    print(f"Short reviews kept and flagged: {int(df['is_short'].sum()):,} | rider-blamed: "
          f"{int(df['rider_blamed'].sum()):,}")
    print(f"Branches {len(branch_stats)}, usable (>= {MIN_REVIEWS_PER_BRANCH} reviews) "
          f"{int(branch_stats['usable'].sum())} | complaint corpus {len(complaints):,}")
    return df[cols].reset_index(drop=True), complaints[cols].reset_index(drop=True)


# =============================================================================
# STEP 2b - MANUAL PROXY-LABEL CHECK
# First run writes the sheet; fill `manual_label` from the TEXT ONLY (hide rating
# and star_label). Later runs report agreement and Cohen's kappa. An existing
# sheet is never overwritten.
# =============================================================================
def step2b_label_check(df):
    rule("STEP 2b - MANUAL PROXY-LABEL CHECK")
    if not LABEL_SHEET.exists():
        per_class = LABEL_CHECK_N // len(LABELS)
        sample = (df.groupby("sentiment").sample(n=per_class, random_state=SEED)
                    .sample(frac=1, random_state=SEED))   # shuffle so classes are mixed
        sheet = sample[["uuid", "Branch", "text_clean", "rating", "sentiment"]].rename(
            columns={"sentiment": "star_label"})
        sheet["manual_label"] = ""
        sheet["notes"] = ""
        sheet.to_csv(LABEL_SHEET, index=False, encoding="utf-8-sig")
        print(f"Wrote {len(sheet)} reviews to {LABEL_SHEET}. Fill `manual_label` and re-run.")
        return

    sheet = pd.read_csv(LABEL_SHEET, encoding="utf-8-sig")
    manual = sheet["manual_label"].fillna("").astype(str).str.strip().str.capitalize()
    n_missing = int((~manual.isin(LABELS)).sum())
    if n_missing:
        print(f"{n_missing} of {len(sheet)} rows in {LABEL_SHEET.name} still need a "
              f"manual_label ({' / '.join(LABELS)}). Skipping.")
        return

    star = sheet["star_label"]
    kappa = cohen_kappa_score(star, manual, labels=LABELS)
    per_class = sheet.assign(ok=star == manual).groupby("star_label")["ok"].mean().reindex(LABELS)
    text = "\n".join([
        f"Reviews checked by hand: {len(sheet)}",
        f"Agreement between star label and manual label: {pct(float((star == manual).mean()))}",
        f"Cohen's kappa: {kappa:.2f}  (>0.60 substantial, >0.80 almost perfect)",
        "",
        "Agreement by star label:",
        per_class.to_string(float_format=pct),
        "",
        "Confusion (rows = star label, columns = manual label):",
        labelled_confusion(star, manual, "star", "manual").to_string(),
    ])
    print(text)
    LABEL_RESULT.write_text(text + "\n", encoding="utf-8")


# =============================================================================
# STEP 3 - TRAIN / TEST SPLIT
# =============================================================================
def step3_split(df):
    rule("STEP 3 - STRATIFIED 80/20 SPLIT (GROUPED BY IDENTICAL TEXT)")
    groups = df["text_key"].fillna("").astype(str)
    splitter = StratifiedGroupKFold(n_splits=round(1 / TEST_FRACTION), shuffle=True,
                                    random_state=SEED)
    train_idx, test_idx = next(splitter.split(df, df["sentiment"], groups=groups))
    train, test = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
    assert not set(train["text_key"]) & set(test["text_key"]), "text shared across the split"
    train.to_csv(TRAIN_CSV, index=False, encoding="utf-8-sig")
    test.to_csv(TEST_CSV, index=False, encoding="utf-8-sig")

    balance = pd.DataFrame({"train_n": train.sentiment.value_counts(),
                            "train_pct": train.sentiment.value_counts(normalize=True),
                            "test_n": test.sentiment.value_counts(),
                            "test_pct": test.sentiment.value_counts(normalize=True)})
    print(f"Pool {len(df):,} -> train {len(train):,} | test {len(test):,}; "
          "texts shared between train and test: 0")
    print(balance.to_string(float_format=pct))
    return train.reset_index(drop=True), test.reset_index(drop=True)


# =============================================================================
# STEP 4 - MODEL EXPERIMENTS (5-fold CV on the training set only)
# =============================================================================
VECTORIZERS = {"CountVectorizer": CountVectorizer, "TfidfVectorizer": TfidfVectorizer}
GRID_MODELS = {
    "MultinomialNB": lambda: MultinomialNB(alpha=NB_ALPHA),
    "LogisticRegression": lambda: LogisticRegression(
        class_weight="balanced", max_iter=LR_MAX_ITER, C=LR_C, random_state=SEED),
    "LinearSVC": lambda: LinearSVC(class_weight="balanced", C=SVC_C, random_state=SEED),
}
WORD_CHAR = "Word+Char TF-IDF"
WORD_CHAR_MODELS = {
    "MultinomialNB": lambda: MultinomialNB(alpha=WC_NB_ALPHA),
    "LogisticRegression": lambda: LogisticRegression(
        max_iter=LR_MAX_ITER, class_weight="balanced", C=WC_LR_C, random_state=SEED),
    "LinearSVC": lambda: LinearSVC(class_weight="balanced", C=WC_SVC_C, random_state=SEED),
}
PROPOSED, EXTRA, BASELINE = "Proposed grid", "Extra", "Baseline"   # config groups
AFINN_NAME = "AFINN lexicon (baseline)"
TL_LEXICON_NAME = "AFINN + Tagalog lexicon (baseline)"
MAJORITY_NAME = "Majority class (baseline)"
GROUP_NOTES = {
    PROPOSED: "",
    EXTRA: "extra experiment (outside the proposed 24-configuration grid)",
    BASELINE: "baseline",
}
CONFIG_NOTES = {
    AFINN_NAME: "English-only lexicon; Tagalog/Taglish words are unknown to it.",
    TL_LEXICON_NAME: ("AFINN plus the unedited Chen & Skiena (2014) Tagalog lexicon "
                      "(+1/-1 per word); machine-built and noisy (e.g. 'pagkain' = positive)."),
}


def build_pipeline(vec_name, ngram, model_name):
    """Proposed grid. The vectorizer sits inside the Pipeline, so it is re-fit on
    each training fold (no vocabulary/IDF leakage from the validation fold)."""
    vectorizer = VECTORIZERS[vec_name](preprocessor=normalize, tokenizer=tokenize,
                                       token_pattern=None, lowercase=False, ngram_range=ngram)
    return Pipeline([("vec", vectorizer), ("clf", GRID_MODELS[model_name]())])


def build_word_char_pipeline(model_name):
    """Extra experiment: word n-grams and character n-grams side by side."""
    features = FeatureUnion([
        ("word", TfidfVectorizer(preprocessor=normalize, tokenizer=tokenize, token_pattern=None,
                                 lowercase=False, ngram_range=WORD_NGRAM, min_df=WORD_MIN_DF,
                                 sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", preprocessor=normalize,
                                 ngram_range=CHAR_NGRAM, min_df=CHAR_MIN_DF, sublinear_tf=True)),
    ])
    return Pipeline([("features", features), ("clf", WORD_CHAR_MODELS[model_name]())])


def read_word_list(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    return {word.strip().lower() for word in lines if word.strip()}


class LexiconClassifier:
    """Sum of word scores: > 0 Positive, < 0 Negative, 0 Neutral. Nothing to train.

    AFINN scores English words. With tagalog=True each word in the Tagalog
    positive/negative list adds +1/-1, unless AFINN already scores it (the
    Tagalog list contains some English words, e.g. "top", "right")."""

    _WORD_RE = re.compile(r"[a-zñ]+")

    def __init__(self, tagalog=False):
        self.afinn = Afinn(language="en", emoticons=False)
        self.afinn_words = set(self.afinn._dict)  # afinn has no public word list
        self.tl_pos = read_word_list(TL_POSITIVE_FILE) if tagalog else set()
        self.tl_neg = read_word_list(TL_NEGATIVE_FILE) if tagalog else set()

    def _tagalog_scores(self, norm):
        words = [w for w in self._WORD_RE.findall(norm) if w not in self.afinn_words]
        return [1 for w in words if w in self.tl_pos] + [-1 for w in words if w in self.tl_neg]

    def score(self, text):
        norm = normalize(text)
        return self.afinn.score(norm) + sum(self._tagalog_scores(norm))

    def fit(self, X, y):  # noqa: ARG002 - same interface as the sklearn models
        return self

    def predict(self, texts):
        return np.array([POSITIVE if s > 0 else NEGATIVE if s < 0 else NEUTRAL
                         for s in map(self.score, texts)])

    def n_without_known_words(self, texts):
        norms = map(normalize, texts)
        return sum(1 for n in norms if not self.afinn.find_all(n) and not self._tagalog_scores(n))


def candidates():
    """config name -> (group, row metadata, model factory)."""
    out = {}
    for vec_name in VECTORIZERS:
        for ngram in NGRAM_RANGES:
            for model_name in GRID_MODELS:
                meta = dict(vectorizer=vec_name, ngram_range=str(ngram), model=model_name)
                out[f"{vec_name} {ngram} + {model_name}"] = (
                    PROPOSED, meta, partial(build_pipeline, vec_name, ngram, model_name))
    for model_name in WORD_CHAR_MODELS:
        meta = dict(vectorizer=WORD_CHAR, ngram_range=f"word {WORD_NGRAM} + char_wb {CHAR_NGRAM}",
                    model=model_name)
        out[f"{WORD_CHAR} + {model_name}"] = (
            EXTRA, meta, partial(build_word_char_pipeline, model_name))
    out[MAJORITY_NAME] = (
        BASELINE, dict(vectorizer="none", ngram_range="", model="most_frequent"),
        partial(DummyClassifier, strategy="most_frequent"))
    out[AFINN_NAME] = (
        BASELINE, dict(vectorizer="AFINN lexicon", ngram_range="(1, 1)", model="word-score sum"),
        LexiconClassifier)
    out[TL_LEXICON_NAME] = (
        BASELINE,
        dict(vectorizer="AFINN + Tagalog lexicon", ngram_range="(1, 1)", model="word-score sum"),
        partial(LexiconClassifier, tagalog=True))
    return out


def fold_scores(y_true, y_pred):
    macro = {"labels": LABELS, "average": "macro", "zero_division": 0}
    per_class = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    return {
        "macro_f1": f1_score(y_true, y_pred, **macro),
        "macro_precision": precision_score(y_true, y_pred, **macro),
        "macro_recall": recall_score(y_true, y_pred, **macro),
        "accuracy": accuracy_score(y_true, y_pred),
        **{f"f1_{lab.lower()}": s for lab, s in zip(LABELS, per_class, strict=True)},
    }


def step4_experiments(train):
    X = train["text_clean"].astype(str).to_numpy()
    y = train["sentiment"].to_numpy()
    groups = train["text_key"].fillna("").astype(str).to_numpy()
    splitter = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(splitter.split(X, y, groups))   # identical texts stay in one fold
    configs = candidates()
    rule(f"STEP 4 - EXPERIMENTS ({len(configs)} configurations, {N_FOLDS}-fold text-grouped CV, "
         f"{len(X):,} training reviews)")

    rows = []
    for i, (name, (group, meta, make)) in enumerate(configs.items(), 1):
        start, fold_rows = time.perf_counter(), []
        for train_idx, val_idx in folds:
            model = make().fit(X[train_idx], y[train_idx])  # fresh, unfitted per fold
            fold_rows.append(fold_scores(y[val_idx], model.predict(X[val_idx])))
        scores = pd.DataFrame(fold_rows)
        row = {"config": name, "group": group, **meta,
               "note": CONFIG_NOTES.get(name, GROUP_NOTES[group])}
        for col in scores.columns:
            row[f"{col}_mean"], row[f"{col}_std"] = scores[col].mean(), scores[col].std(ddof=1)
        row["fit_seconds"] = round(time.perf_counter() - start, 2)
        rows.append(row)
        print(f"  [{i:2d}/{len(configs)}] {name:<52} macro-F1 "
              f"{row['macro_f1_mean']:.3f} ± {row['macro_f1_std']:.3f}")

    cv = pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)
    cv.insert(0, "rank", range(1, len(cv) + 1))
    cv["in_proposed_grid"] = cv["group"] == PROPOSED
    best = cv[cv["in_proposed_grid"]].iloc[0]   # final model: best of the proposed 24
    cv["is_best"] = cv["config"] == best["config"]
    cv.to_csv(CV_RESULTS_CSV, index=False)

    model = configs[best["config"]][2]().fit(X, y)   # refit on all training data
    joblib.dump(model, BEST_MODEL_PATH)
    lex_stats = {}
    for name, tagalog in ((AFINN_NAME, False), (TL_LEXICON_NAME, True)):
        lex = LexiconClassifier(tagalog=tagalog)
        lex_stats[name] = {"zero_score": int(sum(lex.score(t) == 0 for t in X)),
                           "no_known_words": lex.n_without_known_words(X), "n": len(X)}
    BEST_CONFIG_JSON.write_text(json.dumps({
        "config": best["config"],
        "vectorizer": best["vectorizer"],
        "ngram_range": best["ngram_range"],
        "model": best["model"],
        "selection": "best macro-F1 within the proposed 24-configuration grid",
        "cv_macro_f1_mean": float(best["macro_f1_mean"]),
        "cv_macro_f1_std": float(best["macro_f1_std"]),
        "cv_macro_precision_mean": float(best["macro_precision_mean"]),
        "cv_macro_recall_mean": float(best["macro_recall_mean"]),
        "cv_accuracy_mean": float(best["accuracy_mean"]),
        "lexicon_baselines": lex_stats,
        "n_train": len(X),
        "cv_folds": N_FOLDS,
        "seed": SEED,
    }, indent=2), encoding="utf-8")

    table = cv.rename(columns={"macro_precision_mean": "prec", "macro_recall_mean": "rec",
                               "accuracy_mean": "acc", "f1_negative_mean": "F1 neg",
                               "f1_neutral_mean": "F1 neu", "f1_positive_mean": "F1 pos"})
    table["macro-F1"] = [f"{m:.3f} ± {s:.3f}"
                         for m, s in zip(cv["macro_f1_mean"], cv["macro_f1_std"], strict=True)]
    table["config"] = [c + ("  <-- selected" if b else "  (extra)" if g == EXTRA else "")
                       for c, b, g in zip(cv["config"], cv["is_best"], cv["group"], strict=True)]
    columns = ["rank", "config", "macro-F1", "prec", "rec", "acc", "F1 neg", "F1 neu", "F1 pos"]
    print("\n" + table[columns].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n>> Selected (proposed grid): {best['config']}  CV macro-F1 "
          f"{best['macro_f1_mean']:.3f} ± {best['macro_f1_std']:.3f}, precision "
          f"{best['macro_precision_mean']:.3f}, recall {best['macro_recall_mean']:.3f}, "
          f"accuracy {pct(best['accuracy_mean'])}")
    for name, s in lex_stats.items():
        print(f"   {name}: {s['zero_score']:,} of {s['n']:,} reviews scored 0 (-> Neutral); "
              f"{s['no_known_words']:,} contain no known word")
    return model, best


# =============================================================================
# STEP 5 - HELD-OUT TEST EVALUATION (scored once)
# =============================================================================
def step5_evaluate(model, best, train, test):
    rule(f"STEP 5 - HELD-OUT TEST SET ({len(test):,} reviews)")
    X_test, y_test = test["text_clean"].astype(str).to_numpy(), test["sentiment"].to_numpy()
    pred = model.predict(X_test)
    correct = pred == y_test
    acc = accuracy_score(y_test, pred)
    macro_f1 = f1_score(y_test, pred, labels=LABELS, average="macro")
    report = classification_report(y_test, pred, labels=LABELS, output_dict=True, zero_division=0)
    cm = labelled_confusion(y_test, pred, "actual", "pred")
    majority_label = train["sentiment"].value_counts().idxmax()
    majority_acc = float((y_test == majority_label).mean())
    short = test["is_short"].astype(bool).to_numpy()
    short_acc, long_acc = float(correct[short].mean()), float(correct[~short].mean())

    pred_df = test[["uuid", "Branch", "City", "review_date", "rating", "sentiment", "text_raw",
                    "text_clean", "word_count", "is_short", "rider_blamed",
                    "branch_usable"]].copy()
    pred_df["predicted"], pred_df["correct"] = pred, correct
    pred_df.to_csv(PREDICTIONS_CSV, index=False)
    TEST_METRICS_JSON.write_text(json.dumps({
        "config": best["config"],
        "n_test": len(test),
        "test_accuracy": float(acc),
        "test_macro_f1": float(macro_f1),
        "test_report": report,
        "confusion_matrix": {"labels": LABELS, "matrix": cm.to_numpy().tolist()},
        "majority_baseline": {"label": majority_label, "accuracy": majority_acc},
        "short_accuracy": short_acc,
        "long_accuracy": long_acc,
    }, indent=2), encoding="utf-8")

    print(f"Model: {best['config']}")
    print(f"Test accuracy {pct(acc)} | test macro-F1 {macro_f1:.3f} | majority baseline "
          f"(always '{majority_label}') {pct(majority_acc)}\n")
    print(classification_report(y_test, pred, labels=LABELS, digits=3, zero_division=0))
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(cm.to_string())
    print(f"\nShort reviews (<{SHORT_REVIEW_WORDS} words): {pct(short_acc)} accuracy on "
          f"{short.sum():,} | longer reviews: {pct(long_acc)} on {(~short).sum():,}")


# =============================================================================
# STEP 6 - COMPLAINT EXTRACTION (negative, branch-attributable reviews)
# Level 1: chain-wide and per-branch n-gram term frequency.
# Level 2: regex theme tagging; per-branch rate and lift vs the chain average.
# =============================================================================
# A term is hidden only if EVERY word in it is generic ("late delivery" stays).
DISPLAY_STOPWORDS = {
    # brand and platform names
    "foodpanda", "jabee", "jollibe", "jollibee", "jolibee", "panda",
    # generic ordering vocabulary
    "branch", "deliver", "delivered", "delivery", "food", "foods", "item",
    "items", "meal", "meals", "order", "ordered", "ordering", "orders",
    "restaurant", "store", "time", "times",
    # fillers (English and Tagalog)
    "ave", "did", "does", "get", "got", "just", "like", "nag", "niyo", "nyo",
    "ok", "okay", "please", "pls", "plss", "po", "really", "sana", "sobrang",
    "super", "ung",
} | NEGATIONS

ITEMS = r"""(gravy|gravi|rice|kanin|kan-?on|drinks?|fries|spaghetti|spag|sauce|ketchup
    |catsup|utensils?|cutler(y|ies)|spoons?|forks?|tissues?|straws?|ice|ice\s*cream
    |sundaes?|coke|float|toy|burgers?|yumburger|burger\s*steak|chicken|chiken|nuggets
    |dessert|cheese|patty|pie|soup|macaroni|condiments?|hotdog|pancake|iced\s+tea)"""

THEMES = {
    "MISSING_ITEM": r"""
        \bmissing\b | \bkulang\b | \bincomplete\b | \black(ing)?\b
        | \bforg[eo]t | \bnakalimutan | \bnaiwan
        | \b(kulangan|kompletohin|kumpleto)
        | \bwala\b | \bwala\s+(po|man|nga)\b
        | \b(no|walang?|wla)\s+
          (po(ng)?\s+|yung\s+|ung\s+|man\s+lang\s+|pang?\s+|na\s+)?""" + ITEMS + r"""
        | \b(doesn'?t|don'?t|didn'?t)\s+have\s+(any\s+|my\s+)?""" + ITEMS + r"""
        | \bonly\s+(\d+|one|two|three)\b | \bonly\s+(got|received|delivered|arrived|came)
        | \b(\d+|isa)\s*(pcs?\s+)?lang\b
        | \b(weren'?t|wasn'?t|not)\s+(been\s+)?includ | \bno\s+cutlery
        | \b(hindi|di|d|wala)\s+(po\s+)?(kasama|kasali|sinama|naisama)
        | \b(wala|hindi|di|d)\b.{0,25}\b(recieve|receive|natanggap)
        | \b(didn'?t|did\s+not|never)\s+(receive|include|give|get|come|arrive)
        | \bnot\s+(included|received|delivered|complete)\b
        | \b(hindi|di|wala)\s+(dumating|nakarating|naibigay|binigay|kasama|sinama|nilagay)
        | \bnever\s+(arrived|came)\b
    """,
    "WRONG_ORDER": r"""
        \bwrong\b | \bmali\b | \bmaling\b | \bnot\s+what\s+i\s+order
        | \binstead\s+of\b | \bdifferent\s+(order|item|flavor|size)
        | \bibang\s+(order|item|flavor) | \bnapalitan\b | \bpinalitan\b
        | \b(didn'?t|did\s+not|don'?t|do\s+not|never|not)\s+follow
        | \b(hindi|di|wala)\s+(sinunod|sinusunod|sinundan)
        | \bignored?\s+(my\s+|the\s+|our\s+)?(request|instruction|note)
        | \brequest(ed)?\b.{0,40}\b(but|pero|yet|instead)\b
        | \binstructions?\b.{0,20}\b(not|ignored|wasn'?t)\b | \bincorrect\b
        | \b(request|requested|requesting|noted?|notes|asked\s+for|specifically|sabi\s+ko)\b
        | \b(thighs?|breast|legs?|drumsticks?|paa|pecho|petcho)\b
        | \biba\s+(ang|yung|ung|pinadala|binigay|na)\b
        | \b(i|we)\s+ordered\b.{0,40}\b(but|pero)\b.{0,30}
          \b(received|got|gave|served|sent|delivered|dumating|binigay|pinadala)
        | \b(you|they)\s+(served|sent|gave)\b
    """,
    "LATE_DELIVERY": r"""
        \blate\b | \bdelay(ed|s)?\b | \btook\s+(so\s+|too\s+|very\s+)?long
        | \b(so|too|very)\s+long\b | \blong\s+(wait|time)\b | \bwait(ed|ing)\b
        | \b(an?|one|two|three|\d+)\s*(hours?|hrs?|oras)\b | \bhours?\b
        | \bslow\b | \bsluggish\b
        | \b(ang\s*|an|napaka|sobrang\s*)?tagal\b | \bmatagal\b | \bnatagalan\b
        | \bmabagal\b | \bbagal\b
        | \bpinaghintay | \bhintay | \b\d+\s*(mins?|minutes)\b | \bminutes\b
        | \bto+\s+lo+ng\b | \btoo\s+much\s+time\b | \btook\s+time\b
        | \blong(er)?\s+(than|prep|preparing)\b | \bestimated\s+time\b
    """,
    "COLD_FOOD": r"""
        \bcold\b | \bmalamig\b | \blamig\b
        | \b(not|hindi|di|wala)\s+(na\s+)?(hot|warm|mainit|init|inet)\b
        | \bluke\s*warm\b | \broom\s+temp
    """,
    "FOOD_QUALITY": r"""
        \bsoggy\b | \bstale\b | \bundercooked\b | \bovercooked\b | \braw\b | \bhilaw\b
        | \bburnt?\b | \bsunog\b | \bnangingitim | \bmatigas\b | \btigas\b | \bhard\b
        | \bdry\b | \bdried\b | \brubbery\b | \boily\b | \bmamantika\b | \bmantika
        | \btasteless\b | \bbland\b | \bwalang\s+lasa | \bmatabang\b | \btabang\b
        | \bsalty\b | \bmaalat\b | \balat\b | \bpanis | \bspoil | \bsira\b | \bsour\b
        | \bmaasim\b | \bsmell | \bmabaho\b | \bbaho\b | \bnot\s+fresh\b | \breheat
        | \btast(e|es|ed|y)\s+(bad|weird|off|old|funny|different|awful|terrible)
        | \b(pangit|panget)\s+(ng\s+|ang\s+)?lasa
        | \b(not|hindi|di)\s+(tasty|delicious|masarap|crispy|crunchy|malutong)
        | \brancid\b | \bwatery\b | \bbulok\b | \bexpired\b | \bweeks?\s+old\b
        | \bdisgusting\b | \bgross\b | \byuck | \bkadiri\b | \bnot\s+cooked
    """,
    "SMALL_PORTION": r"""
        \bsmall(er|est)?\b | \btiny\b | \bliit\b | \bmaliit\b | \bnapakaliit\b
        | \bapakaliit\b | \bang\s*liit\b | \bsobrang\s+(ma)?liit\b | \bkonti\b
        | \bkaunti\b | \bonti\b | \bdiet\s+size\b | \bpayat\b | \bportions?\b
        | \bserving\b | \bskinny\b | \bthin\b | \bquantity\b
        | \b(back|ribs?|wings?|pakpak|likod)\s+part
        | \bpuro\s+(back|pakpak|wings?|likod|ribs?|buto|bones?)
        | \ball\s+(back|ribs?|wings?|bones?)\b | \bpuro\s+buto
    """,
    "STAFF_SERVICE": r"""
        \bstaff\b | \bcrew\b | \bcashier\b | \bmanager\b | \brude\b | \battitude\b
        | \bbastos\b | \bsuplad[ao]\b | \bservice\b | \bserbisyo\b | \bincompetent\b
        | \bunprofessional\b | \bcareless | \bpabaya\b | \bwalang\s+(pakialam|pake)
        | \b(no|not)\s+(response|reply|answer)| \bnot\s+answering\b
        | \bcancel+(ed|ing|led)?\b | \bunavailable\b | \bout\s+of\s+stock\b
        | \bnot\s+available\b | \bnotif(y|ied) | \bcoordinat
    """,
    "PACKAGING": r"""
        \bpackag | \bspill(ed|s|ing)?\b | \bspilt\b | \btapon\b | \bnatapon\b
        | \btumapon\b | \bleak | \btagas\b | \btumagas\b | \bcontainers?\b
        | \bbox(es)?\b | \blids?\b | \bsealed?\b | \bcrushed\b | \bsquish
        | \bnayupi\b | \byupi\b | \bdurog\b | \bsabog\b | \bnagkalat\b | \bmessy\b
        | \bnapisa\b | \bbasa\b | \bwet\b | \bcups?\b | \bbroken\b | \bseep
        | \blalagyan\b | \bkarton\b
    """,
    "HYGIENE": r"""
        \bhairs?\b | \bbuhok\b | \binsects?\b | \bipis\b | \bcockroach | \broach
        | \blanggam\b | \bants?\b | \bflies\b | \bfly\b | \blangaw\b | \bdirty\b
        | \bmadumi\b | \bmarumi\b | \bdumi\b | \bunsanitary\b | \bunhygienic\b
        | \bforeign\s+(object|matter) | \buod\b | \bworms?\b | \bmold | \bamag\b
        | \binsekto | \bbugs?\b
    """,
}
THEME_LABELS = {
    "MISSING_ITEM": "Missing item", "WRONG_ORDER": "Wrong order", "LATE_DELIVERY": "Late delivery",
    "COLD_FOOD": "Cold food", "FOOD_QUALITY": "Food quality", "SMALL_PORTION": "Small portion",
    "STAFF_SERVICE": "Staff & service", "PACKAGING": "Packaging", "HYGIENE": "Hygiene",
}
THEME_RE = {theme: re.compile(pattern, re.VERBOSE) for theme, pattern in THEMES.items()}
TERM_COLUMNS = ["term", "n_words", "n_reviews", "count"]


def theme_text(text):
    """Lowercase and squash elongations to one char ("lateeee" -> "late") so the
    regexes match the dictionary spelling."""
    return _REPEAT_RE.sub(r"\1", str(text).lower())


def is_display_term(term, extra_stop=frozenset()):
    words = term.split()
    if not any(c.isalpha() for c in term):
        return False   # emoji/number-only terms say nothing actionable
    if len(words) > 1 and words[-1] in NEGATIONS:
        return False   # "chicken not" is a stopword-removal artifact
    return not all(w in DISPLAY_STOPWORDS or w in extra_stop for w in words)


def term_counts(texts, min_df, ngram_range, extra_stop=frozenset()):
    """Terms ranked by how many reviews contain them, then by total count."""
    vec = CountVectorizer(preprocessor=normalize, tokenizer=tokenize, token_pattern=None,
                          lowercase=False, ngram_range=ngram_range, min_df=min_df)
    try:
        counts = vec.fit_transform(texts)
    except ValueError:   # no term reaches min_df
        return pd.DataFrame(columns=TERM_COLUMNS)
    df = pd.DataFrame({"term": vec.get_feature_names_out(),
                       "n_reviews": (counts > 0).sum(axis=0).A1,
                       "count": counts.sum(axis=0).A1})
    df["n_words"] = df["term"].str.count(" ") + 1
    df = df[df["term"].map(lambda t: is_display_term(t, extra_stop))]
    return df.sort_values(["n_reviews", "count"], ascending=False)[TERM_COLUMNS]


def branch_name_words(branch):
    """Words of a branch name ("acacia lane"): they show up in complaints but are not issues."""
    return frozenset(tokenize(normalize(branch)))


def step6_complaints(complaints):
    texts = complaints["text_clean"].astype(str)
    n_total = len(complaints)
    rule(f"STEP 6 - COMPLAINT EXTRACTION ({n_total:,} complaints, "
         f"{complaints['Branch'].nunique()} branches)")

    # Level 1 - term frequency, chain-wide and per branch.
    all_branch_words = frozenset().union(*map(branch_name_words, complaints["Branch"].unique()))
    chain = term_counts(texts, CHAIN_MIN_DF, CHAIN_NGRAM, all_branch_words).head(TOP_N_CHAIN)
    chain = chain.reset_index(drop=True)
    chain.insert(0, "rank", range(1, len(chain) + 1))
    chain["pct_reviews"] = chain["n_reviews"] / n_total
    chain.to_csv(TERMS_CHAIN_CSV, index=False)

    branch_terms = []
    for branch, grp in complaints.groupby("Branch"):
        terms = term_counts(grp["text_clean"].astype(str), BRANCH_MIN_DF, BRANCH_NGRAM,
                            branch_name_words(branch)).head(TOP_N_BRANCH)
        terms["Branch"] = branch
        terms["pct_reviews"] = terms["n_reviews"] / len(grp)
        branch_terms.append(terms)
    branch_terms = pd.concat(branch_terms)[["Branch", *TERM_COLUMNS, "pct_reviews"]]
    branch_terms.to_csv(TERMS_BRANCH_CSV, index=False)

    # Level 2 - themes, rate per branch, lift vs the chain.
    complaints = complaints.copy()
    norm = texts.map(theme_text)
    for theme, rx in THEME_RE.items():
        complaints[theme] = norm.map(lambda s, rx=rx: bool(rx.search(s)))
    theme_cols = list(THEMES)
    complaints["n_themes"] = complaints[theme_cols].sum(axis=1)
    chain_rate = complaints[theme_cols].mean()
    by_branch = complaints.groupby("Branch")
    hits, sizes = by_branch[theme_cols].sum(), by_branch.size()
    rates = hits.div(sizes, axis=0)
    lift = rates.div(chain_rate, axis=1)

    long = (rates.stack().rename("rate").to_frame()
            .join(hits.stack().rename("n_hits")).join(lift.stack().rename("lift"))
            .reset_index().rename(columns={"level_1": "theme"}))
    long["theme_label"] = long["theme"].map(THEME_LABELS)
    long["n_complaints"] = long["Branch"].map(sizes)
    long["chain_rate"] = long["theme"].map(chain_rate)
    long["notable"] = (long["lift"] > NOTABLE_LIFT) & (long["n_hits"] >= MIN_THEME_HITS)
    long["City"] = long["Branch"].map(by_branch["City"].first())
    long = long[["Branch", "City", "n_complaints", "theme", "theme_label", "n_hits",
                 "rate", "chain_rate", "lift", "notable"]]
    long.to_csv(THEMES_BRANCH_CSV, index=False)

    priorities = []
    for branch, grp in long.groupby("Branch"):
        eligible = grp[grp["n_hits"] >= MIN_THEME_HITS].sort_values("lift", ascending=False)
        row = {"Branch": branch, "City": grp["City"].iat[0],
               "n_complaints": int(grp["n_complaints"].iat[0]),
               "share_of_chain_complaints": grp["n_complaints"].iat[0] / n_total}
        for i in (1, 2):
            issue = eligible.iloc[i - 1] if len(eligible) >= i else None
            row[f"top_issue_{i}"] = issue["theme_label"] if issue is not None else None
            row[f"top_issue_{i}_rate"] = issue["rate"] if issue is not None else None
            row[f"top_issue_{i}_lift"] = issue["lift"] if issue is not None else None
            row[f"top_issue_{i}_notable"] = bool(issue["notable"]) if issue is not None else False
        priorities.append(row)
    priorities = pd.DataFrame(priorities).sort_values(["top_issue_1_lift", "n_complaints"],
                                                      ascending=False)
    priorities.to_csv(BRANCH_PRIORITIES_CSV, index=False)

    print("Top 15 complaint terms chain-wide:")
    print(chain.head(15)[["rank", "term", "n_reviews"]].to_string(index=False))
    print("\nTheme coverage (share of all complaints):")
    for theme in theme_cols:
        n_hits = int(complaints[theme].sum())
        print(f"  {THEME_LABELS[theme]:<16} {chain_rate[theme]:6.1%}  ({n_hits:,})")
    print(f"  {'(no theme)':<16} {(complaints['n_themes'] == 0).mean():6.1%}")
    print(f"\nBranches with a notable issue (lift > {NOTABLE_LIFT}, >= {MIN_THEME_HITS} "
          f"complaints): {int(priorities['top_issue_1_notable'].sum())} of {len(priorities)}")
    print(priorities.head(10)[["Branch", "n_complaints", "top_issue_1", "top_issue_1_lift",
                               "top_issue_2", "top_issue_2_lift"]]
          .to_string(index=False, float_format=lambda v: f"{v:.2f}"))


# =============================================================================
# RUN
# =============================================================================
def main():
    raw = resolve_raw_dir()
    restos = pd.read_csv(raw / RESTOS_FILE, low_memory=False)
    reviews = pd.read_csv(raw / REVIEWS_FILE, low_memory=False)
    legacy = pd.read_csv(raw / LEGACY_RESTOS_FILE, low_memory=False)

    step1_inspect(restos, reviews, legacy)
    df, complaints = step2_build(restos, reviews)
    step2b_label_check(df)
    train, test = step3_split(df)
    model, best = step4_experiments(train)
    step5_evaluate(model, best, train, test)
    step6_complaints(complaints)
    rule(f"DONE - outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
