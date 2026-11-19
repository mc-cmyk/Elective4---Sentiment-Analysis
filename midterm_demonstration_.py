"""
ReviewLens - Midterm System Progress Demonstration  (ONE self-contained file)

This single file runs the whole analytical pipeline from the raw Kaggle data:

    PART 1  Load and inspect the raw data                 (PDF Part 2)
    PART 2  Prepare the dataset (filter, label, clean)    (PDF Part 2)
            + manual check of the star-rating labels
    PART 3  Text preprocessing                            (PDF Part 2)
    PART 4  Train / test split                            (PDF Part 3)
    PART 5  Experiments: 5-fold CV of all models          (PDF Part 3)
    PART 6  Final model + held-out test evaluation        (PDF Part 4)
    PART 7  Complaint analysis                            (PDF Part 4)
    PART 8  Limitations                                   (Q&A)

It follows the same methods and settings as sentiment_pipeline.py (scope: English
and Tagalog/Taglish; Bisaya reviews are dropped), so the results match it. It only
writes the experiment cache to output/midterm_demo/. The cache is reused only if
this file and the training data are unchanged since it was saved.

Run:   python midterm_demonstration_.py
       --pause     wait for Enter between parts (for presenting)
       --rerun-cv  re-run the experiments even if valid cached results exist
Needs: pandas numpy scikit-learn afinn  (+ kagglehub if no data/raw)
"""
import hashlib
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, cohen_kappa_score, confusion_matrix,
                             f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")
pd.set_option("display.width", 160)

# =============================================================================
# SETTINGS  (same values as the project's config.py)
# =============================================================================
PROJECT_DIR = Path(__file__).resolve().parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
OUT_DIR = PROJECT_DIR / "output" / "midterm_demo"
OUT_DIR.mkdir(parents=True, exist_ok=True)
KAGGLE_DATASET = "bwandowando/food-panda-restaurant-reviews"

BRAND = "Jollibee"
SEED = 42
TEST_FRACTION = 0.20
N_FOLDS = 5
LABELS = ["Negative", "Neutral", "Positive"]
SHORT_REVIEW_WORDS = 3          # fewer words -> flagged as short (kept)
SPAM_MIN_WORDS = 4              # identical texts this long -> copy-paste spam
MIN_REVIEWS_PER_BRANCH = 30     # smaller branches flagged, not deleted
RIDER_BLAME_MAX_STAR = 2        # rider score <= 2 -> complaint blamed on the rider
NGRAM_RANGES = [(1, 1), (2, 2), (3, 3), (1, 3)]
NOTABLE_LIFT, MIN_THEME_HITS = 1.3, 3

# Bisaya-only words (alphabetical): a review containing one is out of scope
# (English/Tagalog/Taglish only). Words shared with Tagalog are left out.
BISAYA_MARKERS = [
    "bitaw", "bugnaw", "daghan", "didto", "dili", "diri", "dugay", "gamay",
    "ganahan", "gyud", "imong", "jud", "kaau", "kaayo", "kaayong", "kaayu",
    "karon", "kuwang", "lami", "lamia", "naa", "ngano", "nganong", "nimo",
    "nindot", "pud", "unsa", "unya", "walay",
]

# Tagalog sentiment lexicon (Chen & Skiena 2014, Kaggle "Sentiment Lexicons for 81 Languages").
LEXICON_DIR = PROJECT_DIR / "81_lang_lexicon" / "sentiment-lexicons"
TL_POSITIVE_FILE = LEXICON_DIR / "positive_words_tl.txt"
TL_NEGATIVE_FILE = LEXICON_DIR / "negative_words_tl.txt"

# Manual label check sheet written by sentiment_pipeline.py (read only here).
LABEL_SHEET = PROJECT_DIR / "output" / "02_dataset" / "label_check_sample.csv"


# =============================================================================
# Helper for presenting
# =============================================================================
def part(title):
    print(f"\n\n{'=' * 80}\n{title.upper()}\n{'=' * 80}")
    if "--pause" in sys.argv:
        input("[Enter] ")


# #############################################################################
# PART 1 - LOAD AND INSPECT THE RAW DATA
# #############################################################################
part("Part 1 - Load and inspect the raw data")

raw_dir = RAW_DIR
if not (RAW_DIR / "ph_reviews_2025.csv").exists():
    import kagglehub                                   # downloads once, then cached
    raw_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))
reviews = pd.read_csv(raw_dir / "ph_reviews_2025.csv", low_memory=False)
restos = pd.read_csv(raw_dir / "ph_restos_2025.csv", low_memory=False)

stars = pd.to_numeric(reviews["overall"], errors="coerce")
no_text = reviews["text"].isna() | (reviews["text"].astype(str).str.strip() == "")
print(f"  Source           : Kaggle {KAGGLE_DATASET}")
print(f"  Reviews          : {len(reviews):,} rows x {reviews.shape[1]} columns")
print(f"  Stores           : {len(restos):,} rows ({restos['StoreId'].nunique():,} unique StoreId)")
print(f"  Missing text     : {no_text.sum():,} ({no_text.mean():.1%})")
print(f"  Missing stars    : {stars.isna().sum():,}")
print(f"  Duplicated uuid  : {reviews.duplicated('uuid').sum():,}")
print(f"  Shared StoreId   : {restos.duplicated('StoreId', keep=False).sum():,} store rows")


def to_sentiment(s):
    """Target variable: 1-2 stars Negative, 3 Neutral, 4-5 Positive."""
    if pd.isna(s):
        return None
    return "Negative" if s <= 2 else "Neutral" if s == 3 else "Positive"


raw_share = stars.map(to_sentiment).value_counts(normalize=True).reindex(LABELS)
print("\n  Sentiment share (all brands): " + ", ".join(f"{k} {v:.1%}" for k, v in raw_share.items()))
print(f"  >> Always guessing '{raw_share.idxmax()}' gives {raw_share.max():.1%} accuracy,")
print("     so accuracy is misleading. We use macro-F1 (all three classes count equally).")



# #############################################################################
# PART 2 - PREPARE THE DATASET
# #############################################################################
part("Part 2 - Data preparation")


def clean_text(s):
    """Remove invisible Unicode characters and extra spaces (keeps case and emoji)."""
    if pd.isna(s):
        return ""
    s = re.sub(r"[\u200b\u200c\u200d\u200e\u200f\u2060\ufeff\u00ad]", "", str(s))   # zero-width chars, BOM, soft hyphen
    return re.sub(r"\s+", " ", s).strip()


steps = {"All reviews": len(reviews)}

# 1. Keep only Jollibee stores (store name starts with the brand). Duplicate StoreIds dropped.
restos["CompleteStoreName"] = restos["CompleteStoreName"].map(clean_text)
restos = restos.drop_duplicates(subset="StoreId", keep="first")
stores = restos[restos["CompleteStoreName"].str.match(rf"^\s*{BRAND}\b", case=False, na=False)].copy()
stores["Branch"] = stores["CompleteStoreName"].str.replace(rf"^\s*{BRAND}\s*[-–—]?\s*", "",
                                                           regex=True, flags=re.I).str.strip()
stores.loc[stores["Branch"] == "", "Branch"] = stores["CompleteStoreName"]

# 2. Join reviews to those stores, remove duplicated reviews.
df = reviews.merge(stores[["StoreId", "Branch", "City"]], on="StoreId", how="inner")
steps[f"Joined to a {BRAND} store"] = len(df)
df = df.drop_duplicates(subset="uuid", keep="first")
steps["Duplicate reviews removed"] = len(df)

# 3. Target variable from the star rating (reviews without stars cannot be labelled).
df["rating"] = pd.to_numeric(df["overall"], errors="coerce")
df = df[df["rating"].notna()].copy()
df["sentiment"] = df["rating"].map(to_sentiment)
steps["Has a star rating"] = len(df)

# 4. Flags (kept, not deleted): rider-blamed complaints and very short reviews.
df["rider_blamed"] = pd.to_numeric(df["rider"], errors="coerce") <= RIDER_BLAME_MAX_STAR
df["text_clean"] = df["text"].map(clean_text)
df["word_count"] = df["text_clean"].str.split().str.len()
df["is_short"] = df["word_count"] < SHORT_REVIEW_WORDS
df = df[df["text_clean"].str.len() > 0].copy()
steps["Has review text"] = len(df)
before_cleaning = df.copy()

# 5. Language scope: drop reviews containing a Bisaya-only word.
is_bisaya = df["text_clean"].str.contains(r"\b(?:" + "|".join(BISAYA_MARKERS) + r")\b", case=False)
df = df[~is_bisaya].copy()
steps["Bisaya reviews removed"] = len(df)

# 6. Copy-paste spam: identical long texts (>= 4 words). Short repeats like "good" are real opinions.
# (normalize() is defined in Part 3; it lowercases, removes URLs and squashes "soooo" -> "soo".)
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_REPEAT_RE = re.compile(r"(.)\1{2,}")


def normalize(text):
    text = str(text).lower()
    text = _URL_RE.sub(" ", text)
    return _REPEAT_RE.sub(r"\1\1", text)


df["text_key"] = df["text_clean"].map(normalize).str.strip()
spam = (df["word_count"] >= SPAM_MIN_WORDS) & df.duplicated("text_key", keep="first")
df = df[~spam].copy()
steps[f"Spam removed (repeated, >= {SPAM_MIN_WORDS} words)"] = len(df)

# 7. Branches with too few reviews are flagged (their complaint rates would be unreliable).
store_counts = df.groupby(["StoreId", "Branch", "City"]).size()
usable_stores = store_counts[store_counts >= MIN_REVIEWS_PER_BRANCH].index.get_level_values("StoreId")
df["branch_usable"] = df["StoreId"].isin(usable_stores)
df = df.reset_index(drop=True)

print(f"  {BRAND} stores found: {len(stores)}\n")
for step, n in steps.items():
    print(f"    {step:<42} {n:>8,}")
share = df["sentiment"].value_counts().reindex(LABELS)
print(f"\n  Final dataset: {len(df):,} reviews | "
      + " | ".join(f"{k} {v:,} ({v / len(df):.1%})" for k, v in share.items()))
print(f"  Short reviews (kept, flagged): {df['is_short'].sum():,} | rider-blamed: {df['rider_blamed'].sum():,}")
compare = pd.DataFrame({"Before cleaning": before_cleaning["sentiment"].value_counts(normalize=True),
                        "After cleaning": df["sentiment"].value_counts(normalize=True)}).reindex(LABELS)
print(compare.round(3).to_string())
print(f"  Branches: {df['StoreId'].nunique()} ({df.loc[df.branch_usable, 'StoreId'].nunique()} "
      f"with >= {MIN_REVIEWS_PER_BRANCH} reviews)")

# Manual check of the star-rating labels: 99 reviews (33 per class) labelled by hand from
# the text only. The sheet is created by sentiment_pipeline.py; here it is only read.
print("\n  Manual label check (star label vs. label from reading the text):")
if not LABEL_SHEET.exists():
    print("    Sheet not created yet - run sentiment_pipeline.py once to create it.")
else:
    sheet = pd.read_csv(LABEL_SHEET, encoding="utf-8-sig")
    manual = sheet["manual_label"].fillna("").astype(str).str.strip().str.capitalize()
    done = manual.isin(LABELS)
    if not done.all():
        print(f"    {done.sum()} of {len(sheet)} reviews labelled so far - finish the sheet to get the result.")
    else:
        star = sheet["star_label"]
        print(f"    {len(sheet)} reviews | agreement {(star == manual).mean():.1%} | "
              f"Cohen's kappa {cohen_kappa_score(star, manual, labels=LABELS):.2f} "
              "(> 0.60 = substantial)")



# #############################################################################
# PART 3 - TEXT PREPROCESSING
# #############################################################################
part("Part 3 - Text preprocessing")

# Negation words are NEVER removed: "not good", "hindi masarap" flip the meaning.
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

_APOS_RE = re.compile(r"['’`]")                                   # didn't -> didnt
_EMOJI_MODIFIERS_RE = re.compile(r"[\U0001F3FB-\U0001F3FF\ufe0e\ufe0f\u200d]")   # skin tones etc.
_EMOJI = (
    r"\U0001F300-\U0001F5FF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF"
    r"\U0001F900-\U0001F9FF\U0001FA70-\U0001FAFF\u2600-\u27bf\u2b50\u2b55"
)
_TOKEN_RE = re.compile(rf"[a-z0-9ñ]+|[{_EMOJI}]")


def tokenize(text):
    """Split into words and emoji, drop stopwords (but keep negations). No stemming."""
    text = _APOS_RE.sub("", text)
    text = _EMOJI_MODIFIERS_RE.sub("", text)
    return [t for t in _TOKEN_RE.findall(text) if t not in STOPWORDS]


print("  normalize(): lowercase, remove URLs, 'soooo' -> 'soo'")
print(f"  tokenize() : words + emoji, remove {len(STOPWORDS)} English/Tagalog stopwords,")
print(f"               but KEEP {len(NEGATIONS)} negation words")
print("  Not done on purpose: stemming (breaks Tagalog words), scaling (word counts and")
print("  TF-IDF need none), outlier removal (not meaningful for text).\n")
for text in df[df["word_count"].between(5, 20)].sample(3, random_state=SEED)["text_clean"]:
    print(f"  {text[:90]}\n     -> {tokenize(normalize(text))}")

tokens = df["text_clean"].map(lambda t: tokenize(normalize(t)))
has_negation = tokens.map(lambda ts: any(t in NEGATIONS for t in ts))
print(f"\n  Reviews where a negation word was kept: {has_negation.sum():,} ({has_negation.mean():.1%})")



# #############################################################################
# PART 4 - TRAIN / TEST SPLIT
# #############################################################################
part("Part 4 - Train / test split")

# Stratified: every class keeps the same share in train and test.
# Grouped by text_key: identical texts (e.g. many "good") stay on ONE side, so the
# model is never tested on a review it saw word-for-word in training.
splitter = StratifiedGroupKFold(n_splits=round(1 / TEST_FRACTION), shuffle=True, random_state=SEED)
train_idx, test_idx = next(splitter.split(df, df["sentiment"], groups=df["text_key"]))
train, test = df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
shared = set(train["text_key"]) & set(test["text_key"])

print(f"  Train {len(train):,} | Test {len(test):,} | identical texts in both: {len(shared)}")
balance = pd.DataFrame({"Train": train["sentiment"].value_counts(normalize=True),
                        "Test": test["sentiment"].value_counts(normalize=True)}).reindex(LABELS)
print(balance.round(3).to_string())
print("  >> The test set is put aside now and used only once, in Part 6.")



# #############################################################################
# PART 5 - EXPERIMENTS (5-fold cross-validation on the TRAINING set only)
# #############################################################################
part("Part 5 - Experiments: 24-configuration grid + extras + baselines")

MODELS = {   # class_weight="balanced" gives the rare Neutral/Negative classes more weight
    "MultinomialNB": lambda: MultinomialNB(alpha=1.0),
    "LogisticRegression": lambda: LogisticRegression(class_weight="balanced", max_iter=2000,
                                                     C=1.0, random_state=SEED),
    "LinearSVC": lambda: LinearSVC(class_weight="balanced", C=1.0, random_state=SEED),
}
WORD_CHAR_MODELS = {   # extra experiment (not part of the proposed grid)
    "MultinomialNB": lambda: MultinomialNB(alpha=0.5),
    "LogisticRegression": lambda: LogisticRegression(max_iter=2000, class_weight="balanced",
                                                     C=4.0, random_state=SEED),
    "LinearSVC": lambda: LinearSVC(class_weight="balanced", C=0.5, random_state=SEED),
}


def make_pipeline(vectorizer, ngram, model):
    """Vectorizer + classifier in one Pipeline: the vocabulary is learned from the
    training fold only, so no information leaks from the validation fold."""
    vec_class = CountVectorizer if vectorizer == "CountVectorizer" else TfidfVectorizer
    vec = vec_class(preprocessor=normalize, tokenizer=tokenize, token_pattern=None,
                    lowercase=False, ngram_range=ngram)
    return Pipeline([("vec", vec), ("clf", MODELS[model]())])


def make_word_char_pipeline(model):
    words = TfidfVectorizer(preprocessor=normalize, tokenizer=tokenize, token_pattern=None,
                            lowercase=False, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    chars = TfidfVectorizer(analyzer="char_wb", preprocessor=normalize, ngram_range=(3, 5),
                            min_df=3, sublinear_tf=True)
    return Pipeline([("features", FeatureUnion([("word", words), ("char", chars)])),
                     ("clf", WORD_CHAR_MODELS[model]())])


# All candidates: name -> (group, factory)
candidates = {}
for vectorizer in ["CountVectorizer", "TfidfVectorizer"]:
    for ngram in NGRAM_RANGES:
        for model in MODELS:
            candidates[f"{vectorizer} {ngram} + {model}"] = (
                "Proposed grid", lambda v=vectorizer, n=ngram, m=model: make_pipeline(v, n, m))
for model in WORD_CHAR_MODELS:
    candidates[f"Word+Char TF-IDF + {model}"] = ("Extra (word+char)",
                                                 lambda m=model: make_word_char_pipeline(m))
candidates["Majority class (baseline)"] = ("Baseline",
                                           lambda: DummyClassifier(strategy="most_frequent"))


def read_word_list(path):
    return {w.strip().lower() for w in path.read_text(encoding="utf-8").splitlines() if w.strip()}


class LexiconBaseline:
    """Lexicon baseline: sum of word scores (>0 Positive, <0 Negative, 0 Neutral). Nothing to learn.

    AFINN scores English words. With tagalog=True, each word in the Tagalog positive/negative
    list adds +1/-1, unless AFINN already scores it (the list contains some English words)."""

    _WORD_RE = re.compile(r"[a-zñ]+")

    def __init__(self, tagalog=False):
        from afinn import Afinn
        self.afinn = Afinn(language="en", emoticons=False)
        self.afinn_words = set(self.afinn._dict)       # afinn has no public word list
        self.tl_pos = read_word_list(TL_POSITIVE_FILE) if tagalog else set()
        self.tl_neg = read_word_list(TL_NEGATIVE_FILE) if tagalog else set()

    def score(self, text):
        norm = normalize(text)
        words = [w for w in self._WORD_RE.findall(norm) if w not in self.afinn_words]
        tagalog = sum(w in self.tl_pos for w in words) - sum(w in self.tl_neg for w in words)
        return self.afinn.score(norm) + tagalog

    def fit(self, X, y):
        return self                                  # nothing to learn

    def predict(self, X):
        scores = [self.score(t) for t in X]
        return np.array(["Positive" if s > 0 else "Negative" if s < 0 else "Neutral" for s in scores])


candidates["AFINN lexicon (baseline)"] = ("Baseline", LexiconBaseline)
candidates["AFINN + Tagalog lexicon (baseline)"] = ("Baseline",
                                                    lambda: LexiconBaseline(tagalog=True))


def scores(y_true, y_pred):
    per_class = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    return {"macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
            "precision": precision_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
            "recall": recall_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
            "accuracy": accuracy_score(y_true, y_pred),
            **{f"f1_{lab.lower()}": s for lab, s in zip(LABELS, per_class)}}


X_train = train["text_clean"].to_numpy()
y_train = train["sentiment"].to_numpy()
cache = OUT_DIR / "cv_results.csv"
cache_key_file = OUT_DIR / "cv_results.key"
# The cache is only valid for this exact code and training data: any edit to this file
# or change in the data gives a new key, and the experiments run again.
cache_key = hashlib.sha256("\n".join([Path(__file__).read_text(encoding="utf-8"),
                                      *train["text_key"], *train["sentiment"]])
                           .encode("utf-8")).hexdigest()
cache_valid = (cache.exists() and cache_key_file.exists()
               and cache_key_file.read_text().strip() == cache_key)

if cache_valid and "--rerun-cv" not in sys.argv:
    cv = pd.read_csv(cache)
    print(f"  Loaded saved results from {cache.name} - code and data unchanged since they were "
          "saved (use --rerun-cv to run the experiments again)")
else:
    if cache.exists():
        print("  Saved results are out of date (code or data changed) - running the experiments.")
    # Same 5 folds for every candidate; identical texts stay in the same fold.
    folds = list(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED)
                 .split(X_train, y_train, groups=train["text_key"]))
    rows = []
    for i, (name, (group, factory)) in enumerate(candidates.items(), 1):
        start, fold_scores = time.perf_counter(), []
        for tr, va in folds:
            model = factory().fit(X_train[tr], y_train[tr])     # fresh model every fold
            fold_scores.append(scores(y_train[va], model.predict(X_train[va])))
        fold_scores = pd.DataFrame(fold_scores)
        rows.append({"config": name, "group": group,
                     **{f"{c}_mean": fold_scores[c].mean() for c in fold_scores},
                     "macro_f1_std": fold_scores["macro_f1"].std(),
                     "seconds": round(time.perf_counter() - start, 1)})
        print(f"  [{i:2d}/{len(candidates)}] {name:<50} macro-F1 {rows[-1]['macro_f1_mean']:.3f}")
    cv = pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)
    cv.to_csv(cache, index=False)
    cache_key_file.write_text(cache_key)

# The final model is the best macro-F1 inside the PROPOSED grid only.
best_name = cv[cv["group"] == "Proposed grid"].iloc[0]["config"]
cv["selected"] = cv["config"] == best_name
cols = ["config", "group", "macro_f1_mean", "macro_f1_std", "accuracy_mean",
        "f1_negative_mean", "f1_neutral_mean", "f1_positive_mean"]
print(f"\n  Top 10 of {len(cv)} candidates (5-fold CV on training data):")
print(cv.head(10)[cols].to_string(index=False, float_format="{:.3f}".format))
print("\n  Baselines vs selected model:")
print(cv[(cv.group == "Baseline") | cv.selected][cols].to_string(index=False, float_format="{:.3f}".format))
print(f"\n  >> Selected: {best_name}")

grid = cv[cv["group"] == "Proposed grid"].copy()
grid[["features", "model"]] = grid["config"].str.split(" \\+ ", expand=True)
print("\n  Proposed grid, CV macro-F1 (features x model):")
print(grid.pivot(index="features", columns="model", values="macro_f1_mean").round(3).to_string())


# #############################################################################
# PART 6 - FINAL MODEL AND HELD-OUT TEST EVALUATION
# #############################################################################
part("Part 6 - Final model and test-set evaluation")

final_model = candidates[best_name][1]().fit(X_train, y_train)      # train on ALL training data
vec, clf = final_model.named_steps["vec"], final_model.named_steps["clf"]
print(f"  Model: {best_name}")
print(f"  Vectorizer: {type(vec).__name__}(ngram_range={vec.ngram_range}, preprocessor=normalize, "
      "tokenizer=tokenize)")
print(f"  Classifier: {clf}")
print(f"  Vocabulary learned from training data: {len(vec.vocabulary_):,} features")

# What the model learned: strongest words per class
weights = clf.coef_ if hasattr(clf, "coef_") else clf.feature_log_prob_ - clf.feature_log_prob_.mean(0)
words = vec.get_feature_names_out()
print("\n  Strongest words per class:")
print(pd.DataFrame({c: words[np.argsort(weights[i])[::-1][:8]]
                    for i, c in enumerate(clf.classes_)}).to_string(index=False))

# The test set is used here for the first and only time.
y_test = test["sentiment"].to_numpy()
y_pred = final_model.predict(test["text_clean"].to_numpy())
macro, acc = f1_score(y_test, y_pred, labels=LABELS, average="macro"), accuracy_score(y_test, y_pred)
majority = train["sentiment"].value_counts().idxmax()
base_acc = (y_test == majority).mean()
base_f1 = f1_score(y_test, np.full_like(y_test, majority), labels=LABELS, average="macro", zero_division=0)
cv_f1 = cv.loc[cv.selected, "macro_f1_mean"].iat[0]

print(f"\n  Test reviews: {len(test):,}")
print(f"  Test macro-F1 {macro:.3f}  (CV estimate {cv_f1:.3f})  |  accuracy {acc:.1%}")
print(f"  Majority baseline (always '{majority}'): accuracy {base_acc:.1%}, macro-F1 {base_f1:.3f}\n")
print(classification_report(y_test, y_pred, labels=LABELS, digits=3, zero_division=0))
correct = y_pred == y_test
short = test["is_short"].to_numpy()
print(f"  Accuracy on short reviews (<{SHORT_REVIEW_WORDS} words): {correct[short].mean():.1%} (n={short.sum():,})"
      f" | longer: {correct[~short].mean():.1%} (n={(~short).sum():,})")
print("\n  Example errors:")
for i in np.where(~correct)[0][:5]:
    print(f"    actual {y_test[i]:<8} predicted {y_pred[i]:<8} {test.at[i, 'text_clean'][:70]}")

cm = confusion_matrix(y_test, y_pred, labels=LABELS)
print("  Confusion matrix (rows = actual, columns = predicted):")
print(pd.DataFrame(cm, index=LABELS, columns=LABELS).to_string())


# #############################################################################
# PART 7 - COMPLAINT ANALYSIS
# #############################################################################
part("Part 7 - Complaint analysis")

# Complaint corpus: negative, not the rider's fault, from branches with enough reviews.
complaints = df[(df["sentiment"] == "Negative") & ~df["rider_blamed"] & df["branch_usable"]].copy()
print(f"  {len(complaints):,} complaints from {complaints['Branch'].nunique()} branches")

# Level 1 - most frequent complaint terms (1-3 word phrases in >= 3 reviews).
GENERIC = {"order", "orders", "ordered", "ordering", "food", "foods", "jollibee", "jolibee",
           "jollibe", "jabee", "delivery", "deliver", "delivered", "foodpanda", "panda", "branch",
           "store", "restaurant", "meal", "meals", "item", "items", "time", "times", "got", "get",
           "just", "like", "really", "please", "pls", "plss", "sana", "ok", "okay", "ung", "nyo",
           "niyo", "po", "nag", "did", "does", "super", "sobrang", "ave"} | NEGATIONS
branch_words = {w for b in complaints["Branch"].unique() for w in tokenize(normalize(b))}


def is_useful_term(term):
    """Hide terms made only of generic words, emoji/numbers, or ending on a negation."""
    w = term.split()
    if not any(c.isalpha() for c in term) or (len(w) > 1 and w[-1] in NEGATIONS):
        return False
    return not all(x in GENERIC or x in branch_words for x in w)


counter = CountVectorizer(preprocessor=normalize, tokenizer=tokenize, token_pattern=None,
                          lowercase=False, ngram_range=(1, 3), min_df=3)
X = counter.fit_transform(complaints["text_clean"])
terms = pd.DataFrame({"term": counter.get_feature_names_out(), "reviews": (X > 0).sum(0).A1,
                      "count": X.sum(0).A1})
terms = terms[terms["term"].map(is_useful_term)].sort_values(["reviews", "count"], ascending=False).head(50)

# Level 2 - complaint themes with regular expressions (English and Tagalog spellings).
ITEMS = r"""(gravy|gravi|rice|kanin|kan-?on|drinks?|fries|spaghetti|spag|sauce|ketchup
    |catsup|utensils?|cutler(y|ies)|spoons?|forks?|tissues?|straws?|ice|ice\s*cream
    |sundaes?|coke|float|toy|burgers?|yumburger|burger\s*steak|chicken|chiken|nuggets
    |dessert|cheese|patty|pie|soup|macaroni|condiments?|hotdog|pancake|iced\s+tea)"""
THEMES = {
    "Missing item": r"""
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
    "Wrong order": r"""
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
    "Late delivery": r"""
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
    "Cold food": r"""
        \bcold\b | \bmalamig\b | \blamig\b
        | \b(not|hindi|di|wala)\s+(na\s+)?(hot|warm|mainit|init|inet)\b
        | \bluke\s*warm\b | \broom\s+temp
    """,
    "Food quality": r"""
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
    "Small portion": r"""
        \bsmall(er|est)?\b | \btiny\b | \bliit\b | \bmaliit\b | \bnapakaliit\b
        | \bapakaliit\b | \bang\s*liit\b | \bsobrang\s+(ma)?liit\b | \bkonti\b
        | \bkaunti\b | \bonti\b | \bdiet\s+size\b | \bpayat\b | \bportions?\b
        | \bserving\b | \bskinny\b | \bthin\b | \bquantity\b
        | \b(back|ribs?|wings?|pakpak|likod)\s+part | \bpuro\s+(back|pakpak|wings?|likod|ribs?|buto|bones?)
        | \ball\s+(back|ribs?|wings?|bones?)\b | \bpuro\s+buto
    """,
    "Staff & service": r"""
        \bstaff\b | \bcrew\b | \bcashier\b | \bmanager\b | \brude\b | \battitude\b
        | \bbastos\b | \bsuplad[ao]\b | \bservice\b | \bserbisyo\b | \bincompetent\b
        | \bunprofessional\b | \bcareless | \bpabaya\b | \bwalang\s+(pakialam|pake)
        | \b(no|not)\s+(response|reply|answer)| \bnot\s+answering\b
        | \bcancel+(ed|ing|led)?\b | \bunavailable\b | \bout\s+of\s+stock\b
        | \bnot\s+available\b | \bnotif(y|ied) | \bcoordinat
    """,
    "Packaging": r"""
        \bpackag | \bspill(ed|s|ing)?\b | \bspilt\b | \btapon\b | \bnatapon\b
        | \btumapon\b | \bleak | \btagas\b | \btumagas\b | \bcontainers?\b
        | \bbox(es)?\b | \blids?\b | \bsealed?\b | \bcrushed\b | \bsquish
        | \bnayupi\b | \byupi\b | \bdurog\b | \bsabog\b | \bnagkalat\b | \bmessy\b
        | \bnapisa\b | \bbasa\b | \bwet\b | \bcups?\b | \bbroken\b | \bseep
        | \blalagyan\b | \bkarton\b
    """,
    "Hygiene": r"""
        \bhairs?\b | \bbuhok\b | \binsects?\b | \bipis\b | \bcockroach | \broach
        | \blanggam\b | \bants?\b | \bflies\b | \bfly\b | \blangaw\b | \bdirty\b
        | \bmadumi\b | \bmarumi\b | \bdumi\b | \bunsanitary\b | \bunhygienic\b
        | \bforeign\s+(object|matter) | \buod\b | \bworms?\b | \bmold | \bamag\b
        | \binsekto | \bbugs?\b
    """,
}
# Lowercase and squash elongations ("lateeee" -> "late"). Python's re is used (not .str
# methods) because pandas 3 string columns use a regex engine without back-references.
theme_text = complaints["text_clean"].map(lambda t: _REPEAT_RE.sub(r"\1", str(t).lower()))
for theme, pattern in THEMES.items():
    rx = re.compile(pattern, re.VERBOSE)
    complaints[theme] = theme_text.map(lambda t, rx=rx: bool(rx.search(t)))

# Branch lift = branch theme rate / chain theme rate (1.0 = average, 2.0 = twice as common).
theme_names = list(THEMES)
chain_rate = complaints[theme_names].mean()
branch_hits = complaints.groupby("Branch")[theme_names].sum()
branch_size = complaints.groupby("Branch").size()
lift = branch_hits.div(branch_size, axis=0).div(chain_rate, axis=1)

priorities = []
for branch in lift.index:
    eligible = lift.loc[branch][branch_hits.loc[branch] >= MIN_THEME_HITS].sort_values(ascending=False)
    priorities.append({"Branch": branch, "complaints": branch_size[branch],
                       "top_issue": eligible.index[0] if len(eligible) else None,
                       "lift": eligible.iloc[0] if len(eligible) else np.nan,
                       "second_issue": eligible.index[1] if len(eligible) > 1 else None})
priorities = pd.DataFrame(priorities).sort_values(["lift", "complaints"], ascending=False)

print("\n  Top 15 complaint terms:")
print(terms.head(15)[["term", "reviews"]].to_string(index=False))
print("\n  Theme share of all complaints:")
print(chain_rate.sort_values(ascending=False).map("{:.1%}".format).to_string())
print(f"  (no theme matched: {(complaints[theme_names].sum(axis=1) == 0).mean():.1%})")
print(f"\n  Branch priorities (notable = lift > {NOTABLE_LIFT} with >= {MIN_THEME_HITS} complaints):")
print(priorities.head(10).to_string(index=False, float_format="{:.2f}".format))
print(f"  >> {(priorities['lift'] > NOTABLE_LIFT).sum()} of {len(priorities)} branches have a notable "
      "issue: this tells each manager WHICH problem is unusually common at THEIR branch.")

print("\n  Theme lift for the 12 branches with the most complaints (1.0 = chain average):")
biggest = branch_size.sort_values(ascending=False).head(12).index
print(lift.loc[biggest].round(2).to_string())


# #############################################################################
# PART 8 - LIMITATIONS
# #############################################################################
part("Part 8 - Limitations and next steps")
print("  - Labels come from star ratings, not from reading the text; 3-star reviews are noisy.")
print("    A hand-labelled sample of 99 reviews checks how far the star labels can be trusted (Part 2).")
print("  - Scope is English and Tagalog/Taglish: Bisaya reviews are dropped by a word list,")
print("    which can miss a few mixed reviews.")
print("  - Hyper-parameters are fixed (alpha = 1.0, C = 1.0), not tuned.")
print("  - Complaint themes are keyword regexes, not validated by hand; some complaints get no theme.")
print("  - The Tagalog lexicon baseline is machine-built and noisy (e.g. 'pagkain' counts as positive).")
print("  - Bag-of-words ignores word order and sarcasm; one brand on one platform only.")
print("  - Next: feed the model and branch priorities into the dashboard.")
print(f"\n  Experiment results cached in {OUT_DIR}")
