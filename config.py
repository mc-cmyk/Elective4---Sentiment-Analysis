"""Single source of every constant and path in the ReviewLens pipeline.

Every script imports from here; no other file defines these values.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
PROJECT_DIR = Path(__file__).resolve().parent
RAW_DIR = PROJECT_DIR / "data" / "raw"      # read-only input, never written to
OUTPUT_DIR = PROJECT_DIR / "output"         # everything generated goes here
OUTPUT_DIR.mkdir(exist_ok=True)

KAGGLE_DATASET = "bwandowando/food-panda-restaurant-reviews"
RESTOS_FILE = "ph_restos_2025.csv"
REVIEWS_FILE = "ph_reviews_2025.csv"
LEGACY_RESTOS_FILE = "restos.csv"
RAW_FILES = (RESTOS_FILE, REVIEWS_FILE, LEGACY_RESTOS_FILE)

# ---------------------------------------------------------------- project
BRAND = "Jollibee"
SEED = 42
TEST_FRACTION = 0.20
N_FOLDS = 5

# ---------------------------------------------------------------- labels
NEGATIVE_MAX_STAR = 2       # 1-2 stars -> Negative
NEUTRAL_STAR = 3            # 3 stars   -> Neutral; 4-5 -> Positive
LABELS = ["Negative", "Neutral", "Positive"]
RIDER_BLAME_MAX_STAR = 2    # rider score <= this -> complaint blamed on the rider
SHORT_REVIEW_WORDS = 3      # fewer words than this -> is_short
MIN_REVIEWS_PER_BRANCH = 30 # branches below this are flagged, not deleted

# ---------------------------------------------------------------- model grid (train_grid.py)
NGRAM_RANGES = [(1, 1), (2, 2), (3, 3), (1, 3)]
LR_MAX_ITER = 2000
NB_ALPHA = 1.0              # sklearn defaults, as in train_grid.py
LR_C = 1.0
SVC_C = 1.0

# ------------------------------------- word+char TF-IDF configuration (Train_Sentiment.py)
WORD_NGRAM = (1, 2)
WORD_MIN_DF = 2
CHAR_NGRAM = (3, 5)
CHAR_MIN_DF = 3
WC_NB_ALPHA = 0.5
WC_LR_C = 4.0
WC_SVC_C = 0.5

# ---------------------------------------------------------------- complaint extraction
TOP_N_CHAIN = 50
TOP_N_BRANCH = 15
NOTABLE_LIFT = 1.3
MIN_THEME_HITS = 3          # complaints a theme needs at a branch to rank as a top issue
CHAIN_MIN_DF = 3
CHAIN_NGRAM = (1, 3)
BRANCH_MIN_DF = 2
BRANCH_NGRAM = (2, 3)

# ---------------------------------------------------------------- output files
_B = BRAND.lower()
ALL_CSV = OUTPUT_DIR / f"{_B}_all.csv"
COMPLAINTS_CSV = OUTPUT_DIR / f"{_B}_complaints.csv"
BRANCH_SUMMARY_CSV = OUTPUT_DIR / f"{_B}_branch_summary.csv"
TRAIN_CSV = OUTPUT_DIR / f"{_B}_train.csv"
TEST_CSV = OUTPUT_DIR / f"{_B}_test.csv"
CV_RESULTS_CSV = OUTPUT_DIR / "cv_results.csv"
BEST_MODEL_PATH = OUTPUT_DIR / "best_model.joblib"
BEST_CONFIG_JSON = OUTPUT_DIR / "best_config.json"
PREDICTIONS_CSV = OUTPUT_DIR / "predictions_test.csv"
TEST_METRICS_JSON = OUTPUT_DIR / "test_metrics.json"
TERMS_CHAIN_CSV = OUTPUT_DIR / "complaint_terms_chain.csv"
TERMS_BRANCH_CSV = OUTPUT_DIR / "complaint_terms_branch.csv"
THEMES_BRANCH_CSV = OUTPUT_DIR / "complaint_themes_branch.csv"
BRANCH_PRIORITIES_CSV = OUTPUT_DIR / "branch_priorities.csv"


def resolve_raw_dir():
    """data/raw/ if it holds the raw CSVs; otherwise kagglehub's cached download.
    Either way the files are only read."""
    if all((RAW_DIR / f).exists() for f in RAW_FILES):
        return RAW_DIR
    import kagglehub
    return Path(kagglehub.dataset_download(KAGGLE_DATASET))
