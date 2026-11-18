# ReviewLens: Sentiment Analytics for Jollibee Foodpanda Reviews

## Users
Jollibee branch managers, who need to know which branches get negative reviews
and what those complaints are about, so they can fix the right problem first.

## Objectives
1. **Preprocessing:** normalize English, Tagalog and Taglish review text (Bisaya
   reviews are out of scope and dropped) without stemming, and keep negations
   ("hindi", "wala", "not") and emoji.
2. **Sentiment classification:** label each review Negative, Neutral or Positive
   from its text.
3. **Complaint extraction:** find the terms and themes behind branch-attributable
   complaints, and rank each branch's top issues against the chain average.
5. **Evaluation:** report cross-validated macro-F1, per-class scores and a confusion
   matrix against majority-class and AFINN baselines.

## Dataset
Kaggle [`bwandowando/food-panda-restaurant-reviews`](https://www.kaggle.com/datasets/bwandowando/food-panda-restaurant-reviews)
(version 13). It covers Foodpanda Philippines reviews from 2024-02-12 to
2025-04-18 for the Jollibee subset.

| File | Rows | Used for |
|---|---|---|
| `ph_reviews_2025.csv` | 212,005 | review text, `overall` and `rider` stars |
| `ph_restos_2025.csv`  | 12,409  | store name, city |
| `restos.csv`          | legacy store table | shape/missing/duplicates check in step 01 only |

- **Labels from stars:** 1–2 → Negative, 3 → Neutral, 4–5 → Positive.
- **Jollibee subset:** 6,490 reviews from 35 branches; 6,287 after dropping 168
  Bisaya reviews and 35 near-duplicate spam reviews.
- **Split:** stratified 80/20 into train 5,029 and test 1,258, identical texts
  kept on one side.

Put the three CSVs in `data/raw/`. This folder is read-only, and nothing writes
to it. If it is empty, the pipeline reads Kaggle's cached copy through `kagglehub`.
The Tagalog lexicon (Chen & Skiena, 2014) is read from `81_lang_lexicon/`.
Everything generated goes to `output/`, one subfolder per step (`02_dataset/`,
`03_split/`, `04_training/`, `05_evaluation/`, `06_complaints/`).

## Run
```
pip install -r requirements.txt
python sentiment_pipeline.py     # ~2 min
```
`sentiment_pipeline.py` holds the whole process, in order:
1. Inspect the raw data
2. Build the dataset: Jollibee filter, star labels, drop Bisaya + near-duplicate spam
2b. Manual label check: writes a 99-review sheet (`output/02_dataset/label_check_sample.csv`)
   on the first run; once `manual_label` is filled in, reports agreement and Cohen's kappa
3. Stratified 80/20 split
4. Experiments: proposed 24-config grid (+3 extra word+char, AFINN, AFINN+Tagalog,
   majority baselines); final model = best macro-F1 of the 24
5. Held-out test evaluation
6. Complaint terms, themes and branch priorities

Each step prints the numbers to present. All settings are in the CONFIGURATION
section at the top of the file.

The Streamlit dashboard is parked outside the project (`../Elect4-sentimenAnalysis_parked/dashboard/`)
until the visualization phase; this phase covers the analysis process only.
