# ReviewLens: Sentiment Analytics for Jollibee Foodpanda Reviews

## Users
Jollibee branch managers, who need to know which branches get negative reviews
and what those complaints are about, so they can fix the right problem first.

## Objectives
1. **Preprocessing:** normalize English, Tagalog, Bisaya and Taglish review text
   without stemming, and keep negations ("hindi", "wala", "dili", "not") and emoji.
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
- **Jollibee subset:** 6,490 reviews from 35 branches.
- **Split:** stratified 80/20 into train 5,192 and test 1,298.

Put the three CSVs in `data/raw/`. This folder is read-only, and no script writes
to it. If it is empty, the scripts read Kaggle's cached copy through `kagglehub`.
Everything generated goes to `output/`.

## Run order
```
pip install -r requirements.txt
python 01_inspect_data.py        # shape, missing values, duplicates, ratings, language mix
python 02_build_dataset.py       # Jollibee filter, labels, short / rider-blamed flags
python 03_split.py               # stratified 80/20 split, class balance
python 04_train_experiments.py   # 29-row CV grid incl. word+char TF-IDF and baselines (~2 min)
python 05_evaluate.py            # held-out test metrics
python 06_extract_complaints.py  # complaint terms, themes, branch priorities
```
Each script reads only what the previous ones wrote to `output/`, and ends by
printing the numbers to present. All constants and paths are set in `config.py`.
`text_prep.py` is the only preprocessing module.

`dashboard/app.py` (Streamlit) is not part of the midterm and has been left
unchanged. It reads from `dashboard/output/`, not `output/`.
