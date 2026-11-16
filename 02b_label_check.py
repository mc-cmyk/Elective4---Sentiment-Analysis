"""Step 02b - manual check of the star-rating proxy labels.

The proposal labels reviews from stars (4-5 Positive, 3 Neutral, 1-2 Negative)
and promises to verify a sample by hand. This script does both halves:

  1st run  -> writes output/02_dataset/label_check_sample.csv: LABEL_CHECK_N
              reviews, an equal share from each class, with an empty
              `manual_label` column. Fill it in (Negative / Neutral / Positive)
              from the TEXT ONLY - hide the rating/sentiment columns while
              labelling so the stars don't bias you.
  next run -> once every row has a manual_label, reports agreement and Cohen's
              kappa between the star label and the human label, per class.

Run: python 02b_label_check.py
"""
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from config import ALL_CSV, DATASET_DIR, LABEL_CHECK_N, LABELS, SEED

SHEET = DATASET_DIR / "label_check_sample.csv"
RESULT = DATASET_DIR / "label_check_result.txt"


def make_sheet():
    df = pd.read_csv(ALL_CSV, encoding="utf-8-sig")
    per_class = LABEL_CHECK_N // len(LABELS)
    sample = (df.groupby("sentiment").sample(n=per_class, random_state=SEED)
                .sample(frac=1, random_state=SEED))  # shuffle so classes are mixed
    sheet = sample[["uuid", "Branch", "text_clean", "rating", "sentiment"]].rename(
        columns={"sentiment": "star_label"})
    sheet["manual_label"] = ""
    sheet["notes"] = ""
    sheet.to_csv(SHEET, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(sheet)} reviews to {SHEET}")
    print("Fill `manual_label` with Negative / Neutral / Positive using the text only, "
          "then run this script again.")


def score_sheet():
    sheet = pd.read_csv(SHEET, encoding="utf-8-sig")
    manual = sheet["manual_label"].fillna("").astype(str).str.strip().str.capitalize()
    done = manual.isin(LABELS)
    if not done.all():
        print(f"{(~done).sum()} of {len(sheet)} rows still need a manual_label "
              f"(Negative / Neutral / Positive) in {SHEET.name}.")
        return
    star, manual = sheet["star_label"], manual
    agree = (star == manual).mean()
    kappa = cohen_kappa_score(star, manual, labels=LABELS)
    cm = pd.DataFrame(confusion_matrix(star, manual, labels=LABELS),
                      index=[f"star {l}" for l in LABELS],
                      columns=[f"manual {l}" for l in LABELS])
    per_class = sheet.assign(ok=star == manual).groupby("star_label")["ok"].mean()

    lines = [
        "=" * 78, "NUMBERS TO PRESENT - PROXY LABEL CHECK", "=" * 78,
        f"Reviews checked by hand: {len(sheet)}",
        f"Agreement between star label and manual label: {agree:.1%}",
        f"Cohen's kappa: {kappa:.2f}  (>0.60 substantial, >0.80 almost perfect)",
        "", "Agreement by star label:",
        per_class.reindex(LABELS).to_string(float_format=lambda v: f"{v:.1%}"),
        "", "Confusion (rows = star label, columns = manual label):", cm.to_string(),
    ]
    text = "\n".join(lines)
    print(text)
    RESULT.write_text(text + "\n", encoding="utf-8")
    print(f"\nSaved: {RESULT.name}")


if __name__ == "__main__":
    score_sheet() if SHEET.exists() else make_sheet()
