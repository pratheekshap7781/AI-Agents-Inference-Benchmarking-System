"""
One-time setup script. The original data/amazon_cleaned.db, data/questions.py
and data/expected_results.csv are missing from this repo (data/ is
gitignored and was never committed) and are NOT reconstructed from thin air.

This script builds them from a real CSV you provide: the public "Amazon
Sales Dataset" on Kaggle (search "Amazon Sales Dataset karkavelrajaj" —
https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset). Its
columns (product_id, category, discounted_price, actual_price,
discount_percentage, rating, rating_count, ...) match the columns referenced
in the project's own crew.log.txt ('category', 'discount_percentage',
'rating', product_id).

Usage:
    1. Download the CSV from Kaggle.
    2. Save it as:  data/raw/amazon.csv   (relative to this project folder)
    3. Run:  python app/data_setup.py

What it does:
    - Cleans the raw CSV (strips currency symbols/commas from price columns,
      strips '%' from discount_percentage, coerces rating to numeric,
      drops rows that don't parse) and loads it into a new SQLite database
      data/amazon_cleaned.db, table name `products`.
    - Writes data/questions.py with the 5 question strings recovered from
      crew.log.txt (see app/known_questions.py for provenance).
    - Computes the expected answer for each question by running the actual
      SQL against the real cleaned data with sqlite3 — these are REAL
      computed values, not invented ones — and writes data/expected_results.csv
      in the question order that the benchmark iterates in.

If your CSV's columns differ from the Kaggle dataset, edit COLUMN_MAP below
and re-run.
"""
import re
import sqlite3
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).parent.parent
RAW_CSV = PROJECT_DIR / "data" / "raw" / "amazon.csv"
DB_PATH = PROJECT_DIR / "data" / "amazon_cleaned.db"
QUESTIONS_PATH = PROJECT_DIR / "data" / "questions.py"
EXPECTED_RESULTS_PATH = PROJECT_DIR / "data" / "expected_results.csv"
TABLE_NAME = "products"

# Map from this script's internal names -> the column name in your CSV.
# Defaults match the Kaggle "Amazon Sales Dataset".
COLUMN_MAP = {
    "product_id": "product_id",
    "product_name": "product_name",
    "category": "category",
    "discounted_price": "discounted_price",
    "actual_price": "actual_price",
    "discount_percentage": "discount_percentage",
    "rating": "rating",
    "rating_count": "rating_count",
}

QUESTIONS = [
    {
        "key": "question_1",
        "question": "How many unique product IDs are there in the table?",
        "sql": f"SELECT COUNT(DISTINCT product_id) FROM {TABLE_NAME};",
    },
    {
        "key": "question_2",
        "question": "How many unique values for 'category' are in the database?",
        "sql": f"SELECT COUNT(DISTINCT category) FROM {TABLE_NAME};",
    },
    {
        "key": "question_3",
        "question": "Order the products by product_id and return the rating of the first product?",
        "sql": f"SELECT rating FROM {TABLE_NAME} ORDER BY product_id ASC LIMIT 1;",
    },
    {
        "key": "question_4",
        "question": "What is the average rating, rounded to two decimal places, of the products in the category 'Electronics'?",
        "sql": f"SELECT ROUND(AVG(rating), 2) FROM {TABLE_NAME} WHERE category LIKE '%Electronics%';",
    },
    {
        "key": "question_5",
        "question": "What is the maximum value in the column 'discount_percentage' in the table?",
        "sql": f"SELECT MAX(discount_percentage) FROM {TABLE_NAME};",
    },
]


def _clean_money(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(r"[₹$,]", "", regex=True).str.strip(),
        errors="coerce",
    )


def _clean_percent(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace("%", "", regex=False).str.strip(),
        errors="coerce",
    )


def build_database() -> None:
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"Expected raw CSV at {RAW_CSV}. Download the Kaggle 'Amazon Sales "
            f"Dataset' and save it there (or edit RAW_CSV in this script)."
        )

    df = pd.read_csv(RAW_CSV)
    missing_cols = [c for c in COLUMN_MAP.values() if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"CSV is missing expected column(s): {missing_cols}. "
            f"Edit COLUMN_MAP in app/data_setup.py to match your file's headers."
        )

    cleaned = pd.DataFrame()
    cleaned["product_id"] = df[COLUMN_MAP["product_id"]].astype(str)
    cleaned["product_name"] = df[COLUMN_MAP["product_name"]].astype(str)
    cleaned["category"] = df[COLUMN_MAP["category"]].astype(str)
    cleaned["discounted_price"] = _clean_money(df[COLUMN_MAP["discounted_price"]])
    cleaned["actual_price"] = _clean_money(df[COLUMN_MAP["actual_price"]])
    cleaned["discount_percentage"] = _clean_percent(df[COLUMN_MAP["discount_percentage"]])
    cleaned["rating"] = pd.to_numeric(df[COLUMN_MAP["rating"]], errors="coerce")
    cleaned["rating_count"] = _clean_money(df[COLUMN_MAP["rating_count"]])

    before = len(cleaned)
    cleaned = cleaned.dropna(subset=["product_id", "category", "rating", "discount_percentage"])
    after = len(cleaned)
    print(f"Loaded {before} rows, kept {after} after dropping unparseable rows.")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)
    cleaned.to_sql(TABLE_NAME, conn, index=False)
    conn.close()
    print(f"Wrote {DB_PATH} with table '{TABLE_NAME}' ({after} rows).")


def write_questions_file() -> None:
    lines = ["# Auto-generated by app/data_setup.py from crew.log.txt-recovered questions.",
             "# `queries` matches the schema expected by utils/helper_functions.extract_questions().",
             "queries = {"]
    for q in QUESTIONS:
        safe_q = q["question"].replace('"', '\\"')
        lines.append(f'    "{q["key"]}": {{"question": "{safe_q}"}},')
    lines.append("}")
    QUESTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUESTIONS_PATH.write_text("\n".join(lines) + "\n")
    # Package needs an __init__.py to be importable as `data.questions`
    (QUESTIONS_PATH.parent / "__init__.py").touch(exist_ok=True)
    print(f"Wrote {QUESTIONS_PATH}")


def write_expected_results() -> None:
    conn = sqlite3.connect(DB_PATH)
    rows = []
    for q in QUESTIONS:
        cur = conn.execute(q["sql"])
        value = cur.fetchone()[0]
        rows.append({"question": q["question"], "answer": value})
        print(f"  {q['question']}\n    -> SQL: {q['sql']}\n    -> real answer: {value}")
    conn.close()

    df = pd.DataFrame(rows)
    EXPECTED_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(EXPECTED_RESULTS_PATH, index=False)
    print(f"Wrote {EXPECTED_RESULTS_PATH} ({len(df)} rows, computed from real SQL execution).")


if __name__ == "__main__":
    print("=== Building amazon_cleaned.db from your CSV ===")
    build_database()
    print("\n=== Writing data/questions.py (5 recovered questions) ===")
    write_questions_file()
    print("\n=== Computing real expected answers via SQL ===")
    write_expected_results()
    print("\nDone. Live mode in the Streamlit app will now work.")
