"""
The original data/questions.py (and the SQLite database it was tested
against) is not present in this repository or in the project zip — it was
listed in .gitignore and never committed. It is NOT invented here.

These 5 question strings were recovered verbatim from crew.log.txt, a real
CrewAI execution log already committed to this repository, which recorded
every task description from real past runs. They are shown in the UI as a
clearly-labeled fallback ONLY until data/questions.py exists (see
app/data_setup.py) — at that point the app reads the real questions.py
instead of this file.
"""

RECOVERED_QUESTIONS = [
    "How many unique product IDs are there in the table?",
    "How many unique values for 'category' are in the database?",
    "Order the products by product_id and return the rating of the first product?",
    "What is the average rating, rounded to two decimal places, of the products in the category 'Electronics'?",
    "What is the maximum value in the column 'discount_percentage' in the table?",
]

SOURCE_NOTE = (
    "Recovered from crew.log.txt (a real log file already in this repo) — "
    "data/questions.py itself is missing because data/ is gitignored."
)
