# Personal Expense Analyzer

A small Python data-processing app. It loads a raw CSV of personal expenses, cleans it with functions, and analyzes it with NumPy.

All amounts are in Ethiopian birr (ETB). The dataset is self-created (fictional transactions).

## What it does

1. **Loads** `personal_expenses_self_created.csv` (50 rows, 5 columns).
2. **Cleans** the data with small functions:
   - removes duplicate rows
   - trims spaces and fixes capital letters in `category`
   - fills missing categories with `Uncategorized`
   - converts `amount_etb` to numbers (text like `not recorded` becomes missing)
   - removes rows with a missing or non-positive amount
   - converts `date` to real dates
3. **Processes with NumPy**: total expenditure, average expense, median, min, max, total per category, total per month, and the month with the highest spending.
4. **Shows results** and saves the cleaned data to `personal_expenses_clean.csv`.

## Data problems in the raw file

| Problem | Example | Fix |
|---|---|---|
| Duplicate row | Housing 2026-09-11 appears twice | Remove the copy |
| Inconsistent text | ` entertainment ` | Strip spaces, fix capitals |
| Missing category | Concert ticket row | Fill with `Uncategorized` |
| Invalid amount | `not recorded` | Convert, then remove the row |

Result: 50 rows in, 48 rows out.

## Results

- Total expenditure: 26,420 ETB
- Average expense: 550.42 ETB
- Highest spending month: September 2026 (15,555 ETB), against October 2026 (10,865 ETB, data up to 19 October)

## Project files

```
personal_expense_analyzer/
├── main.py                              # the script
├── personal_expense_analyzer.ipynb      # the notebook (Colab / Jupyter)
├── personal_expenses_self_created.csv   # raw dataset
├── requirements.txt                     # dependencies
├── .gitignore
└── README.md
```

## How to run

### Locally (Linux / macOS)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

On Windows, activate with `venv\Scripts\activate` instead.

When asked, type the CSV path: `personal_expenses_self_created.csv`

### Notebook

```bash
jupyter notebook personal_expense_analyzer.ipynb
```

### Google Colab

Open the notebook in Colab, run all cells (Runtime > Run all). If the CSV is not found, the notebook asks you to upload it.

## Libraries

- Python 3
- NumPy
- pandas (loading the CSV and basic cleaning)