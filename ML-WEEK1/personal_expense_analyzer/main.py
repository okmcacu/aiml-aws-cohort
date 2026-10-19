import os
import numpy as np
import pandas as pd


def load_data(file_path):
    """Read the CSV file and return a table (DataFrame)."""
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print("Error: file not found ->", file_path)
        return None


def remove_duplicates(data):
    """Remove rows that are exactly the same as another row."""
    data = data.drop_duplicates()
    return data


def clean_category_text(data):
    """Remove extra spaces and make the capital letters consistent."""
    data["category"] = data["category"].str.strip().str.title()
    return data


def fill_missing_category(data):
    """Rows with no category get the label 'Uncategorized'."""
    data["category"] = data["category"].fillna("Uncategorized")
    return data


def fix_amount_column(data):
    """Turn the amount column into numbers. Text like 'not recorded' becomes NaN."""
    data["amount_etb"] = pd.to_numeric(data["amount_etb"], errors="coerce")
    return data


def remove_invalid_amounts(data):
    """Drop rows where the amount is missing or not above zero."""
    data = data[data["amount_etb"].notna()]
    data = data[data["amount_etb"] > 0]
    return data


def fix_date_column(data):
    """Convert the date column from text into real dates."""
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data[data["date"].notna()]
    return data


def clean_data(data):
    """Run all cleaning steps in order and return the cleaned table."""
    data = data.copy()
    data = remove_duplicates(data)
    data = clean_category_text(data)
    data = fill_missing_category(data)
    data = fix_amount_column(data)
    data = remove_invalid_amounts(data)
    data = fix_date_column(data)
    data = data.reset_index(drop=True)
    return data


def analyze_amounts(data):
    """Use NumPy to calculate the main statistics on the amount column."""
    amounts = data["amount_etb"].to_numpy()

    results = {
        "count": len(amounts),
        "total": np.sum(amounts),
        "mean": np.mean(amounts),
        "median": np.median(amounts),
        "minimum": np.min(amounts),
        "maximum": np.max(amounts),
    }
    return results


def total_per_category(data):
    """Use NumPy to add up the spending of each category."""
    amounts = data["amount_etb"].to_numpy()
    categories = data["category"].to_numpy()

    totals = {}
    for name in np.unique(categories):
        mask = categories == name
        totals[name] = np.sum(amounts[mask])
    return totals


def total_per_month(data):
    """Use NumPy to add up the spending of each month."""
    amounts = data["amount_etb"].to_numpy()
    months = data["date"].dt.strftime("%Y-%m").to_numpy()

    totals = {}
    for month in np.unique(months):
        mask = months == month
        totals[month] = np.sum(amounts[mask])
    return totals


def find_highest_month(month_totals):
    """Return the month with the highest total spending and its amount."""
    names = list(month_totals.keys())
    values = np.array(list(month_totals.values()))
    position = np.argmax(values)
    return names[position], values[position]


def show_results(results, category_totals, month_totals):
    print("\n===== SPENDING SUMMARY (ETB) =====")
    print("Number of transactions:", results["count"])
    print("Total expenditure:     ", round(results["total"], 2))
    print("Average expense:       ", round(results["mean"], 2))
    print("Median expense:        ", round(results["median"], 2))
    print("Smallest expense:      ", round(results["minimum"], 2))
    print("Largest expense:       ", round(results["maximum"], 2))

    print("\n===== TOTAL PER MONTH (ETB) =====")
    for month, total in month_totals.items():
        print(f"{month}   {total:10.2f}")

    top_month, top_total = find_highest_month(month_totals)
    print("\nMonth with the highest spending:", top_month, "->", round(top_total, 2), "ETB")

    print("\n===== TOTAL PER CATEGORY (biggest first) =====")
    sorted_totals = sorted(category_totals.items(), key=lambda item: item[1], reverse=True)
    for name, total in sorted_totals:
        print(f"{name:20s} {total:10.2f}")


def main():
    
    file_path = input("Enter the path to your CSV file: ").strip()

    raw = load_data(file_path)
    if raw is None:
        return

    if raw.empty:
        print("The file is empty. Nothing to process.")
        return

    clean = clean_data(raw)

    if clean.empty:
        print("No valid rows left after cleaning.")
        return

    results = analyze_amounts(clean)
    category_totals = total_per_category(clean)
    month_totals = total_per_month(clean)

    show_results(results, category_totals, month_totals)

    clean.to_csv("personal_expenses_clean.csv", index=False)
    print("\nCleaned data saved to personal_expenses_clean.csv \n Personal Expense Tracker , By Mikiyas Aschalew")


if __name__ == "__main__":
    main()