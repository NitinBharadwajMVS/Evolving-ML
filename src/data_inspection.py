from pathlib import Path
import pandas as pd


def main():
    # 1. Reliably construct the path to the CSV file using pathlib
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "raw" / "creditcard.csv"

    print(f"Loading dataset from: {data_path}\n")

    # Check if the dataset exists before loading
    if not data_path.exists():
        print(f"Error: Dataset file not found at {data_path}")
        print("Please place 'creditcard.csv' in the 'data/raw/' directory.")
        return

    # 2. Load the CSV file into a pandas DataFrame
    df = pd.read_csv(data_path)

    # 3. Print dataset shape (rows and columns)
    print("=" * 50)
    print("1. DATASET SHAPE")
    print("=" * 50)
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n")

    # 4. Print the first 5 rows
    print("=" * 50)
    print("2. FIRST 5 ROWS")
    print("=" * 50)
    print(df.head(), "\n")

    # 5. Print all column names
    print("=" * 50)
    print("3. COLUMN NAMES")
    print("=" * 50)
    print(list(df.columns), "\n")

    # 6. Print data types and missing value counts
    print("=" * 50)
    print("4. DATA TYPES & MISSING VALUES")
    print("=" * 50)
    summary_df = pd.DataFrame({
        "Data Type": df.dtypes,
        "Missing Values": df.isnull().sum()
    })
    print(summary_df, "\n")

    # 7. Print Class distribution (counts and percentages)
    print("=" * 50)
    print("5. CLASS DISTRIBUTION (FRAUD vs. NON-FRAUD)")
    print("=" * 50)
    class_counts = df["Class"].value_counts()
    class_percentages = df["Class"].value_counts(normalize=True) * 100

    distribution_df = pd.DataFrame({
        "Count": class_counts,
        "Percentage (%)": class_percentages.round(4)
    })
    print(distribution_df)


if __name__ == "__main__":
    main()
