import pandas as pd

from src.processing.title_normalizer import normalize_employee_titles


def clean_employee_data(employees_df):
    """
    Clean employee data while preserving records with incomplete information.

    Operations:
    1. Validate required columns.
    2. Remove duplicate employee IDs.
    3. Standardize whitespace in text columns.
    4. Convert empty strings into missing values.
    5. Validate employee and company identifiers.
    """

    required_columns = ["employee_id", "company_id"]

    missing_columns = [
        column for column in required_columns if column not in employees_df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    cleaned_df = employees_df.copy()

    # Employee IDs are essential for identifying individual employees.
    if cleaned_df["employee_id"].isna().any():
        raise ValueError("Employee IDs contain missing values.")

    if cleaned_df["company_id"].isna().any():
        raise ValueError("Company IDs contain missing values.")

    # Standardize whitespace and convert empty strings to missing values.
    text_columns = cleaned_df.select_dtypes(include=["object", "string"]).columns

    for column in text_columns:
        cleaned_df[column] = (
            cleaned_df[column].astype("string").str.strip().replace("", pd.NA)
        )

    # Remove duplicate employee IDs, keeping the first occurrence.
    cleaned_df = cleaned_df.drop_duplicates(subset=["employee_id"], keep="first")

    # Validate identifiers after cleaning.
    if not cleaned_df["employee_id"].is_unique:
        raise ValueError("Duplicate employee IDs remain after cleaning.")

    if cleaned_df["company_id"].isna().any():
        raise ValueError("Company IDs contain missing values.")

    return cleaned_df.reset_index(drop=True)


def print_data_quality_report(original_df, cleaned_df):
    """Print a summary of the cleaning results."""

    print("\n========== DATA QUALITY REPORT ==========")

    print("Original employee count:", len(original_df))
    print("Cleaned employee count:", len(cleaned_df))
    print("Records removed:", len(original_df) - len(cleaned_df))

    print("Unique employee IDs:", cleaned_df["employee_id"].nunique())

    print("Unique companies:", cleaned_df["company_id"].nunique())

    print("\nMissing values after cleaning:")

    print(cleaned_df.isna().sum().sort_values(ascending=False).to_string())

    print(
        "\nDuplicate employee IDs remaining:",
        cleaned_df["employee_id"].duplicated().sum(),
    )

    print("=========================================")


if __name__ == "__main__":

    # Read the original synthetic employee dataset.
    employees = pd.read_parquet("data/raw/employees.parquet")

    # Normalize job titles using Step 1.
    normalized_employees = normalize_employee_titles(employees)

    # Clean employee records.
    cleaned_employees = clean_employee_data(normalized_employees)

    # Save the cleaned dataset.
    cleaned_employees.to_parquet(
        "data/processed/employees_cleaned.parquet", index=False
    )

    # Display data quality report.
    print_data_quality_report(normalized_employees, cleaned_employees)

    print("\nCleaned dataset saved to: " "data/processed/employees_cleaned.parquet")
