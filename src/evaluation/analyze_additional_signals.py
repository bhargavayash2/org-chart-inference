from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

EMPLOYEES_PATH = ROOT / "data/raw/employees.parquet"
COMPARISON_PATH = ROOT / "data/processed/actual_vs_inferred_candidates.csv"
OUTPUT_PATH = ROOT / "data/processed/additional_signals_analysis.csv"


def load_data():
    print("Loading datasets...")

    employees = pd.read_parquet(EMPLOYEES_PATH)
    comparison = pd.read_csv(COMPARISON_PATH)

    employees["hire_date"] = pd.to_datetime(employees["hire_date"], errors="coerce")

    print(f"Employees: {len(employees):,}")
    print(f"Comparison rows: {len(comparison):,}")

    return employees, comparison


def attach_employee_details(comparison, employees):
    # Employee details
    employee_details = employees[
        [
            "employee_id",
            "hire_date",
            "city",
            "country",
            "department",
        ]
    ].copy()

    employee_details = employee_details.rename(
        columns={
            "employee_id": "employee_id",
            "hire_date": "employee_hire_date",
            "city": "employee_city",
            "country": "employee_country",
            "department": "employee_department_raw",
        }
    )

    result = comparison.merge(
        employee_details,
        on="employee_id",
        how="left",
    )

    # Actual manager details (ground truth: diagnostics only)
    actual_details = employees[
        ["employee_id", "hire_date", "city", "country", "department"]
    ].rename(
        columns={
            "employee_id": "manager_employee_id",
            "hire_date": "actual_hire_date",
            "city": "actual_city",
            "country": "actual_country",
            "department": "actual_department_raw",
        }
    )

    result = result.merge(
        actual_details,
        on="manager_employee_id",
        how="left",
    )

    # Inferred manager details
    inferred_details = employees[
        ["employee_id", "hire_date", "city", "country", "department"]
    ].rename(
        columns={
            "employee_id": "inferred_manager_employee_id",
            "hire_date": "inferred_hire_date",
            "city": "inferred_city",
            "country": "inferred_country",
            "department": "inferred_department_raw",
        }
    )

    result = result.merge(
        inferred_details,
        on="inferred_manager_employee_id",
        how="left",
    )

    return result


def add_signal_columns(df):
    # Hire-date comparisons
    df["actual_hired_before_employee"] = (
        df["actual_hire_date"] < df["employee_hire_date"]
    )

    df["inferred_hired_before_employee"] = (
        df["inferred_hire_date"] < df["employee_hire_date"]
    )

    df["actual_hire_gap_days"] = (
        df["employee_hire_date"] - df["actual_hire_date"]
    ).dt.days

    df["inferred_hire_gap_days"] = (
        df["employee_hire_date"] - df["inferred_hire_date"]
    ).dt.days

    df["actual_hired_before_inferred"] = (
        df["actual_hire_date"] < df["inferred_hire_date"]
    )

    df["inferred_hired_before_actual"] = (
        df["inferred_hire_date"] < df["actual_hire_date"]
    )

    # City comparisons
    df["actual_same_city"] = (
        df["actual_city"].notna()
        & df["employee_city"].notna()
        & df["actual_city"].eq(df["employee_city"])
    )

    df["inferred_same_city"] = (
        df["inferred_city"].notna()
        & df["employee_city"].notna()
        & df["inferred_city"].eq(df["employee_city"])
    )

    # Country comparisons
    df["actual_same_country"] = df["actual_country"].eq(df["employee_country"])

    df["inferred_same_country"] = df["inferred_country"].eq(df["employee_country"])

    return df


def print_boolean_comparison(df, actual_col, inferred_col, label):
    actual = df[actual_col]
    inferred = df[inferred_col]

    valid = actual.notna() & inferred.notna()

    actual = actual[valid].astype(bool)
    inferred = inferred[valid].astype(bool)

    print(f"\n{label}")
    print("-" * 55)
    print(f"Comparable rows: {len(actual):,}")
    print(f"Actual manager satisfies signal: " f"{actual.mean():.2%}")
    print(f"Inferred manager satisfies signal: " f"{inferred.mean():.2%}")
    print(f"Actual satisfies, inferred does not: " f"{(actual & ~inferred).sum():,}")
    print(f"Inferred satisfies, actual does not: " f"{(~actual & inferred).sum():,}")
    print(f"Both satisfy: {(actual & inferred).sum():,}")
    print(f"Neither satisfies: {(~actual & ~inferred).sum():,}")


def main():
    employees, comparison = load_data()

    print("\nAttaching employee attributes...")
    df = attach_employee_details(comparison, employees)

    print("\nCalculating additional signals...")
    df = add_signal_columns(df)

    print("\n========== HIRE DATE ANALYSIS ==========")

    print_boolean_comparison(
        df,
        "actual_hired_before_employee",
        "inferred_hired_before_employee",
        "Manager hired before employee",
    )

    print("\nHire-date gap (days):")
    print(
        df[["actual_hire_gap_days", "inferred_hire_gap_days"]]
        .describe()
        .round(1)
        .to_string()
    )

    print_boolean_comparison(
        df,
        "actual_hired_before_inferred",
        "inferred_hired_before_actual",
        "Which manager was hired earlier?",
    )

    print("\n========== CITY ANALYSIS ==========")

    print_boolean_comparison(
        df,
        "actual_same_city",
        "inferred_same_city",
        "Manager and employee share city",
    )

    print("\n========== COUNTRY ANALYSIS ==========")

    print_boolean_comparison(
        df,
        "actual_same_country",
        "inferred_same_country",
        "Manager and employee share country",
    )

    print("\n========== DATA AVAILABILITY ==========")

    columns = [
        "employee_hire_date",
        "actual_hire_date",
        "inferred_hire_date",
        "employee_city",
        "actual_city",
        "inferred_city",
        "employee_country",
        "actual_country",
        "inferred_country",
    ]

    for column in columns:
        print(f"{column}: " f"{df[column].notna().sum():,} available / " f"{len(df):,}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nDetailed results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
