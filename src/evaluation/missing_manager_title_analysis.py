import pandas as pd


def run_missing_manager_title_analysis():

    employees = pd.read_parquet("data/processed/employees_cleaned.parquet")

    ground_truth = pd.read_parquet("data/ground_truth/ground_truth_edges.parquet")

    # Employee information
    employee_info = employees[
        [
            "employee_id",
            "company_id",
            "normalized_job_title",
            "department",
        ]
    ].copy()

    # Actual manager information
    manager_info = employee_info.rename(
        columns={
            "employee_id": "manager_employee_id",
            "normalized_job_title": "manager_title",
            "department": "manager_department",
        }
    )

    # Actual reporting relationships
    actual = ground_truth.merge(
        employee_info.rename(
            columns={
                "normalized_job_title": "employee_title",
                "department": "employee_department",
            }
        ),
        on=["employee_id", "company_id"],
        how="left",
    )

    actual = actual.merge(
        manager_info,
        on=["manager_employee_id", "company_id"],
        how="left",
    )

    # Keep relationships where actual manager title is missing.
    missing = actual[actual["manager_title"].isna()].copy()

    missing["same_department"] = (
        missing["employee_department"].notna()
        & missing["manager_department"].notna()
        & (missing["employee_department"] == missing["manager_department"])
    )

    print("\n========== MISSING MANAGER TITLE ANALYSIS ==========")

    print("\nRelationships with missing manager titles:")
    print(len(missing))

    print(
        "\nUnique actual managers with missing titles:",
        missing["manager_employee_id"].nunique(),
    )

    print(
        "\nRelationships where manager department is available:",
        int(missing["manager_department"].notna().sum()),
    )

    print(
        "Relationships where employee department is available:",
        int(missing["employee_department"].notna().sum()),
    )

    print(
        "Relationships where both departments are available:",
        int(
            (
                missing["employee_department"].notna()
                & missing["manager_department"].notna()
            ).sum()
        ),
    )

    print(
        "Relationships with matching departments:",
        int(missing["same_department"].sum()),
    )

    print("\n--- Manager department distribution ---")

    print(missing["manager_department"].value_counts(dropna=False).head(15).to_string())

    print("\n--- Employee department distribution ---")

    print(
        missing["employee_department"].value_counts(dropna=False).head(15).to_string()
    )

    print("\n--- Most common employee/manager department combinations ---")

    print(
        missing.groupby(
            ["employee_department", "manager_department"],
            dropna=False,
        )
        .size()
        .reset_index(name="relationships")
        .sort_values("relationships", ascending=False)
        .head(20)
        .to_string(index=False)
    )

    # Unique missing-title manager details
    unique_managers = (
        missing.groupby(
            [
                "manager_employee_id",
                "company_id",
                "manager_department",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="direct_reports")
    )

    print("\n--- Missing-title managers with most direct reports ---")

    print(
        unique_managers.sort_values(
            "direct_reports",
            ascending=False,
        )
        .head(15)
        .to_string(index=False)
    )


if __name__ == "__main__":
    run_missing_manager_title_analysis()
