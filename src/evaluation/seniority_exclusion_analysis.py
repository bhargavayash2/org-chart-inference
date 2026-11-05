import pandas as pd

from src.inference.candidate_generator import get_seniority_rank


def run_seniority_exclusion_analysis():
    employees = pd.read_parquet("data/processed/employees_cleaned.parquet")
    ground_truth = pd.read_parquet("data/ground_truth/ground_truth_edges.parquet")

    title_column = (
        "normalized_job_title"
        if "normalized_job_title" in employees.columns
        else "job_title"
    )

    employee_info = employees[["employee_id", "company_id", title_column]].copy()

    employee_info["seniority_rank"] = employee_info[title_column].apply(
        get_seniority_rank
    )

    employee_info = employee_info.rename(
        columns={
            title_column: "employee_title",
            "seniority_rank": "employee_rank",
        }
    )

    manager_info = employees[["employee_id", "company_id", title_column]].copy()

    manager_info["seniority_rank"] = manager_info[title_column].apply(
        get_seniority_rank
    )

    manager_info = manager_info.rename(
        columns={
            "employee_id": "manager_employee_id",
            title_column: "manager_title",
            "seniority_rank": "manager_rank",
        }
    )

    actual = ground_truth.merge(
        employee_info,
        on=["employee_id", "company_id"],
        how="left",
    )

    actual = actual.merge(
        manager_info,
        on=["manager_employee_id", "company_id"],
        how="left",
    )

    actual["seniority_eligible"] = actual["manager_rank"] > actual["employee_rank"]

    excluded = actual[~actual["seniority_eligible"]].copy()

    print("\n========== SENIORITY EXCLUSION ANALYSIS ==========")

    print("\nTotal actual relationships:", len(actual))
    print("Excluded by seniority:", len(excluded))

    print("\n--- Employee title and rank distribution ---")
    print(
        excluded.groupby(
            ["employee_title", "employee_rank"],
            dropna=False,
        )
        .size()
        .reset_index(name="relationships")
        .sort_values("relationships", ascending=False)
        .head(20)
        .to_string(index=False)
    )

    print("\n--- Actual manager title and rank distribution ---")
    print(
        excluded.groupby(
            ["manager_title", "manager_rank"],
            dropna=False,
        )
        .size()
        .reset_index(name="relationships")
        .sort_values("relationships", ascending=False)
        .head(20)
        .to_string(index=False)
    )

    print("\n--- Most common employee-manager title combinations ---")
    print(
        excluded.groupby(
            [
                "employee_title",
                "employee_rank",
                "manager_title",
                "manager_rank",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="relationships")
        .sort_values("relationships", ascending=False)
        .head(25)
        .to_string(index=False)
    )

    print("\n--- Seniority rank combinations ---")
    print(
        excluded.groupby(
            ["employee_rank", "manager_rank"],
            dropna=False,
        )
        .size()
        .reset_index(name="relationships")
        .sort_values("relationships", ascending=False)
        .to_string(index=False)
    )

    print("\n--- Missing title impact ---")
    print(
        "Employee title missing:",
        int(excluded["employee_title"].isna().sum()),
    )
    print(
        "Manager title missing:",
        int(excluded["manager_title"].isna().sum()),
    )

    print("\nAll excluded relationships accounted for:", len(excluded) == 10316)


if __name__ == "__main__":
    run_seniority_exclusion_analysis()
