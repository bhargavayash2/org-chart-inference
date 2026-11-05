import pandas as pd

from src.inference.candidate_generator import (
    generate_candidates_for_company,
    get_seniority_rank,
    AMBIGUOUS_MANAGER_TITLES,
)


def run_candidate_generation_diagnostics():

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

    actual["equal_seniority_allowed"] = actual["employee_title"].fillna(
        ""
    ).str.strip().str.lower().isin(AMBIGUOUS_MANAGER_TITLES) & (
        actual["manager_rank"] == actual["employee_rank"]
    )

    actual["manager_in_pool"] = actual["manager_rank"] > 1

    actual["seniority_eligible"] = (
        actual["manager_rank"] > actual["employee_rank"]
    ) | actual["equal_seniority_allowed"]

    candidate_pairs = []

    for company_id in employees["company_id"].unique():

        candidates = generate_candidates_for_company(
            employees,
            company_id=company_id,
            max_candidates=10,
        )

        if not candidates.empty:

            candidate_pairs.append(
                candidates[
                    [
                        "company_id",
                        "employee_id",
                        "candidate_manager_id",
                    ]
                ]
            )

    generated = pd.concat(
        candidate_pairs,
        ignore_index=True,
    )

    generated_set = set(
        zip(
            generated["company_id"],
            generated["employee_id"],
            generated["candidate_manager_id"],
        )
    )

    actual["found_in_candidates"] = [
        (company_id, employee_id, manager_id) in generated_set
        for company_id, employee_id, manager_id in zip(
            actual["company_id"],
            actual["employee_id"],
            actual["manager_employee_id"],
        )
    ]

    actual["reason"] = "Found"

    actual.loc[
        ~actual["manager_in_pool"],
        "reason",
    ] = "Excluded from manager pool"

    actual.loc[
        actual["manager_in_pool"] & ~actual["seniority_eligible"],
        "reason",
    ] = "Excluded by seniority filter"

    actual.loc[
        actual["seniority_eligible"]
        & actual["manager_in_pool"]
        & ~actual["found_in_candidates"],
        "reason",
    ] = "Excluded by top-10 candidate limit"

    print("\n========== UPDATED CANDIDATE DIAGNOSTICS ==========")

    print("\nTotal actual relationships:", len(actual))

    breakdown = (
        actual.groupby("reason")
        .size()
        .reset_index(name="relationships")
        .sort_values("relationships", ascending=False)
    )

    breakdown["percentage"] = (breakdown["relationships"] / len(actual) * 100).round(2)

    print("\nReason breakdown:")
    print(breakdown.to_string(index=False))

    print("\nAdditional information:")
    print(
        "Actual managers with missing titles:",
        int(actual["manager_title"].isna().sum()),
    )

    print(
        "Equal-seniority exceptions:",
        int(actual["equal_seniority_allowed"].sum()),
    )

    print(
        "Relationships found:",
        int(actual["found_in_candidates"].sum()),
    )

    print(
        "\nAll relationships accounted for:",
        int(breakdown["relationships"].sum()) == len(actual),
    )


if __name__ == "__main__":
    run_candidate_generation_diagnostics()
