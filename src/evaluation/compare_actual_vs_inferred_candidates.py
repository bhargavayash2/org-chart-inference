import pandas as pd

from src.inference.candidate_generator import (
    generate_candidates_for_company,
)

from src.inference.scoring_engine import (
    calculate_candidate_scores,
)

EMPLOYEES_PATH = "data/processed/employees_cleaned.parquet"
GROUND_TRUTH_PATH = "data/ground_truth/ground_truth_edges.parquet"
INFERRED_PATH = "data/processed/inferred_edges.parquet"

OUTPUT_PATH = "data/processed/actual_vs_inferred_candidates.csv"

MAX_CANDIDATES = 10
MISSING_TITLE_SLOTS = 2


def main():

    print("Loading datasets...")

    employees = pd.read_parquet(EMPLOYEES_PATH)
    ground_truth = pd.read_parquet(GROUND_TRUTH_PATH)
    inferred = pd.read_parquet(INFERRED_PATH)

    print("Employees:", len(employees))
    print("Ground-truth relationships:", len(ground_truth))
    print("Inferred relationships:", len(inferred))

    all_candidates = []

    companies = employees["company_id"].unique()

    print("\nRegenerating and scoring candidates...")

    for index, company_id in enumerate(companies, start=1):

        candidates = generate_candidates_for_company(
            employees,
            company_id=company_id,
            max_candidates=MAX_CANDIDATES,
            missing_title_slots=MISSING_TITLE_SLOTS,
        )

        if candidates.empty:
            continue

        scored = calculate_candidate_scores(candidates)

        # Reproduce the graph builder's candidate ordering.
        scored = scored.sort_values(
            by="confidence_score",
            ascending=False,
        )

        ranked_groups = []

        for employee_id, group in scored.groupby(
            "employee_id",
            sort=False,
        ):

            group = group.sort_values(
                by="confidence_score",
                ascending=False,
            ).copy()

            group["graph_order_rank"] = range(1, len(group) + 1)

            ranked_groups.append(group)

        if ranked_groups:
            all_candidates.append(pd.concat(ranked_groups, ignore_index=True))

        if index % 50 == 0:
            print(f"Processed {index}/{len(companies)} companies")

    if not all_candidates:
        raise ValueError("No candidates generated.")

    candidates_df = pd.concat(
        all_candidates,
        ignore_index=True,
    )

    print("\nTotal scored candidates:", len(candidates_df))

    # -----------------------------------------------------
    # Prepare candidate scoring columns.
    # -----------------------------------------------------

    possible_score_columns = [
        "confidence_score",
        "department_score",
        "seniority_score",
        "title_score",
        "manager_role_score",
        "employee_seniority_rank",
        "manager_seniority_rank",
        "candidate_type",
        "explanation",
        "graph_order_rank",
    ]

    score_columns = [
        col for col in possible_score_columns if col in candidates_df.columns
    ]

    candidate_lookup = candidates_df[
        [
            "company_id",
            "employee_id",
            "candidate_manager_id",
        ]
        + score_columns
    ].copy()

    # -----------------------------------------------------
    # Actual manager candidate.
    # -----------------------------------------------------

    actual_lookup = candidate_lookup.rename(
        columns={
            "candidate_manager_id": "manager_employee_id",
            **{col: f"actual_{col}" for col in score_columns},
        }
    )

    comparison = ground_truth.merge(
        actual_lookup,
        on=[
            "company_id",
            "employee_id",
            "manager_employee_id",
        ],
        how="left",
    )

    # -----------------------------------------------------
    # Manager selected by the inference pipeline.
    # -----------------------------------------------------

    selected_lookup = inferred[
        [
            "company_id",
            "employee_id",
            "manager_employee_id",
        ]
    ].rename(columns={"manager_employee_id": "inferred_manager_employee_id"})

    comparison = comparison.merge(
        selected_lookup,
        on=[
            "company_id",
            "employee_id",
        ],
        how="left",
    )

    # -----------------------------------------------------
    # Score of the selected manager candidate.
    # -----------------------------------------------------

    inferred_candidate_lookup = candidate_lookup.rename(
        columns={
            "candidate_manager_id": "inferred_manager_employee_id",
            **{col: f"inferred_{col}" for col in score_columns},
        }
    )

    comparison = comparison.merge(
        inferred_candidate_lookup,
        on=[
            "company_id",
            "employee_id",
            "inferred_manager_employee_id",
        ],
        how="left",
    )

    # -----------------------------------------------------
    # Add employee and manager titles/departments.
    # -----------------------------------------------------

    employee_lookup = employees[
        [
            "employee_id",
            "job_title",
            "department",
        ]
    ].rename(
        columns={
            "job_title": "employee_title",
            "department": "employee_department",
        }
    )

    comparison = comparison.merge(
        employee_lookup,
        on="employee_id",
        how="left",
    )

    manager_lookup = employees[
        [
            "employee_id",
            "job_title",
            "department",
        ]
    ].rename(
        columns={
            "employee_id": "manager_employee_id",
            "job_title": "actual_manager_title",
            "department": "actual_manager_department",
        }
    )

    comparison = comparison.merge(
        manager_lookup,
        on="manager_employee_id",
        how="left",
    )

    inferred_manager_lookup = employees[
        [
            "employee_id",
            "job_title",
            "department",
        ]
    ].rename(
        columns={
            "employee_id": "inferred_manager_employee_id",
            "job_title": "inferred_manager_title",
            "department": "inferred_manager_department",
        }
    )

    comparison = comparison.merge(
        inferred_manager_lookup,
        on="inferred_manager_employee_id",
        how="left",
    )

    # -----------------------------------------------------
    # Identify incorrect assignments.
    # -----------------------------------------------------

    comparison["correct_inference"] = (
        comparison["manager_employee_id"] == comparison["inferred_manager_employee_id"]
    )

    wrong = comparison[
        (~comparison["correct_inference"])
        & comparison["inferred_manager_employee_id"].notna()
    ].copy()

    wrong["score_difference"] = (
        wrong["actual_confidence_score"] - wrong["inferred_confidence_score"]
    )

    wrong["same_confidence_score"] = (
        wrong["actual_confidence_score"] == wrong["inferred_confidence_score"]
    )

    # -----------------------------------------------------
    # Summary.
    # -----------------------------------------------------

    print("\n========== ACTUAL VS INFERRED ==========")

    print("Incorrect assignments:", len(wrong))

    print("\nActual manager candidate rank " "(using graph ordering):")

    print(
        wrong["actual_graph_order_rank"]
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )

    print("\nActual manager confidence:")
    print(wrong["actual_confidence_score"].describe().round(2).to_string())

    print("\nInferred manager confidence:")
    print(wrong["inferred_confidence_score"].describe().round(2).to_string())

    print("\nActual minus inferred confidence:")
    print(wrong["score_difference"].describe().round(2).to_string())

    print("\nSame confidence score:")
    print(wrong["same_confidence_score"].value_counts(dropna=False).to_string())

    # -----------------------------------------------------
    # Component score comparison.
    # -----------------------------------------------------

    for component in [
        "department_score",
        "seniority_score",
        "title_score",
        "manager_role_score",
    ]:

        actual_col = f"actual_{component}"
        inferred_col = f"inferred_{component}"

        if actual_col in wrong.columns and inferred_col in wrong.columns:

            print(f"\n{component} comparison:")

            print(
                "Actual manager average:",
                round(wrong[actual_col].mean(), 3),
            )

            print(
                "Inferred manager average:",
                round(wrong[inferred_col].mean(), 3),
            )

    # -----------------------------------------------------
    # Sample relationships.
    # -----------------------------------------------------

    sample_columns = [
        "company_id",
        "employee_id",
        "employee_title",
        "employee_department",
        "manager_employee_id",
        "actual_manager_title",
        "actual_manager_department",
        "inferred_manager_employee_id",
        "inferred_manager_title",
        "inferred_manager_department",
        "actual_graph_order_rank",
        "actual_confidence_score",
        "inferred_confidence_score",
        "score_difference",
        "actual_department_score",
        "inferred_department_score",
        "actual_seniority_score",
        "inferred_seniority_score",
        "actual_title_score",
        "inferred_title_score",
        "actual_manager_role_score",
        "inferred_manager_role_score",
    ]

    available_columns = [col for col in sample_columns if col in wrong.columns]

    print("\n========== SAMPLE COMPARISONS ==========")

    print(wrong[available_columns].head(30).to_string(index=False))

    # -----------------------------------------------------
    # Save report.
    # -----------------------------------------------------

    wrong.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nDetailed report saved to:")
    print(OUTPUT_PATH)

    print("\n========================================")


if __name__ == "__main__":
    main()
