import pandas as pd

from src.inference.candidate_generator import generate_candidates_for_company
from src.inference.scoring_engine import calculate_candidate_scores


def main():

    employees = pd.read_parquet("data/processed/employees_cleaned.parquet")

    ground_truth = pd.read_parquet("data/ground_truth/ground_truth_edges.parquet")

    inferred = pd.read_parquet("data/processed/inferred_edges.parquet")

    # Detect the manager ID column in inferred edges.
    possible_manager_columns = [
        "manager_employee_id",
        "candidate_manager_id",
    ]

    manager_column = next(
        (col for col in possible_manager_columns if col in inferred.columns),
        None,
    )

    if manager_column is None:
        raise ValueError(
            f"Cannot identify manager column. "
            f"Inferred columns: {inferred.columns.tolist()}"
        )

    print("Generating and scoring candidates for all companies...")

    scored_parts = []

    for company_id in employees["company_id"].unique():

        candidates = generate_candidates_for_company(
            employees,
            company_id,
            max_candidates=10,
            missing_title_slots=2,
        )

        if not candidates.empty:
            scored = calculate_candidate_scores(candidates)
            scored_parts.append(scored)

    all_scored = pd.concat(
        scored_parts,
        ignore_index=True,
    )

    # Match each actual manager against the generated candidates.
    actual_candidates = ground_truth.merge(
        all_scored[
            [
                "employee_id",
                "candidate_manager_id",
                "confidence_score",
                "candidate_rank",
                "candidate_type",
            ]
        ],
        left_on=["employee_id", "manager_employee_id"],
        right_on=["employee_id", "candidate_manager_id"],
        how="left",
    )

    actual_candidates["candidate_found"] = actual_candidates["confidence_score"].notna()

    # Check whether the actual manager was selected by inference.
    inferred_lookup = inferred[["employee_id", manager_column]].drop_duplicates(
        "employee_id"
    )

    # Rename the inferred manager column to avoid conflicts
    # with the ground-truth manager_employee_id column.
    inferred_lookup = inferred_lookup.rename(
        columns={manager_column: "inferred_manager_employee_id"}
    )

    actual_candidates = actual_candidates.merge(
        inferred_lookup,
        on="employee_id",
        how="left",
    )

    actual_candidates["correctly_inferred"] = (
        actual_candidates["manager_employee_id"]
        == actual_candidates["inferred_manager_employee_id"]
    )

    actual_candidates["failure_category"] = "Candidate generation failure"

    actual_candidates.loc[
        actual_candidates["candidate_found"]
        & (actual_candidates["confidence_score"] < 65),
        "failure_category",
    ] = "Scoring below threshold"

    actual_candidates.loc[
        actual_candidates["candidate_found"]
        & (actual_candidates["confidence_score"] >= 65)
        & ~actual_candidates["correctly_inferred"],
        "failure_category",
    ] = "Ranking or graph construction failure"

    actual_candidates.loc[
        actual_candidates["correctly_inferred"],
        "failure_category",
    ] = "Correctly inferred"

    print("\n========== ACTUAL MANAGER DIAGNOSTICS ==========")

    print("Actual relationships:", len(actual_candidates))

    print("\nFailure category:")
    print(actual_candidates["failure_category"].value_counts().to_string())

    print("\nCandidate inclusion:")
    print(
        actual_candidates["candidate_found"]
        .value_counts()
        .rename(
            {
                True: "Actual manager found in candidates",
                False: "Actual manager missing from candidates",
            }
        )
        .to_string()
    )

    found = actual_candidates[actual_candidates["candidate_found"]]

    print("\nActual manager candidate rank:")
    print(found["candidate_rank"].value_counts().sort_index().to_string())

    print("\nActual manager confidence score:")
    print(found["confidence_score"].describe().round(2).to_string())

    print("\n========== DISCONNECTED SENIOR EMPLOYEES ==========")

    disconnected = actual_candidates[~actual_candidates["correctly_inferred"]]

    print("\nEmployee title:")
    print(
        disconnected["employee_id"]
        .map(employees.set_index("employee_id")["job_title"])
        .value_counts()
        .head(20)
        .to_string()
    )

    print("\nSample actual managers scoring below 65:")

    print(
        actual_candidates[
            actual_candidates["candidate_found"]
            & (actual_candidates["confidence_score"] < 65)
        ][
            [
                "company_id",
                "employee_id",
                "manager_employee_id",
                "confidence_score",
                "candidate_rank",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print("\nDiagnostics completed.")


if __name__ == "__main__":
    main()
