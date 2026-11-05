from pathlib import Path

import pandas as pd

from src.inference.candidate_generator import (
    generate_candidates_for_company,
)

from src.inference.scoring_engine import (
    calculate_candidate_scores,
)

COMPONENT_COLUMNS = [
    "department_score",
    "seniority_score",
    "title_score",
    "manager_role_score",
]

OUTPUT_DIR = Path("data/processed")


def get_top_candidate_diagnostics(scored):
    """
    Analyze each employee's highest-scoring candidates.

    Tie categories:
    1. identical_evidence:
       All tied candidates have identical component scores.

    2. aggregate_score_collision:
       Confidence scores are equal, but component scores differ.
    """

    if scored.empty:
        return pd.DataFrame()

    highest_scores = scored.groupby("employee_id")["confidence_score"].transform("max")

    top_candidates = scored[scored["confidence_score"] == highest_scores].copy()

    records = []

    for employee_id, group in top_candidates.groupby("employee_id"):

        if len(group) <= 1:
            continue

        component_variations = {
            column: group[column].nunique(dropna=False) for column in COMPONENT_COLUMNS
        }

        identical_evidence = all(count == 1 for count in component_variations.values())

        actual_manager = group["actual_manager"].iloc[0]

        actual_manager_is_top = (group["candidate_manager_id"] == actual_manager).any()

        records.append(
            {
                "employee_id": employee_id,
                "top_score": group["confidence_score"].iloc[0],
                "top_candidate_count": len(group),
                "tie_type": (
                    "identical_evidence"
                    if identical_evidence
                    else "aggregate_score_collision"
                ),
                "actual_manager_is_top_candidate": actual_manager_is_top,
                "top_candidate_ids": ", ".join(
                    group["candidate_manager_id"].astype(str)
                ),
            }
        )

    return pd.DataFrame(records)


def get_changed_assignment_diagnostics(
    scored,
    assignments,
):
    """
    Compare baseline and experimental assignments.

    Checks whether changed assignments:
    - Have equal confidence scores.
    - Have identical component evidence.
    - Have different component evidence but equal aggregate scores.
    - Match the actual manager.
    """

    score_lookup = scored[
        [
            "employee_id",
            "candidate_manager_id",
            "confidence_score",
        ]
        + COMPONENT_COLUMNS
    ].drop_duplicates(["employee_id", "candidate_manager_id"])

    baseline_lookup = score_lookup.rename(
        columns={
            "candidate_manager_id": "baseline_manager",
            "confidence_score": "baseline_score",
            **{column: f"baseline_{column}" for column in COMPONENT_COLUMNS},
        }
    )

    experimental_lookup = score_lookup.rename(
        columns={
            "candidate_manager_id": "experimental_manager",
            "confidence_score": "experimental_score",
            **{column: f"experimental_{column}" for column in COMPONENT_COLUMNS},
        }
    )

    comparison = assignments.merge(
        baseline_lookup,
        on=["employee_id", "baseline_manager"],
        how="left",
    ).merge(
        experimental_lookup,
        on=["employee_id", "experimental_manager"],
        how="left",
    )

    changed = comparison[
        comparison["baseline_manager"].notna()
        & comparison["experimental_manager"].notna()
        & (comparison["baseline_manager"] != comparison["experimental_manager"])
    ].copy()

    if changed.empty:
        return changed

    changed["score_difference"] = (
        changed["experimental_score"] - changed["baseline_score"]
    )

    changed["scores_equal"] = changed["score_difference"] == 0

    component_comparisons = []

    for column in COMPONENT_COLUMNS:

        baseline_column = f"baseline_{column}"
        experimental_column = f"experimental_{column}"

        equal = changed[baseline_column].eq(changed[experimental_column]) | (
            changed[baseline_column].isna() & changed[experimental_column].isna()
        )

        component_comparisons.append(equal)

    changed["identical_component_evidence"] = pd.concat(
        component_comparisons,
        axis=1,
    ).all(axis=1)

    changed["tie_type"] = "different_confidence_scores"

    equal_score = changed["scores_equal"]

    changed.loc[
        equal_score & changed["identical_component_evidence"],
        "tie_type",
    ] = "identical_evidence"

    changed.loc[
        equal_score & ~changed["identical_component_evidence"],
        "tie_type",
    ] = "aggregate_score_collision"

    changed["baseline_is_correct"] = (
        changed["baseline_manager"] == changed["actual_manager"]
    )

    changed["experimental_is_correct"] = (
        changed["experimental_manager"] == changed["actual_manager"]
    )

    return changed


def main():

    print("Loading employee and assignment data...")

    employees = pd.read_parquet("data/processed/employees_cleaned.parquet")

    baseline = pd.read_parquet("data/processed/inferred_edges.parquet")

    experimental = pd.read_parquet("data/processed/experimental_inferred_edges.parquet")

    ground_truth = pd.read_parquet("data/ground_truth/ground_truth_edges.parquet")

    company_ids = sorted(employees["company_id"].dropna().unique())

    print("Total companies:", len(company_ids))
    print("Total employees:", len(employees))

    all_tie_details = []
    all_changed_details = []
    company_summaries = []

    for index, company_id in enumerate(company_ids, start=1):

        print(f"\nAnalyzing company {index}/{len(company_ids)}: " f"{company_id}")

        company_employees = employees[employees["company_id"] == company_id]

        baseline_company = baseline[baseline["company_id"] == company_id]

        experimental_company = experimental[experimental["company_id"] == company_id]

        actual_company = ground_truth[ground_truth["company_id"] == company_id]

        # --------------------------------------------------
        # 1. Combine all three assignment datasets
        # --------------------------------------------------

        assignments = (
            baseline_company[["employee_id", "manager_employee_id"]]
            .rename(columns={"manager_employee_id": "baseline_manager"})
            .merge(
                experimental_company[["employee_id", "manager_employee_id"]].rename(
                    columns={"manager_employee_id": "experimental_manager"}
                ),
                on="employee_id",
                how="outer",
            )
            .merge(
                actual_company[["employee_id", "manager_employee_id"]].rename(
                    columns={"manager_employee_id": "actual_manager"}
                ),
                on="employee_id",
                how="left",
            )
        )

        # --------------------------------------------------
        # 2. Generate and score candidates
        # --------------------------------------------------

        candidates = generate_candidates_for_company(
            employees,
            company_id=company_id,
            max_candidates=10,
            missing_title_slots=2,
        )

        scored = calculate_candidate_scores(candidates)

        scored = scored.merge(
            assignments[["employee_id", "actual_manager"]],
            on="employee_id",
            how="left",
        )

        # --------------------------------------------------
        # 3. Analyze top candidate ties
        # --------------------------------------------------

        tie_details = get_top_candidate_diagnostics(scored)

        if not tie_details.empty:

            tie_details.insert(
                0,
                "company_id",
                company_id,
            )

            all_tie_details.append(tie_details)

        employee_count = len(company_employees)

        employees_with_candidates = (
            scored["employee_id"].nunique() if not scored.empty else 0
        )

        tied_count = len(tie_details)

        identical_evidence_count = (
            int((tie_details["tie_type"] == "identical_evidence").sum())
            if not tie_details.empty
            else 0
        )

        aggregate_collision_count = (
            int((tie_details["tie_type"] == "aggregate_score_collision").sum())
            if not tie_details.empty
            else 0
        )

        actual_in_tie_count = (
            int(tie_details["actual_manager_is_top_candidate"].sum())
            if not tie_details.empty
            else 0
        )

        # --------------------------------------------------
        # 4. Analyze changed baseline/experimental assignments
        # --------------------------------------------------

        changed = get_changed_assignment_diagnostics(
            scored,
            assignments,
        )

        if not changed.empty:

            changed.insert(
                0,
                "company_id",
                company_id,
            )

            all_changed_details.append(changed)

        changed_count = len(changed)

        changed_equal_score = int(changed["scores_equal"].sum()) if changed_count else 0

        changed_identical_evidence = (
            int((changed["tie_type"] == "identical_evidence").sum())
            if changed_count
            else 0
        )

        changed_aggregate_collision = (
            int((changed["tie_type"] == "aggregate_score_collision").sum())
            if changed_count
            else 0
        )

        baseline_correct = (
            int(changed["baseline_is_correct"].sum()) if changed_count else 0
        )

        experimental_correct = (
            int(changed["experimental_is_correct"].sum()) if changed_count else 0
        )

        company_summaries.append(
            {
                "company_id": company_id,
                "employee_count": employee_count,
                "employees_with_candidates": employees_with_candidates,
                "employees_with_top_ties": tied_count,
                "employees_with_candidates_without_top_ties": (
                    employees_with_candidates - tied_count
                ),
                "identical_evidence_ties": identical_evidence_count,
                "aggregate_score_collision_ties": aggregate_collision_count,
                "actual_manager_in_top_tie": actual_in_tie_count,
                "changed_assignments": changed_count,
                "changed_equal_score": changed_equal_score,
                "changed_identical_evidence": changed_identical_evidence,
                "changed_aggregate_collision": changed_aggregate_collision,
                "baseline_correct_among_changed": baseline_correct,
                "experimental_correct_among_changed": experimental_correct,
            }
        )

    # ------------------------------------------------------
    # 5. Combine results across all companies
    # ------------------------------------------------------

    company_summary = pd.DataFrame(company_summaries)

    tie_details_all = (
        pd.concat(all_tie_details, ignore_index=True)
        if all_tie_details
        else pd.DataFrame()
    )

    changed_details_all = (
        pd.concat(all_changed_details, ignore_index=True)
        if all_changed_details
        else pd.DataFrame()
    )

    # ------------------------------------------------------
    # 6. Print overall tie statistics
    # ------------------------------------------------------

    total_employees = len(employees)

    total_with_candidates = int(company_summary["employees_with_candidates"].sum())

    total_ties = int(company_summary["employees_with_top_ties"].sum())

    total_identical = int(company_summary["identical_evidence_ties"].sum())

    total_collisions = int(company_summary["aggregate_score_collision_ties"].sum())

    total_actual_in_ties = int(company_summary["actual_manager_in_top_tie"].sum())

    print("\n" + "=" * 75)
    print("FULL DATASET TOP-CANDIDATE TIE ANALYSIS")
    print("=" * 75)

    print("Companies analyzed:", len(company_summary))
    print("Total employees:", total_employees)
    print("Employees with candidates:", total_with_candidates)
    print("Employees with tied top candidates:", total_ties)

    print(
        "Percentage with top ties:",
        (
            round(
                total_ties / total_with_candidates * 100,
                2,
            )
            if total_with_candidates
            else 0
        ),
        "%",
    )

    print("\nTie classification:")

    print(
        "Identical component evidence:",
        total_identical,
    )

    print(
        "Aggregate-score collisions:",
        total_collisions,
    )

    print(
        "Actual manager present among tied top candidates:",
        total_actual_in_ties,
    )

    print(
        "Actual manager coverage within tied top candidates:",
        (
            round(
                total_actual_in_ties / total_ties * 100,
                2,
            )
            if total_ties
            else 0
        ),
        "%",
    )

    # ------------------------------------------------------
    # 7. Print changed-assignment analysis
    # ------------------------------------------------------

    total_changed = len(changed_details_all)

    print("\n" + "=" * 75)
    print("BASELINE VS EXPERIMENTAL CHANGED ASSIGNMENTS")
    print("=" * 75)

    print("Changed assignments:", total_changed)

    if total_changed:

        print(
            "Equal confidence scores:",
            int(changed_details_all["scores_equal"].sum()),
        )

        print("\nChanged assignment tie classification:")

        print(changed_details_all["tie_type"].value_counts(dropna=False).to_string())

        print(
            "\nBaseline correct among changed assignments:",
            int(changed_details_all["baseline_is_correct"].sum()),
        )

        print(
            "Experimental correct among changed assignments:",
            int(changed_details_all["experimental_is_correct"].sum()),
        )

    # ------------------------------------------------------
    # 8. Show company-level summary
    # ------------------------------------------------------

    print("\n" + "=" * 75)
    print("COMPANY-LEVEL SUMMARY")
    print("=" * 75)

    print(
        company_summary[
            [
                "company_id",
                "employee_count",
                "employees_with_top_ties",
                "identical_evidence_ties",
                "aggregate_score_collision_ties",
                "changed_assignments",
                "changed_identical_evidence",
                "changed_aggregate_collision",
            ]
        ].to_string(index=False)
    )

    # ------------------------------------------------------
    # 9. Save diagnostic reports
    # ------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    company_summary.to_csv(
        OUTPUT_DIR / "full_dataset_tie_company_summary.csv",
        index=False,
    )

    tie_details_all.to_csv(
        OUTPUT_DIR / "full_dataset_tie_details.csv",
        index=False,
    )

    changed_details_all.to_csv(
        OUTPUT_DIR / "full_dataset_changed_assignment_ties.csv",
        index=False,
    )

    print("\nReports saved:")

    print(OUTPUT_DIR / "full_dataset_tie_company_summary.csv")

    print(OUTPUT_DIR / "full_dataset_tie_details.csv")

    print(OUTPUT_DIR / "full_dataset_changed_assignment_ties.csv")

    print("\nFull-dataset tie analysis completed.")


if __name__ == "__main__":
    main()
