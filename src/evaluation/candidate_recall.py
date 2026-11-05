import pandas as pd

from src.inference.candidate_generator import generate_candidates_for_company


def evaluate_candidate_recall(employees_df, ground_truth_df, max_candidates=10):
    """
    Measure how often the actual manager appears
    in the generated candidate list.

    This evaluates candidate generation independently
    of scoring and graph construction.
    """

    results = []

    company_ids = ground_truth_df["company_id"].drop_duplicates()

    for index, company_id in enumerate(company_ids, start=1):

        print(f"Evaluating candidates for " f"{index}/{len(company_ids)}: {company_id}")

        company_ground_truth = ground_truth_df[
            ground_truth_df["company_id"] == company_id
        ]

        candidates = generate_candidates_for_company(
            employees_df, company_id=company_id, max_candidates=max_candidates
        )

        # Actual employee-manager pairs.
        actual_pairs = set(
            zip(
                company_ground_truth["employee_id"],
                company_ground_truth["manager_employee_id"],
            )
        )

        # Generated employee-manager candidate pairs.
        candidate_pairs = set()

        if not candidates.empty:
            candidate_pairs = set(
                zip(candidates["employee_id"], candidates["candidate_manager_id"])
            )

        # Actual managers successfully included as candidates.
        matched_pairs = actual_pairs.intersection(candidate_pairs)

        missed_pairs = actual_pairs - candidate_pairs

        actual_relationship_count = len(actual_pairs)
        matched_count = len(matched_pairs)

        candidate_recall = (
            matched_count / actual_relationship_count
            if actual_relationship_count > 0
            else 0.0
        )

        results.append(
            {
                "company_id": company_id,
                "ground_truth_relationships": actual_relationship_count,
                "candidate_pairs": len(candidate_pairs),
                "actual_managers_found": matched_count,
                "actual_managers_missing": len(missed_pairs),
                "candidate_recall": candidate_recall,
            }
        )

    return pd.DataFrame(results)


if __name__ == "__main__":

    employees = pd.read_parquet("data/processed/employees_cleaned.parquet")

    ground_truth = pd.read_parquet("data/ground_truth/ground_truth_edges.parquet")

    results = evaluate_candidate_recall(employees, ground_truth, max_candidates=10)

    # Save company-level results.
    results.to_parquet("data/processed/candidate_recall.parquet", index=False)

    total_actual = results["ground_truth_relationships"].sum()

    total_found = results["actual_managers_found"].sum()

    overall_recall = total_found / total_actual if total_actual > 0 else 0.0

    print("\n========== CANDIDATE RECALL ==========")

    print("Ground-truth relationships:", total_actual)
    print("Actual managers found:", total_found)

    print("Actual managers missing:", results["actual_managers_missing"].sum())

    print(f"Candidate Generation Recall: " f"{overall_recall:.4f}")

    print("\nCompany-level results:")

    print(
        results[["company_id", "candidate_recall", "actual_managers_missing"]]
        .head(10)
        .to_string(index=False)
    )

    print("\nCompanies evaluated:", len(results))

    print("\nSaved: data/processed/candidate_recall.parquet")
