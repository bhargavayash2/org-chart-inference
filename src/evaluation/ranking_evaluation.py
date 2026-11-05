import pandas as pd

from src.inference.candidate_generator import generate_candidates_for_company

from src.inference.scoring_engine import calculate_candidate_scores


def evaluate_candidate_ranking(employees_df, ground_truth_df, max_candidates=10):
    """
    Evaluate where the actual manager appears in the
    scored candidate rankings.

    Employees whose actual managers are absent from
    candidate generation are counted as ranking misses.
    """

    company_results = []

    company_ids = ground_truth_df["company_id"].drop_duplicates()

    for index, company_id in enumerate(company_ids, start=1):

        print(f"Evaluating ranking {index}/{len(company_ids)}: " f"{company_id}")

        company_ground_truth = ground_truth_df[
            ground_truth_df["company_id"] == company_id
        ]

        candidates = generate_candidates_for_company(
            employees_df, company_id=company_id, max_candidates=max_candidates
        )

        scored_candidates = calculate_candidate_scores(candidates)

        actual_relationships = len(company_ground_truth)

        # Match actual manager relationships against
        # the generated and scored candidate relationships.
        if not scored_candidates.empty:

            matched = company_ground_truth.merge(
                scored_candidates[
                    [
                        "company_id",
                        "employee_id",
                        "candidate_manager_id",
                        "candidate_rank",
                        "confidence_score",
                    ]
                ],
                left_on=["company_id", "employee_id", "manager_employee_id"],
                right_on=["company_id", "employee_id", "candidate_manager_id"],
                how="inner",
            )

        else:
            matched = pd.DataFrame(columns=["candidate_rank", "confidence_score"])

        found_count = len(matched)

        top_1 = (matched["candidate_rank"] <= 1).sum()

        top_3 = (matched["candidate_rank"] <= 3).sum()

        top_5 = (matched["candidate_rank"] <= 5).sum()

        top_10 = (matched["candidate_rank"] <= 10).sum()

        company_results.append(
            {
                "company_id": company_id,
                "actual_relationships": actual_relationships,
                "actual_manager_found": found_count,
                "actual_manager_missing": (actual_relationships - found_count),
                "top_1_count": int(top_1),
                "top_3_count": int(top_3),
                "top_5_count": int(top_5),
                "top_10_count": int(top_10),
                "top_1_recall": top_1 / actual_relationships,
                "top_3_recall": top_3 / actual_relationships,
                "top_5_recall": top_5 / actual_relationships,
                "top_10_recall": top_10 / actual_relationships,
            }
        )

    return pd.DataFrame(company_results)


if __name__ == "__main__":

    employees = pd.read_parquet("data/processed/employees_cleaned.parquet")

    ground_truth = pd.read_parquet("data/ground_truth/ground_truth_edges.parquet")

    results = evaluate_candidate_ranking(employees, ground_truth, max_candidates=10)

    results.to_parquet("data/processed/ranking_evaluation.parquet", index=False)

    # Aggregate counts across all companies.
    total_actual = results["actual_relationships"].sum()

    total_found = results["actual_manager_found"].sum()

    total_top_1 = results["top_1_count"].sum()
    total_top_3 = results["top_3_count"].sum()
    total_top_5 = results["top_5_count"].sum()
    total_top_10 = results["top_10_count"].sum()

    print("\n========== CANDIDATE RANKING EVALUATION ==========")

    print("Actual relationships:", total_actual)
    print("Actual managers found:", total_found)
    print("Actual managers missing:", total_actual - total_found)

    print(f"Top-1 Recall: {total_top_1 / total_actual:.4f}")

    print(f"Top-3 Recall: {total_top_3 / total_actual:.4f}")

    print(f"Top-5 Recall: {total_top_5 / total_actual:.4f}")

    print(f"Top-10 Recall: {total_top_10 / total_actual:.4f}")

    print("\nCompany-level results:")

    print(
        results[
            [
                "company_id",
                "top_1_recall",
                "top_3_recall",
                "top_5_recall",
                "top_10_recall",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\nCompanies evaluated:", len(results))

    print("\nSaved: data/processed/ranking_evaluation.parquet")
