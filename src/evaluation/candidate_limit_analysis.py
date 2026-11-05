import pandas as pd

from src.inference.candidate_generator import generate_candidates_for_company

from src.inference.scoring_engine import calculate_candidate_scores


def evaluate_candidate_limit(employees_df, ground_truth_df, candidate_limit):
    all_actual_relationships = 0
    all_candidate_pairs = 0
    all_found = 0

    top_1_count = 0
    top_3_count = 0
    top_5_count = 0

    company_ids = ground_truth_df["company_id"].drop_duplicates()

    for index, company_id in enumerate(company_ids, start=1):

        print(f"Limit {candidate_limit}: " f"Company {index}/{len(company_ids)}")

        company_ground_truth = ground_truth_df[
            ground_truth_df["company_id"] == company_id
        ]

        actual_relationships = len(company_ground_truth)

        candidates = generate_candidates_for_company(
            employees_df, company_id=company_id, max_candidates=candidate_limit
        )

        scored = calculate_candidate_scores(candidates)

        all_actual_relationships += actual_relationships
        all_candidate_pairs += len(candidates)

        if scored.empty:
            continue

        matched = company_ground_truth.merge(
            scored[
                ["company_id", "employee_id", "candidate_manager_id", "candidate_rank"]
            ],
            left_on=["company_id", "employee_id", "manager_employee_id"],
            right_on=["company_id", "employee_id", "candidate_manager_id"],
            how="inner",
        )

        all_found += len(matched)

        top_1_count += (matched["candidate_rank"] <= 1).sum()

        top_3_count += (matched["candidate_rank"] <= 3).sum()

        top_5_count += (matched["candidate_rank"] <= 5).sum()

    return {
        "candidate_limit": candidate_limit,
        "total_candidate_pairs": all_candidate_pairs,
        "average_candidates_per_employee": (all_candidate_pairs / len(employees_df)),
        "candidate_recall": (all_found / all_actual_relationships),
        "top_1_recall": (top_1_count / all_actual_relationships),
        "top_3_recall": (top_3_count / all_actual_relationships),
        "top_5_recall": (top_5_count / all_actual_relationships),
    }


if __name__ == "__main__":

    employees = pd.read_parquet("data/processed/employees_cleaned.parquet")

    ground_truth = pd.read_parquet("data/ground_truth/ground_truth_edges.parquet")

    candidate_limits = [5, 10, 15, 20, 30, 50]

    results = []

    for limit in candidate_limits:

        print(f"\n========== TESTING LIMIT {limit} ==========")

        result = evaluate_candidate_limit(
            employees, ground_truth, candidate_limit=limit
        )

        results.append(result)

    results_df = pd.DataFrame(results)

    results_df.to_parquet(
        "data/processed/candidate_limit_analysis.parquet", index=False
    )

    print("\n========== CANDIDATE LIMIT ANALYSIS ==========")

    print(
        results_df.to_string(
            index=False,
            formatters={
                "average_candidates_per_employee": "{:.2f}".format,
                "candidate_recall": "{:.4f}".format,
                "top_1_recall": "{:.4f}".format,
                "top_3_recall": "{:.4f}".format,
                "top_5_recall": "{:.4f}".format,
            },
        )
    )

    print("\nSaved: data/processed/candidate_limit_analysis.parquet")
