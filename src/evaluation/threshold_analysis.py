import pandas as pd

from src.inference.candidate_generator import generate_candidates_for_company

from src.inference.scoring_engine import calculate_candidate_scores

from src.evaluation.metrics import calculate_metrics


def generate_all_scored_candidates(employees_df, max_candidates=10):
    """
    Generate and score candidates for every company.
    """

    all_candidates = []

    company_ids = employees_df["company_id"].dropna().unique()

    for index, company_id in enumerate(company_ids, start=1):

        print(f"Scoring company {index}/{len(company_ids)}: " f"{company_id}")

        candidates = generate_candidates_for_company(
            employees_df, company_id=company_id, max_candidates=max_candidates
        )

        if candidates.empty:
            continue

        scored = calculate_candidate_scores(candidates)

        all_candidates.append(scored)

    if not all_candidates:
        return pd.DataFrame()

    return pd.concat(all_candidates, ignore_index=True)


def evaluate_confidence_thresholds(scored_candidates, ground_truth_df, thresholds):
    """
    Evaluate the top-ranked candidate at different
    confidence thresholds.

    This isolates scoring performance from graph
    cycle prevention and other graph constraints.
    """

    if scored_candidates.empty:
        raise ValueError("No scored candidates available.")

    # Keep only the highest-ranked manager candidate
    # for each employee.
    top_candidates = scored_candidates[scored_candidates["candidate_rank"] == 1].copy()

    # Convert candidate output into the relationship
    # format expected by calculate_metrics().
    top_candidates = top_candidates.rename(
        columns={"candidate_manager_id": "manager_employee_id"}
    )

    results = []

    for threshold in thresholds:

        # Apply confidence threshold.
        predictions = top_candidates[top_candidates["confidence_score"] >= threshold][
            ["company_id", "employee_id", "manager_employee_id"]
        ].copy()

        metrics = calculate_metrics(ground_truth_df, predictions)

        results.append(
            {
                "confidence_threshold": threshold,
                "predicted_relationships": len(predictions),
                "true_positives": metrics["true_positives"],
                "false_positives": metrics["false_positives"],
                "false_negatives": metrics["false_negatives"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "assignment_coverage": (metrics["assignment_coverage"]),
            }
        )

    return pd.DataFrame(results)


if __name__ == "__main__":

    employees = pd.read_parquet("data/processed/employees_cleaned.parquet")

    ground_truth = pd.read_parquet("data/ground_truth/ground_truth_edges.parquet")

    # Generate and score candidates once.
    scored_candidates = generate_all_scored_candidates(employees, max_candidates=10)

    print("\nTotal scored candidate pairs:", len(scored_candidates))

    # Test multiple confidence thresholds.
    thresholds = [40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

    results = evaluate_confidence_thresholds(
        scored_candidates, ground_truth, thresholds
    )

    # Save results.
    results.to_parquet("data/processed/threshold_analysis.parquet", index=False)

    # Identify the threshold with the highest F1.
    best_result = results.loc[results["f1_score"].idxmax()]

    print("\n========== THRESHOLD ANALYSIS ==========")

    print(
        results.to_string(
            index=False,
            formatters={
                "precision": "{:.4f}".format,
                "recall": "{:.4f}".format,
                "f1_score": "{:.4f}".format,
                "assignment_coverage": "{:.4f}".format,
            },
        )
    )

    print("\n========== BEST F1 THRESHOLD ==========")

    print("Threshold:", best_result["confidence_threshold"])

    print(f"Precision: {best_result['precision']:.4f}")

    print(f"Recall: {best_result['recall']:.4f}")

    print(f"F1-score: {best_result['f1_score']:.4f}")

    print("\nSaved: data/processed/threshold_analysis.parquet")
