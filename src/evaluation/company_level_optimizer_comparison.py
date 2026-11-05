from pathlib import Path

import pandas as pd

BASELINE_PATH = Path("data/processed/inferred_edges.parquet")
EXPERIMENTAL_PATH = Path("data/processed/experimental_inferred_edges.parquet")
GROUND_TRUTH_PATH = Path("data/ground_truth/ground_truth_edges.parquet")

OUTPUT_PATH = Path("data/processed/company_level_optimizer_comparison.csv")


def calculate_company_metrics(inferred, ground_truth):
    inferred_groups = {
        company_id: set(zip(group["employee_id"], group["manager_employee_id"]))
        for company_id, group in inferred.groupby("company_id")
    }

    actual_groups = {
        company_id: set(zip(group["employee_id"], group["manager_employee_id"]))
        for company_id, group in ground_truth.groupby("company_id")
    }

    results = []

    for company_id, actual_edges in actual_groups.items():
        predicted_edges = inferred_groups.get(company_id, set())

        tp = len(predicted_edges & actual_edges)
        fp = len(predicted_edges - actual_edges)
        fn = len(actual_edges - predicted_edges)

        precision = tp / len(predicted_edges) if predicted_edges else 0
        recall = tp / len(actual_edges) if actual_edges else 0

        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

        results.append(
            {
                "company_id": company_id,
                "ground_truth_count": len(actual_edges),
                "inferred_count": len(predicted_edges),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            }
        )

    return pd.DataFrame(results)


def main():
    print("Loading datasets...")

    baseline = pd.read_parquet(BASELINE_PATH)
    experimental = pd.read_parquet(EXPERIMENTAL_PATH)
    ground_truth = pd.read_parquet(GROUND_TRUTH_PATH)

    print("Calculating baseline company metrics...")

    baseline_metrics = calculate_company_metrics(baseline, ground_truth)

    print("Calculating experimental company metrics...")

    experimental_metrics = calculate_company_metrics(experimental, ground_truth)

    baseline_metrics = baseline_metrics.add_suffix("_baseline")
    experimental_metrics = experimental_metrics.add_suffix("_experimental")

    comparison = baseline_metrics.merge(
        experimental_metrics,
        left_on="company_id_baseline",
        right_on="company_id_experimental",
    )

    comparison["company_id"] = comparison["company_id_baseline"]

    comparison["f1_change"] = (
        comparison["f1_score_experimental"] - comparison["f1_score_baseline"]
    )

    comparison["precision_change"] = (
        comparison["precision_experimental"] - comparison["precision_baseline"]
    )

    comparison["recall_change"] = (
        comparison["recall_experimental"] - comparison["recall_baseline"]
    )

    comparison["result"] = comparison["f1_change"].apply(
        lambda value: (
            "Improved"
            if value > 0.000001
            else "Regressed" if value < -0.000001 else "Unchanged"
        )
    )

    print("\n========== COMPANY-LEVEL RESULTS ==========")

    print(
        comparison["result"]
        .value_counts()
        .reindex(
            ["Improved", "Unchanged", "Regressed"],
            fill_value=0,
        )
        .to_string()
    )

    print("\n========== AVERAGE COMPANY METRICS ==========")

    print(
        comparison[
            [
                "precision_baseline",
                "precision_experimental",
                "recall_baseline",
                "recall_experimental",
                "f1_score_baseline",
                "f1_score_experimental",
            ]
        ]
        .mean()
        .to_string()
    )

    print("\n========== TOP 10 IMPROVEMENTS ==========")

    improvements = comparison.nlargest(10, "f1_change")

    print(
        improvements[
            [
                "company_id",
                "f1_score_baseline",
                "f1_score_experimental",
                "f1_change",
            ]
        ].to_string(index=False)
    )

    print("\n========== TOP 10 REGRESSIONS ==========")

    regressions = comparison.nsmallest(10, "f1_change")

    print(
        regressions[
            [
                "company_id",
                "f1_score_baseline",
                "f1_score_experimental",
                "f1_change",
            ]
        ].to_string(index=False)
    )

    comparison = comparison.sort_values("f1_change")

    comparison.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved detailed results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
