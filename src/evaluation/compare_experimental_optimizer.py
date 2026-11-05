from pathlib import Path

import pandas as pd

BASELINE_PATH = Path("data/processed/inferred_edges.parquet")
EXPERIMENTAL_PATH = Path("data/processed/experimental_inferred_edges.parquet")
GROUND_TRUTH_PATH = Path("data/ground_truth/ground_truth_edges.parquet")


def load_data():
    print("Loading baseline, experimental, and ground truth data...")

    baseline = pd.read_parquet(BASELINE_PATH)
    experimental = pd.read_parquet(EXPERIMENTAL_PATH)
    ground_truth = pd.read_parquet(GROUND_TRUTH_PATH)

    required_columns = {
        "company_id",
        "employee_id",
        "manager_employee_id",
    }

    for name, df in [
        ("Baseline", baseline),
        ("Experimental", experimental),
        ("Ground truth", ground_truth),
    ]:
        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(f"{name} is missing required columns: {missing}")

        if df.duplicated(["company_id", "employee_id"]).any():
            raise ValueError(
                f"{name} contains multiple manager assignments "
                "for the same employee."
            )

    return baseline, experimental, ground_truth


def calculate_metrics(name, inferred, ground_truth):
    inferred_edges = set(
        zip(
            inferred["company_id"],
            inferred["employee_id"],
            inferred["manager_employee_id"],
        )
    )

    actual_edges = set(
        zip(
            ground_truth["company_id"],
            ground_truth["employee_id"],
            ground_truth["manager_employee_id"],
        )
    )

    true_positives = len(inferred_edges & actual_edges)
    false_positives = len(inferred_edges - actual_edges)
    false_negatives = len(actual_edges - inferred_edges)

    precision = true_positives / len(inferred_edges) if inferred_edges else 0

    recall = true_positives / len(actual_edges) if actual_edges else 0

    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    coverage = len(inferred_edges) / len(actual_edges) if actual_edges else 0

    return {
        "Pipeline": name,
        "Inferred Relationships": len(inferred_edges),
        "True Positives": true_positives,
        "False Positives": false_positives,
        "False Negatives": false_negatives,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Assignment Coverage": coverage,
    }


def compare_assignments(baseline, experimental):
    baseline_assignments = baseline[
        ["company_id", "employee_id", "manager_employee_id"]
    ].rename(columns={"manager_employee_id": "baseline_manager"})

    experimental_assignments = experimental[
        ["company_id", "employee_id", "manager_employee_id"]
    ].rename(columns={"manager_employee_id": "experimental_manager"})

    comparison = baseline_assignments.merge(
        experimental_assignments,
        on=["company_id", "employee_id"],
        how="outer",
        indicator=True,
    )

    both = comparison[comparison["_merge"] == "both"]

    same_manager = both["baseline_manager"] == both["experimental_manager"]

    changed = both[~same_manager]

    print("\n========== ASSIGNMENT COMPARISON ==========")
    print(f"Employees assigned in both pipelines: {len(both):,}")
    print(f"Same manager selected: {same_manager.sum():,}")
    print(f"Different manager selected: {len(changed):,}")
    print(
        "Only baseline assigned a manager: "
        f"{(comparison['_merge'] == 'left_only').sum():,}"
    )
    print(
        "Only experimental assigned a manager: "
        f"{(comparison['_merge'] == 'right_only').sum():,}"
    )

    return comparison


def main():
    baseline, experimental, ground_truth = load_data()

    print("\n========== DATA COUNTS ==========")
    print(f"Ground truth relationships: {len(ground_truth):,}")
    print(f"Baseline relationships: {len(baseline):,}")
    print(f"Experimental relationships: {len(experimental):,}")

    results = pd.DataFrame(
        [
            calculate_metrics("Baseline", baseline, ground_truth),
            calculate_metrics("Experimental", experimental, ground_truth),
        ]
    )

    print("\n========== ACCURACY COMPARISON ==========")

    print(
        results.to_string(
            index=False,
            formatters={
                "Precision": "{:.4f}".format,
                "Recall": "{:.4f}".format,
                "F1 Score": "{:.4f}".format,
                "Assignment Coverage": "{:.4f}".format,
            },
        )
    )

    comparison = compare_assignments(baseline, experimental)

    results.to_csv(
        "data/processed/optimizer_metrics_comparison.csv",
        index=False,
    )

    comparison.to_csv(
        "data/processed/optimizer_assignment_comparison.csv",
        index=False,
    )

    print("\nSaved evaluation files:")
    print("- data/processed/optimizer_metrics_comparison.csv")
    print("- data/processed/optimizer_assignment_comparison.csv")


if __name__ == "__main__":
    main()
