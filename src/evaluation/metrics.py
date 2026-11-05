import pandas as pd

REQUIRED_COLUMNS = {"company_id", "employee_id", "manager_employee_id"}


def validate_relationship_data(ground_truth_df, inferred_df):
    """
    Validate the columns and uniqueness of relationship datasets.
    """

    for dataset_name, df in [
        ("Ground truth", ground_truth_df),
        ("Inferred relationships", inferred_df),
    ]:
        missing_columns = REQUIRED_COLUMNS - set(df.columns)

        if missing_columns:
            raise ValueError(
                f"{dataset_name} is missing columns: " f"{missing_columns}"
            )

        if df[list(REQUIRED_COLUMNS)].isnull().any().any():
            raise ValueError(f"{dataset_name} contains missing relationship IDs.")

        if df.duplicated(
            subset=["company_id", "employee_id", "manager_employee_id"]
        ).any():
            raise ValueError(f"{dataset_name} contains duplicate relationships.")


def calculate_metrics(ground_truth_df, inferred_df):
    """
    Calculate relationship-level evaluation metrics.

    A relationship is correct only when the company,
    employee, and manager IDs all match.
    """

    validate_relationship_data(ground_truth_df, inferred_df)

    ground_truth_pairs = set(
        zip(
            ground_truth_df["company_id"],
            ground_truth_df["employee_id"],
            ground_truth_df["manager_employee_id"],
        )
    )

    inferred_pairs = set(
        zip(
            inferred_df["company_id"],
            inferred_df["employee_id"],
            inferred_df["manager_employee_id"],
        )
    )

    # Correctly predicted relationships.
    true_positives = len(ground_truth_pairs.intersection(inferred_pairs))

    # Predicted relationships that are incorrect.
    false_positives = len(inferred_pairs - ground_truth_pairs)

    # Actual relationships that were not predicted.
    false_negatives = len(ground_truth_pairs - inferred_pairs)

    # Precision.
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )

    # Recall.
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )

    # F1-score.
    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Assignment coverage:
    # Of employees who actually have a manager,
    # how many received any predicted manager?
    actual_manager_employees = set(ground_truth_df["employee_id"])

    predicted_manager_employees = set(inferred_df["employee_id"])

    employees_with_prediction = actual_manager_employees.intersection(
        predicted_manager_employees
    )

    assignment_coverage = (
        len(employees_with_prediction) / len(actual_manager_employees)
        if actual_manager_employees
        else 0.0
    )

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "assignment_coverage": assignment_coverage,
        "ground_truth_relationships": len(ground_truth_pairs),
        "inferred_relationships": len(inferred_pairs),
    }


def evaluate_by_company(ground_truth_df, inferred_df):
    """
    Calculate evaluation metrics independently for each company.
    """

    validate_relationship_data(ground_truth_df, inferred_df)

    company_ids = sorted(
        set(ground_truth_df["company_id"]) | set(inferred_df["company_id"])
    )

    results = []

    for company_id in company_ids:

        company_ground_truth = ground_truth_df[
            ground_truth_df["company_id"] == company_id
        ]

        company_inferred = inferred_df[inferred_df["company_id"] == company_id]

        metrics = calculate_metrics(company_ground_truth, company_inferred)

        metrics["company_id"] = company_id

        results.append(metrics)

    return pd.DataFrame(results)


if __name__ == "__main__":

    # Load ground truth and inferred relationships.
    ground_truth = pd.read_parquet("data/ground_truth/ground_truth_edges.parquet")

    inferred = pd.read_parquet("data/processed/inferred_edges.parquet")

    # Calculate overall metrics.
    overall_metrics = calculate_metrics(ground_truth, inferred)

    print("\n========== OVERALL EVALUATION ==========")

    print("True Positives:", overall_metrics["true_positives"])

    print("False Positives:", overall_metrics["false_positives"])

    print("False Negatives:", overall_metrics["false_negatives"])

    print(f"Precision: " f"{overall_metrics['precision']:.4f}")

    print(f"Recall: " f"{overall_metrics['recall']:.4f}")

    print(f"F1-score: " f"{overall_metrics['f1_score']:.4f}")

    print(f"Assignment Coverage: " f"{overall_metrics['assignment_coverage']:.4f}")

    print("Ground-truth relationships:", overall_metrics["ground_truth_relationships"])

    print("Inferred relationships:", overall_metrics["inferred_relationships"])

    # Calculate metrics for every company.
    company_metrics = evaluate_by_company(ground_truth, inferred)

    print("\n========== COMPANY-LEVEL EVALUATION ==========")

    print(
        company_metrics[
            ["company_id", "precision", "recall", "f1_score", "assignment_coverage"]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\nTotal companies evaluated:", len(company_metrics))
