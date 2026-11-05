from pathlib import Path

import pandas as pd

BASELINE_PATH = Path("data/processed/inferred_edges.parquet")
EXPERIMENTAL_PATH = Path("data/processed/experimental_inferred_edges.parquet")
GROUND_TRUTH_PATH = Path("data/ground_truth/ground_truth_edges.parquet")
COMPANY_METRICS_PATH = Path("data/processed/company_level_optimizer_comparison.csv")

OUTPUT_PATH = Path("data/processed/optimizer_regression_diagnostics.csv")


def load_data():
    baseline = pd.read_parquet(BASELINE_PATH)
    experimental = pd.read_parquet(EXPERIMENTAL_PATH)
    ground_truth = pd.read_parquet(GROUND_TRUTH_PATH)
    company_metrics = pd.read_csv(COMPANY_METRICS_PATH)

    baseline = baseline[["company_id", "employee_id", "manager_employee_id"]].rename(
        columns={"manager_employee_id": "baseline_manager"}
    )

    experimental = experimental[
        ["company_id", "employee_id", "manager_employee_id"]
    ].rename(columns={"manager_employee_id": "experimental_manager"})

    ground_truth = ground_truth[
        ["company_id", "employee_id", "manager_employee_id"]
    ].rename(columns={"manager_employee_id": "actual_manager"})

    return baseline, experimental, ground_truth, company_metrics


def build_assignment_comparison(baseline, experimental, ground_truth):
    comparison = baseline.merge(
        experimental,
        on=["company_id", "employee_id"],
        how="outer",
    )

    comparison = comparison.merge(
        ground_truth,
        on=["company_id", "employee_id"],
        how="left",
    )

    comparison["baseline_correct"] = comparison["baseline_manager"].notna() & (
        comparison["baseline_manager"] == comparison["actual_manager"]
    )

    comparison["experimental_correct"] = comparison["experimental_manager"].notna() & (
        comparison["experimental_manager"] == comparison["actual_manager"]
    )

    comparison["manager_changed"] = (
        comparison["baseline_manager"].notna()
        & comparison["experimental_manager"].notna()
        & (comparison["baseline_manager"] != comparison["experimental_manager"])
    )

    comparison["baseline_only"] = (
        comparison["baseline_manager"].notna()
        & comparison["experimental_manager"].isna()
    )

    comparison["experimental_only"] = (
        comparison["baseline_manager"].isna()
        & comparison["experimental_manager"].notna()
    )

    comparison["gained_correct_assignment"] = (
        ~comparison["baseline_correct"] & comparison["experimental_correct"]
    )

    comparison["lost_correct_assignment"] = (
        comparison["baseline_correct"] & ~comparison["experimental_correct"]
    )

    return comparison


def summarize_company(company_id, group):
    return {
        "company_id": company_id,
        "employees_compared": len(group),
        "manager_changed": int(group["manager_changed"].sum()),
        "baseline_only": int(group["baseline_only"].sum()),
        "experimental_only": int(group["experimental_only"].sum()),
        "gained_correct": int(group["gained_correct_assignment"].sum()),
        "lost_correct": int(group["lost_correct_assignment"].sum()),
        "baseline_correct": int(group["baseline_correct"].sum()),
        "experimental_correct": int(group["experimental_correct"].sum()),
    }


def main():
    print("Loading datasets...")

    baseline, experimental, ground_truth, company_metrics = load_data()

    comparison = build_assignment_comparison(baseline, experimental, ground_truth)

    # Overall diagnostic counts
    print("\n========== OVERALL DIAGNOSTICS ==========")

    print(
        "Correct assignments gained by experimental:",
        int(comparison["gained_correct_assignment"].sum()),
    )

    print(
        "Correct assignments lost by experimental:",
        int(comparison["lost_correct_assignment"].sum()),
    )

    print(
        "Manager assignments changed:",
        int(comparison["manager_changed"].sum()),
    )

    # Identify 10 companies with the largest F1 regressions
    worst_companies = (
        company_metrics.sort_values("f1_change").head(10)["company_id"].tolist()
    )

    summaries = []

    print("\n========== WORST 10 COMPANY DIAGNOSTICS ==========")

    for company_id in worst_companies:
        group = comparison[comparison["company_id"] == company_id].copy()

        summary = summarize_company(company_id, group)
        summaries.append(summary)

        print(f"\nCompany: {company_id}")
        print(f"Employees: {summary['employees_compared']}")
        print(f"Manager changed: {summary['manager_changed']}")
        print(f"Baseline only: {summary['baseline_only']}")
        print(f"Experimental only: {summary['experimental_only']}")
        print(f"Correct assignments gained: {summary['gained_correct']}")
        print(f"Correct assignments lost: {summary['lost_correct']}")
        print(f"Baseline correct: {summary['baseline_correct']}")
        print(
            "Experimental correct:",
            summary["experimental_correct"],
        )

        # Show examples where baseline was correct
        # but experimental selected the wrong manager.
        lost = group[group["lost_correct_assignment"]].head(5)

        if not lost.empty:
            print("\nExamples of correct baseline assignments lost:")

            print(
                lost[
                    [
                        "employee_id",
                        "actual_manager",
                        "baseline_manager",
                        "experimental_manager",
                    ]
                ].to_string(index=False)
            )

    summary_df = pd.DataFrame(summaries)

    summary_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved diagnostics to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
