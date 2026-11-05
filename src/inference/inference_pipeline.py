import pandas as pd

from src.inference.candidate_generator import generate_candidates_for_company

from src.inference.scoring_engine import calculate_candidate_scores

from src.inference.graph_builder import build_org_graph, validate_org_graph


def run_inference_pipeline(employees_df, max_candidates=10, min_confidence=65):
    """
    Run the complete org-chart inference pipeline
    independently for each company.

    Returns:
        inferred_edges_df: Predicted reporting relationships.
        company_stats_df: Graph statistics for each company.
    """

    all_inferred_edges = []
    all_company_stats = []

    company_ids = employees_df["company_id"].dropna().unique()

    print("Total companies to process:", len(company_ids))

    for index, company_id in enumerate(company_ids, start=1):

        print(f"\nProcessing company {index}/{len(company_ids)}: " f"{company_id}")

        # Step 1: Generate candidate manager relationships.
        candidates = generate_candidates_for_company(
            employees_df, company_id=company_id, max_candidates=max_candidates
        )

        # Step 2: Calculate candidate scores.
        scored_candidates = calculate_candidate_scores(candidates)

        # Step 3: Build organizational graph.
        graph, inferred_edges = build_org_graph(
            employees_df,
            scored_candidates,
            company_id=company_id,
            min_confidence=min_confidence,
        )

        # Step 4: Validate graph.
        validation = validate_org_graph(graph)

        if not validation["is_directed_acyclic"]:
            raise ValueError(f"Cycle detected in company {company_id}")

        if not validation["no_employee_has_multiple_managers"]:
            raise ValueError(f"Multiple managers detected in company {company_id}")

        if not validation["no_self_reporting"]:
            raise ValueError(f"Self-reporting detected in company {company_id}")

        # Store inferred relationships.
        if not inferred_edges.empty:
            all_inferred_edges.append(inferred_edges)

        # Calculate company-level statistics.
        employee_count = graph.number_of_nodes()
        relationship_count = graph.number_of_edges()

        company_stats = {
            "company_id": company_id,
            "employee_count": employee_count,
            "candidate_pair_count": len(candidates),
            "inferred_relationship_count": relationship_count,
            "employees_without_manager": (employee_count - relationship_count),
            "root_count": validation["number_of_roots"],
            "is_directed_acyclic": validation["is_directed_acyclic"],
            "no_multiple_managers": (validation["no_employee_has_multiple_managers"]),
            "no_self_reporting": validation["no_self_reporting"],
        }

        all_company_stats.append(company_stats)

        print(
            f"Employees: {employee_count} | "
            f"Inferred relationships: {relationship_count} | "
            f"Roots: {validation['number_of_roots']}"
        )

    # Combine all company results.
    inferred_edges_df = (
        pd.concat(all_inferred_edges, ignore_index=True)
        if all_inferred_edges
        else pd.DataFrame()
    )

    company_stats_df = pd.DataFrame(all_company_stats)

    return inferred_edges_df, company_stats_df


if __name__ == "__main__":

    # Load only inference input data.
    employees = pd.read_parquet("data/processed/employees_cleaned.parquet")

    # Execute the full pipeline.
    inferred_edges, company_stats = run_inference_pipeline(
        employees, max_candidates=10, min_confidence=65
    )

    # Create output directory if needed.
    import os

    os.makedirs("data/processed", exist_ok=True)

    # Save inferred relationships.
    inferred_edges.to_parquet("data/processed/inferred_edges.parquet", index=False)

    # Save company-level graph statistics.
    company_stats.to_parquet("data/processed/company_graph_stats.parquet", index=False)

    # Display overall summary.
    print("\n========== PIPELINE SUMMARY ==========")

    print("Total employees:", len(employees))
    print("Total companies:", len(company_stats))
    print("Inferred relationships:", len(inferred_edges))

    print("Employees without inferred managers:", len(employees) - len(inferred_edges))

    print("Companies with valid DAGs:", company_stats["is_directed_acyclic"].all())

    print(
        "Companies without multiple managers:",
        company_stats["no_multiple_managers"].all(),
    )

    print("Companies without self-reporting:", company_stats["no_self_reporting"].all())

    print("\nCompany statistics sample:")
    print(company_stats.head(10).to_string(index=False))

    print("\n=====================================")

    print("\nSaved files:")
    print("- data/processed/inferred_edges.parquet")
    print("- data/processed/company_graph_stats.parquet")
