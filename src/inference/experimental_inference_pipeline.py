import os
import pandas as pd

from src.inference.candidate_generator import (
    generate_candidates_for_company,
)

from src.inference.scoring_engine import (
    calculate_candidate_scores,
)

from src.inference.experimental_graph_optimizer import (
    build_experimental_org_graph,
    validate_experimental_graph,
)


def run_experimental_pipeline(
    employees_df,
    max_candidates=10,
    min_confidence=65,
):
    all_inferred_edges = []
    all_company_stats = []

    company_ids = employees_df["company_id"].dropna().unique()

    print("Total companies to process:", len(company_ids))

    for index, company_id in enumerate(company_ids, start=1):

        print(f"\nProcessing company {index}/{len(company_ids)}: " f"{company_id}")

        # Step 1: Generate candidate manager relationships.
        candidates = generate_candidates_for_company(
            employees_df,
            company_id=company_id,
            max_candidates=max_candidates,
        )

        # Step 2: Score candidates.
        scored_candidates = calculate_candidate_scores(candidates)

        # Step 3: Build graph using experimental optimizer.
        graph, inferred_edges = build_experimental_org_graph(
            employees_df,
            scored_candidates,
            company_id=company_id,
            min_confidence=min_confidence,
        )

        # Step 4: Validate graph.
        validation = validate_experimental_graph(graph)

        if not validation["is_directed_acyclic"]:
            raise ValueError(f"Cycle detected in {company_id}")

        if not validation["no_employee_has_multiple_managers"]:
            raise ValueError(f"Multiple managers detected in {company_id}")

        if not validation["no_self_reporting"]:
            raise ValueError(f"Self-reporting detected in {company_id}")

        if not inferred_edges.empty:
            all_inferred_edges.append(inferred_edges)

        employee_count = graph.number_of_nodes()
        relationship_count = len(inferred_edges)

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
            f"Relationships: {relationship_count} | "
            f"Roots: {validation['number_of_roots']}"
        )

        if index % 50 == 0:
            print(f"\nProgress: {index}/{len(company_ids)} companies")

    inferred_edges_df = (
        pd.concat(all_inferred_edges, ignore_index=True)
        if all_inferred_edges
        else pd.DataFrame()
    )

    company_stats_df = pd.DataFrame(all_company_stats)

    return inferred_edges_df, company_stats_df


if __name__ == "__main__":

    print("Loading employee data...")

    employees = pd.read_parquet("data/processed/employees_cleaned.parquet")

    print("Employees:", len(employees))

    inferred_edges, company_stats = run_experimental_pipeline(
        employees,
        max_candidates=10,
        min_confidence=65,
    )

    os.makedirs("data/processed", exist_ok=True)

    # Save separately. Existing baseline files remain untouched.
    inferred_edges.to_parquet(
        "data/processed/experimental_inferred_edges.parquet",
        index=False,
    )

    company_stats.to_parquet(
        "data/processed/experimental_company_graph_stats.parquet",
        index=False,
    )

    print("\n========== EXPERIMENTAL PIPELINE SUMMARY ==========")

    print("Total employees:", len(employees))
    print("Total companies:", len(company_stats))
    print("Inferred relationships:", len(inferred_edges))

    print(
        "Employees without inferred managers:",
        len(employees) - len(inferred_edges),
    )

    print(
        "Companies with valid DAGs:",
        company_stats["is_directed_acyclic"].all(),
    )

    print(
        "Companies without multiple managers:",
        company_stats["no_multiple_managers"].all(),
    )

    print(
        "Companies without self-reporting:",
        company_stats["no_self_reporting"].all(),
    )

    print("\nCompany statistics sample:")
    print(company_stats.head(10).to_string(index=False))

    print("\nSaved files:")
    print("- data/processed/experimental_inferred_edges.parquet")
    print("- data/processed/experimental_company_graph_stats.parquet")

    print("\n===================================================")
