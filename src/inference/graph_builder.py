import networkx as nx
import pandas as pd


def build_org_graph(
    employees_df,
    scored_candidates_df,
    company_id,
    min_confidence=65,
):
    """
    Build an organizational graph for one company.

    Graph direction:
        Manager -> Employee

    Each employee can have at most one manager.

    Normal candidates must have higher estimated seniority.

    Missing-title fallback candidates can bypass the seniority
    check only when both departments are known and match.
    """

    required_employee_columns = [
        "employee_id",
        "company_id",
        "job_title",
        "department",
    ]

    required_candidate_columns = [
        "company_id",
        "employee_id",
        "candidate_manager_id",
        "confidence_score",
        "explanation",
        "employee_seniority_rank",
        "manager_seniority_rank",
        "candidate_manager_title_missing",
        "employee_department",
        "candidate_manager_department",
    ]

    missing_employee_columns = [
        col for col in required_employee_columns if col not in employees_df.columns
    ]

    missing_candidate_columns = [
        col
        for col in required_candidate_columns
        if col not in scored_candidates_df.columns
    ]

    if missing_employee_columns:
        raise ValueError(f"Missing employee columns: {missing_employee_columns}")

    if missing_candidate_columns:
        raise ValueError(f"Missing candidate columns: {missing_candidate_columns}")

    if not 0 <= min_confidence <= 100:
        raise ValueError("min_confidence must be between 0 and 100.")

    company_employees = employees_df[employees_df["company_id"] == company_id].copy()

    if company_employees.empty:
        raise ValueError(f"No employees found for company: {company_id}")

    if not company_employees["employee_id"].is_unique:
        raise ValueError("Employee IDs must be unique.")

    company_candidates = scored_candidates_df[
        scored_candidates_df["company_id"] == company_id
    ].copy()

    graph = nx.DiGraph()

    # Add every employee as a graph node.
    for row in company_employees.itertuples(index=False):

        title = getattr(row, "normalized_job_title", None)

        if pd.isna(title):
            title = row.job_title

        graph.add_node(
            row.employee_id,
            company_id=row.company_id,
            job_title=row.job_title,
            normalized_job_title=title,
            department=row.department,
        )

    # Process higher-confidence candidates first.
    company_candidates = company_candidates.sort_values(
        by="confidence_score",
        ascending=False,
    )

    assigned_employees = set()
    accepted_relationships = []

    rejected_cycles = 0
    rejected_seniority = 0
    rejected_missing_title_fallback = 0

    for employee_id, employee_candidates in company_candidates.groupby(
        "employee_id",
        sort=False,
    ):

        if employee_id in assigned_employees:
            continue

        employee_candidates = employee_candidates.sort_values(
            by="confidence_score",
            ascending=False,
        )

        for candidate in employee_candidates.itertuples(index=False):

            manager_id = candidate.candidate_manager_id

            # Reject self-reporting.
            if employee_id == manager_id:
                continue

            # Both employees must exist in the graph.
            if employee_id not in graph or manager_id not in graph:
                continue

            # Reject low-confidence candidates.
            if candidate.confidence_score < min_confidence:
                continue

            is_missing_title_fallback = bool(candidate.candidate_manager_title_missing)

            if is_missing_title_fallback:

                employee_department = candidate.employee_department
                manager_department = candidate.candidate_manager_department

                # Fallback is allowed only when both departments
                # are known and exactly match.
                departments_match = (
                    pd.notna(employee_department)
                    and pd.notna(manager_department)
                    and employee_department == manager_department
                )

                if not departments_match:
                    rejected_missing_title_fallback += 1
                    continue

            else:

                # Normal candidates must have higher seniority.
                if (
                    candidate.manager_seniority_rank
                    <= candidate.employee_seniority_rank
                ):
                    rejected_seniority += 1
                    continue

            # Adding manager -> employee creates a cycle if
            # employee can already reach manager.
            if nx.has_path(graph, employee_id, manager_id):
                rejected_cycles += 1
                continue

            graph.add_edge(
                manager_id,
                employee_id,
                confidence_score=candidate.confidence_score,
                explanation=candidate.explanation,
                candidate_type=getattr(
                    candidate,
                    "candidate_type",
                    "known_title",
                ),
            )

            assigned_employees.add(employee_id)

            accepted_relationships.append(
                {
                    "company_id": company_id,
                    "employee_id": employee_id,
                    "manager_employee_id": manager_id,
                    "confidence_score": candidate.confidence_score,
                    "explanation": candidate.explanation,
                    "candidate_type": getattr(
                        candidate,
                        "candidate_type",
                        "known_title",
                    ),
                }
            )

            # One manager per employee.
            break

    relationships_df = pd.DataFrame(accepted_relationships)

    print("\nGraph construction diagnostics:")
    print("Rejected due to seniority:", rejected_seniority)
    print(
        "Rejected missing-title fallbacks:",
        rejected_missing_title_fallback,
    )
    print("Rejected due to cycles:", rejected_cycles)

    return graph, relationships_df


def validate_org_graph(graph):
    """
    Validate structural properties of an organizational graph.
    """

    return {
        "is_directed_acyclic": nx.is_directed_acyclic_graph(graph),
        "no_employee_has_multiple_managers": all(
            degree <= 1 for _, degree in graph.in_degree()
        ),
        "no_self_reporting": nx.number_of_selfloops(graph) == 0,
        "all_nodes_in_graph": graph.number_of_nodes() > 0,
        "number_of_nodes": graph.number_of_nodes(),
        "number_of_relationships": graph.number_of_edges(),
        "number_of_roots": len(
            [node for node, degree in graph.in_degree() if degree == 0]
        ),
    }


if __name__ == "__main__":

    from src.inference.candidate_generator import (
        generate_candidates_for_company,
    )

    from src.inference.scoring_engine import (
        calculate_candidate_scores,
    )

    employees = pd.read_parquet("data/processed/employees_cleaned.parquet")

    sample_company_id = employees["company_id"].iloc[0]

    candidates = generate_candidates_for_company(
        employees,
        company_id=sample_company_id,
        max_candidates=10,
        missing_title_slots=2,
    )

    scored_candidates = calculate_candidate_scores(candidates)

    graph, relationships = build_org_graph(
        employees,
        scored_candidates,
        company_id=sample_company_id,
        min_confidence=65,
    )

    validation = validate_org_graph(graph)

    print("\n========== GRAPH CONSTRUCTION ==========")

    print("Company:", sample_company_id)
    print("Employees:", graph.number_of_nodes())
    print("Accepted relationships:", graph.number_of_edges())
    print("Employees with assigned managers:", len(relationships))

    print(
        "Employees without assigned managers:",
        graph.number_of_nodes() - len(relationships),
    )

    print("\nGraph validation:")

    for check, result in validation.items():
        print(f"{check}: {result}")

    print("\nCandidate types accepted:")

    if not relationships.empty:
        print(relationships["candidate_type"].value_counts())

        print("\nSample reporting relationships:")
        print(relationships.head(15).to_string(index=False))

    print("\n========================================")
