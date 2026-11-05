import networkx as nx
import pandas as pd


def build_experimental_org_graph(
    employees_df,
    scored_candidates_df,
    company_id,
    min_confidence=65,
):
    """
    Build an organizational graph using maximum spanning arborescence.

    Graph direction:
        Manager -> Employee

    A virtual root allows employees to remain without an
    inferred manager.

    The virtual root is removed from the final graph.
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

    # Create directed graph.
    graph = nx.DiGraph()

    # Unique virtual root for this company.
    virtual_root = ("VIRTUAL_ROOT", company_id)

    graph.add_node(virtual_root, node_type="virtual_root")

    # Add every employee.
    for row in company_employees.itertuples(index=False):

        title = getattr(row, "normalized_job_title", None)

        if pd.isna(title):
            title = row.job_title

        graph.add_node(
            row.employee_id,
            node_type="employee",
            company_id=row.company_id,
            job_title=row.job_title,
            normalized_job_title=title,
            department=row.department,
        )

    # Connect virtual root to every employee.
    # This represents the option of having no inferred manager.
    #
    # Any eligible candidate with confidence >= min_confidence
    # has a slightly higher weight than this fallback.
    root_weight = min_confidence - 0.000001

    for employee_id in company_employees["employee_id"]:

        graph.add_edge(
            virtual_root,
            employee_id,
            weight=root_weight,
            is_virtual=True,
        )

    rejected_self = 0
    rejected_missing_employee = 0
    rejected_confidence = 0
    rejected_seniority = 0
    rejected_fallback = 0

    # Add eligible candidate relationships.
    for candidate in company_candidates.itertuples(index=False):

        employee_id = candidate.employee_id
        manager_id = candidate.candidate_manager_id

        # Reject self-reporting.
        if employee_id == manager_id:
            rejected_self += 1
            continue

        # Both employees must exist.
        if employee_id not in graph or manager_id not in graph:
            rejected_missing_employee += 1
            continue

        # Reject low-confidence candidates.
        if candidate.confidence_score < min_confidence:
            rejected_confidence += 1
            continue

        is_missing_title_fallback = bool(candidate.candidate_manager_title_missing)

        if is_missing_title_fallback:

            employee_department = candidate.employee_department
            manager_department = candidate.candidate_manager_department

            departments_match = (
                pd.notna(employee_department)
                and pd.notna(manager_department)
                and employee_department == manager_department
            )

            if not departments_match:
                rejected_fallback += 1
                continue

        else:

            # Normal candidates must have higher estimated seniority.
            if candidate.manager_seniority_rank <= candidate.employee_seniority_rank:
                rejected_seniority += 1
                continue

        # Add candidate edge.
        graph.add_edge(
            manager_id,
            employee_id,
            weight=float(candidate.confidence_score),
            is_virtual=False,
            confidence_score=float(candidate.confidence_score),
            explanation=candidate.explanation,
            candidate_type=getattr(
                candidate,
                "candidate_type",
                "known_title",
            ),
        )

    # Find globally optimal maximum-weight arborescence.
    optimized_graph = nx.maximum_spanning_arborescence(
        graph,
        attr="weight",
        default=0,
        preserve_attrs=True,
    )

    # Remove virtual root and its edges.
    optimized_graph.remove_node(virtual_root)

    # Extract inferred relationships.
    accepted_relationships = []

    for manager_id, employee_id, edge_data in optimized_graph.edges(data=True):

        # Ignore virtual relationships.
        if edge_data.get("is_virtual", False):
            continue

        accepted_relationships.append(
            {
                "company_id": company_id,
                "employee_id": employee_id,
                "manager_employee_id": manager_id,
                "confidence_score": edge_data["confidence_score"],
                "explanation": edge_data["explanation"],
                "candidate_type": edge_data["candidate_type"],
            }
        )

    relationships_df = pd.DataFrame(accepted_relationships)

    print("\nExperimental graph diagnostics:")
    print("Candidate pairs:", len(company_candidates))
    print("Rejected self-reporting:", rejected_self)
    print("Rejected missing employees:", rejected_missing_employee)
    print("Rejected low confidence:", rejected_confidence)
    print("Rejected seniority:", rejected_seniority)
    print("Rejected missing-title fallback:", rejected_fallback)

    print(
        "Selected relationships:",
        len(relationships_df),
    )

    print(
        "Employees without inferred managers:",
        len(company_employees) - len(relationships_df),
    )

    return optimized_graph, relationships_df


def validate_experimental_graph(graph):
    """
    Validate experimental organizational graph.
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

    graph, relationships = build_experimental_org_graph(
        employees,
        scored_candidates,
        company_id=sample_company_id,
        min_confidence=65,
    )

    validation = validate_experimental_graph(graph)

    print("\n========== EXPERIMENTAL GRAPH ==========")

    print("Company:", sample_company_id)
    print("Employees:", graph.number_of_nodes())
    print("Inferred relationships:", len(relationships))

    print("\nGraph validation:")

    for check, result in validation.items():
        print(f"{check}: {result}")

    if not relationships.empty:

        print("\nCandidate types accepted:")
        print(relationships["candidate_type"].value_counts())

        print("\nSample relationships:")
        print(relationships.head(15).to_string(index=False))

    print("\n========================================")
