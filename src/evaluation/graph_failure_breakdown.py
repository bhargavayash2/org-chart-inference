import networkx as nx
import pandas as pd

from src.inference.candidate_generator import (
    generate_candidates_for_company,
)

from src.inference.scoring_engine import (
    calculate_candidate_scores,
)

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

EMPLOYEES_PATH = "data/processed/employees_cleaned.parquet"
GROUND_TRUTH_PATH = "data/ground_truth/ground_truth_edges.parquet"
INFERRED_PATH = "data/processed/inferred_edges.parquet"

OUTPUT_PATH = "data/processed/graph_failure_breakdown.csv"

MAX_CANDIDATES = 10
MISSING_TITLE_SLOTS = 2
MIN_CONFIDENCE = 65


# ---------------------------------------------------------
# REPLAY EXISTING GRAPH BUILDER
# ---------------------------------------------------------


def replay_graph_builder(employees_df, scored_candidates_df, company_id):
    """
    Replays the current graph-builder algorithm.

    Unlike the original graph builder, this function records
    decisions for ALL candidates, including candidates that
    were never reached because another manager was accepted.
    """

    company_employees = employees_df[employees_df["company_id"] == company_id].copy()

    company_candidates = scored_candidates_df[
        scored_candidates_df["company_id"] == company_id
    ].copy()

    graph = nx.DiGraph()

    graph.add_nodes_from(company_employees["employee_id"])

    # Match the existing graph builder's global ordering.
    company_candidates = company_candidates.sort_values(
        by="confidence_score",
        ascending=False,
    )

    decisions = []
    accepted_relationships = []

    for employee_id, employee_candidates in company_candidates.groupby(
        "employee_id",
        sort=False,
    ):

        employee_candidates = employee_candidates.sort_values(
            by="confidence_score",
            ascending=False,
        )

        manager_accepted = False

        for candidate_rank, candidate in enumerate(
            employee_candidates.itertuples(index=False),
            start=1,
        ):

            manager_id = candidate.candidate_manager_id

            confidence = candidate.confidence_score

            reason = None

            # -------------------------------------------------
            # Candidate was not reached because another
            # candidate had already been accepted.
            # -------------------------------------------------

            if manager_accepted:

                reason = "alternative_candidate_accepted_first"

            # -------------------------------------------------
            # Match existing graph-builder validations.
            # -------------------------------------------------

            elif employee_id == manager_id:

                reason = "self_reporting"

            elif employee_id not in graph or manager_id not in graph:

                reason = "employee_not_in_graph"

            elif confidence < MIN_CONFIDENCE:

                reason = "below_confidence_threshold"

            else:

                is_fallback = bool(candidate.candidate_manager_title_missing)

                if is_fallback:

                    employee_department = candidate.employee_department

                    manager_department = candidate.candidate_manager_department

                    departments_match = (
                        pd.notna(employee_department)
                        and pd.notna(manager_department)
                        and employee_department == manager_department
                    )

                    if not departments_match:
                        reason = "missing_title_department_mismatch"

                else:

                    if (
                        candidate.manager_seniority_rank
                        <= candidate.employee_seniority_rank
                    ):
                        reason = "seniority_rejected"

                # Cycle validation.
                if reason is None:

                    if nx.has_path(graph, employee_id, manager_id):
                        reason = "cycle_rejected"

            # -------------------------------------------------
            # Record every candidate decision.
            # -------------------------------------------------

            if reason is not None:

                decisions.append(
                    {
                        "company_id": company_id,
                        "employee_id": employee_id,
                        "candidate_manager_id": manager_id,
                        "candidate_rank": candidate_rank,
                        "confidence_score": confidence,
                        "decision": reason,
                    }
                )

                continue

            # -------------------------------------------------
            # Accept manager -> employee relationship.
            # -------------------------------------------------

            graph.add_edge(manager_id, employee_id)

            accepted_relationships.append(
                {
                    "company_id": company_id,
                    "employee_id": employee_id,
                    "manager_employee_id": manager_id,
                    "confidence_score": confidence,
                    "candidate_rank": candidate_rank,
                }
            )

            decisions.append(
                {
                    "company_id": company_id,
                    "employee_id": employee_id,
                    "candidate_manager_id": manager_id,
                    "candidate_rank": candidate_rank,
                    "confidence_score": confidence,
                    "decision": "accepted",
                }
            )

            manager_accepted = True

    decisions_df = pd.DataFrame(decisions)

    accepted_df = pd.DataFrame(accepted_relationships)

    return accepted_df, decisions_df


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------


def main():

    print("Loading datasets...")

    employees = pd.read_parquet(EMPLOYEES_PATH)

    ground_truth = pd.read_parquet(GROUND_TRUTH_PATH)

    inferred = pd.read_parquet(INFERRED_PATH)

    print("Employees:", len(employees))
    print("Ground-truth relationships:", len(ground_truth))
    print("Inferred relationships:", len(inferred))

    all_candidates = []
    all_decisions = []
    all_replayed_edges = []

    companies = employees["company_id"].unique()

    print("\nGenerating candidates and replaying graph construction...")

    # -----------------------------------------------------
    # Process every company.
    # -----------------------------------------------------

    for index, company_id in enumerate(companies, start=1):

        candidates = generate_candidates_for_company(
            employees,
            company_id=company_id,
            max_candidates=MAX_CANDIDATES,
            missing_title_slots=MISSING_TITLE_SLOTS,
        )

        if candidates.empty:
            continue

        scored_candidates = calculate_candidate_scores(candidates)

        # Keep EVERY generated candidate.
        candidate_copy = scored_candidates.copy()

        candidate_copy["candidate_rank"] = (
            candidate_copy.groupby("employee_id")["confidence_score"]
            .rank(method="first", ascending=False)
            .astype(int)
        )

        all_candidates.append(candidate_copy)

        # Replay existing graph builder.
        accepted_df, decisions_df = replay_graph_builder(
            employees,
            scored_candidates,
            company_id,
        )

        if not decisions_df.empty:
            all_decisions.append(decisions_df)

        if not accepted_df.empty:
            all_replayed_edges.append(accepted_df)

        if index % 50 == 0:
            print(f"Processed {index}/{len(companies)} companies")

    # -----------------------------------------------------
    # Combine results.
    # -----------------------------------------------------

    print("\nCombining diagnostic results...")

    if not all_candidates:
        raise ValueError("No candidates were generated.")

    candidates_df = pd.concat(
        all_candidates,
        ignore_index=True,
    )

    decisions_df = (
        pd.concat(
            all_decisions,
            ignore_index=True,
        )
        if all_decisions
        else pd.DataFrame(
            columns=[
                "company_id",
                "employee_id",
                "candidate_manager_id",
                "candidate_rank",
                "confidence_score",
                "decision",
            ]
        )
    )

    replayed_edges = (
        pd.concat(
            all_replayed_edges,
            ignore_index=True,
        )
        if all_replayed_edges
        else pd.DataFrame(
            columns=[
                "company_id",
                "employee_id",
                "manager_employee_id",
            ]
        )
    )

    # -----------------------------------------------------
    # STEP 1: VERIFY REPLAY AGAINST SAVED INFERENCE.
    # -----------------------------------------------------

    print("\n========== REPLAY VERIFICATION ==========")

    saved_edges = inferred[["employee_id", "manager_employee_id"]].rename(
        columns={"manager_employee_id": "saved_manager_employee_id"}
    )

    replay_edges = replayed_edges[["employee_id", "manager_employee_id"]].rename(
        columns={"manager_employee_id": "replayed_manager_employee_id"}
    )

    replay_comparison = saved_edges.merge(
        replay_edges,
        on="employee_id",
        how="outer",
    )

    replay_matches = (
        replay_comparison["saved_manager_employee_id"]
        == replay_comparison["replayed_manager_employee_id"]
    ).fillna(False)

    matching_count = int(replay_matches.sum())

    mismatching_count = int((~replay_matches).sum())

    print("Matching relationships:", matching_count)

    print("Mismatching relationships:", mismatching_count)

    if mismatching_count > 0:

        print("\nSample replay mismatches:")

        print(replay_comparison.loc[~replay_matches].head(10).to_string(index=False))

    # -----------------------------------------------------
    # STEP 2: FIND ACTUAL MANAGER IN CANDIDATE LIST.
    # -----------------------------------------------------

    print("\nAnalyzing ground-truth relationships...")

    candidate_lookup = candidates_df[
        [
            "company_id",
            "employee_id",
            "candidate_manager_id",
            "candidate_rank",
            "confidence_score",
            "candidate_type",
        ]
    ].drop_duplicates(
        subset=[
            "company_id",
            "employee_id",
            "candidate_manager_id",
        ]
    )

    actual_candidates = ground_truth.merge(
        candidate_lookup,
        left_on=[
            "company_id",
            "employee_id",
            "manager_employee_id",
        ],
        right_on=[
            "company_id",
            "employee_id",
            "candidate_manager_id",
        ],
        how="left",
    )

    # -----------------------------------------------------
    # STEP 3: GET THE GRAPH BUILDER'S DECISION FOR THE
    # ACTUAL MANAGER CANDIDATE.
    # -----------------------------------------------------

    decision_lookup = decisions_df[
        [
            "company_id",
            "employee_id",
            "candidate_manager_id",
            "decision",
        ]
    ].drop_duplicates(
        subset=[
            "company_id",
            "employee_id",
            "candidate_manager_id",
        ]
    )

    actual_candidates = actual_candidates.merge(
        decision_lookup,
        on=[
            "company_id",
            "employee_id",
            "candidate_manager_id",
        ],
        how="left",
    )

    # -----------------------------------------------------
    # STEP 4: GET THE ACTUALLY INFERRED MANAGER.
    # -----------------------------------------------------

    inferred_lookup = inferred[
        [
            "employee_id",
            "manager_employee_id",
        ]
    ].rename(columns={"manager_employee_id": "inferred_manager_employee_id"})

    actual_candidates = actual_candidates.merge(
        inferred_lookup,
        on="employee_id",
        how="left",
    )

    # -----------------------------------------------------
    # STEP 5: CLASSIFY EACH GROUND-TRUTH RELATIONSHIP.
    # -----------------------------------------------------

    actual_candidates["failure_category"] = "unclassified"

    correctly_inferred = (
        actual_candidates["manager_employee_id"]
        == actual_candidates["inferred_manager_employee_id"]
    )

    actual_candidates.loc[
        correctly_inferred,
        "failure_category",
    ] = "correctly_inferred"

    missed = ~correctly_inferred

    # Actual manager was never generated.
    missing_candidate = missed & actual_candidates["candidate_manager_id"].isna()

    actual_candidates.loc[
        missing_candidate,
        "failure_category",
    ] = "candidate_generation_failure"

    # Actual manager candidate was generated but rejected.
    rejection_mapping = {
        "below_confidence_threshold": "below_confidence_threshold",
        "seniority_rejected": "seniority_rejection",
        "cycle_rejected": "cycle_rejection",
        "missing_title_department_mismatch": ("missing_title_department_mismatch"),
        "self_reporting": "self_reporting_rejection",
        "employee_not_in_graph": "employee_not_in_graph",
    }

    for decision_name, category in rejection_mapping.items():

        mask = missed & (actual_candidates["decision"] == decision_name)

        actual_candidates.loc[
            mask,
            "failure_category",
        ] = category

    # Actual manager was generated, but another candidate
    # was accepted before the actual manager was reached.
    alternative_accepted = missed & (
        actual_candidates["decision"] == "alternative_candidate_accepted_first"
    )

    actual_candidates.loc[
        alternative_accepted,
        "failure_category",
    ] = "alternative_candidate_accepted_first"

    # The replay accepted the actual manager, but saved inference
    # does not agree. This indicates a replay discrepancy.
    replay_mismatch = missed & (actual_candidates["decision"] == "accepted")

    actual_candidates.loc[
        replay_mismatch,
        "failure_category",
    ] = "replay_inference_mismatch"

    # -----------------------------------------------------
    # STEP 6: SAVE DETAILED REPORT.
    # -----------------------------------------------------

    actual_candidates.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # -----------------------------------------------------
    # STEP 7: PRINT SUMMARY.
    # -----------------------------------------------------

    print("\n========== GRAPH FAILURE BREAKDOWN ==========")

    counts = actual_candidates["failure_category"].value_counts()

    print(counts.to_string())

    print("\nFailure percentages:")

    percentages = (
        actual_candidates["failure_category"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print(percentages.to_string())

    # -----------------------------------------------------
    # STEP 8: CANDIDATE RANK DISTRIBUTION.
    # -----------------------------------------------------

    print("\n========== ACTUAL MANAGER CANDIDATE RANK ==========")

    found_candidates = actual_candidates[actual_candidates["candidate_rank"].notna()]

    print(found_candidates["candidate_rank"].value_counts().sort_index().to_string())

    print("\nActual manager confidence statistics:")

    print(found_candidates["confidence_score"].describe().round(2).to_string())

    # -----------------------------------------------------
    # STEP 9: FAILURE SAMPLES.
    # -----------------------------------------------------

    print("\n========== FAILURE DETAILS ==========")

    categories = [
        "candidate_generation_failure",
        "below_confidence_threshold",
        "seniority_rejection",
        "cycle_rejection",
        "missing_title_department_mismatch",
        "alternative_candidate_accepted_first",
        "replay_inference_mismatch",
    ]

    sample_columns = [
        "company_id",
        "employee_id",
        "manager_employee_id",
        "inferred_manager_employee_id",
        "candidate_rank",
        "confidence_score",
        "candidate_type",
        "decision",
    ]

    for category in categories:

        subset = actual_candidates[actual_candidates["failure_category"] == category]

        print(f"\n{category}: {len(subset)}")

        if not subset.empty:

            available_columns = [col for col in sample_columns if col in subset.columns]

            print(subset[available_columns].head(5).to_string(index=False))

    # -----------------------------------------------------
    # FINAL OUTPUT.
    # -----------------------------------------------------

    print("\n========== FINAL SUMMARY ==========")

    print("Ground-truth relationships:", len(ground_truth))

    print("Generated candidates:", len(candidates_df))

    print("Saved inferred relationships:", len(inferred))

    print("Replayed relationships:", len(replayed_edges))

    print("Replay matching relationships:", matching_count)

    print("Replay mismatches:", mismatching_count)

    print("\nDetailed report saved to:")

    print(OUTPUT_PATH)

    print("\n===================================")


if __name__ == "__main__":
    main()
