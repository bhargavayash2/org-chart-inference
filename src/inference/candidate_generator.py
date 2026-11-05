import pandas as pd

# These titles are explicitly generated as Individual Contributors.
INDIVIDUAL_CONTRIBUTOR_OVERRIDES = {
    "associate product manager",
    "account manager",
}

# This title is used for both ICs and Managers.
AMBIGUOUS_MANAGER_TITLES = {
    "customer success manager",
}


def get_seniority_rank(job_title):
    """
    Estimate seniority from a job title.
    Higher rank means greater potential management seniority.
    """

    if pd.isna(job_title) or not str(job_title).strip():
        return 1

    title = str(job_title).strip().lower()

    if title in INDIVIDUAL_CONTRIBUTOR_OVERRIDES:
        return 1

    if "chief executive officer" in title or title == "ceo":
        return 6

    if "chief " in title or title.startswith("chief"):
        return 5

    if any(word in title for word in ["vice president", "vp ", "director", "head of"]):
        return 4

    if "manager" in title or "mgr" in title:
        return 3

    if "lead" in title:
        return 2

    return 1


def generate_candidates_for_company(
    employees_df,
    company_id,
    max_candidates=10,
    missing_title_slots=2,
):
    """
    Generate manager candidates for one company.

    Known-title candidates are generated using seniority rules.

    Missing-title candidates are included only as controlled
    fallback candidates when both employees have the same,
    known department.

    Ground-truth relationships are never accessed.
    """

    required_columns = [
        "employee_id",
        "company_id",
        "department",
        "job_title",
    ]

    missing_columns = [
        column for column in required_columns if column not in employees_df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if max_candidates < 1:
        raise ValueError("max_candidates must be at least 1.")

    if not 0 <= missing_title_slots <= max_candidates:
        raise ValueError("missing_title_slots must be between 0 and max_candidates.")

    company_df = employees_df[employees_df["company_id"] == company_id].copy()

    if company_df.empty:
        raise ValueError(f"No employees found for company: {company_id}")

    title_column = (
        "normalized_job_title"
        if "normalized_job_title" in company_df.columns
        else "job_title"
    )

    company_df["seniority_rank"] = company_df[title_column].apply(get_seniority_rank)

    company_df["title_missing"] = company_df[title_column].isna() | company_df[
        title_column
    ].astype(str).str.strip().eq("")

    # Known-title candidates with estimated seniority above IC.
    known_managers = company_df[
        (~company_df["title_missing"]) & (company_df["seniority_rank"] > 1)
    ].copy()

    # Missing-title employees are a separate fallback pool.
    unknown_title_candidates = company_df[company_df["title_missing"]].copy()

    candidate_pairs = []

    for _, employee in company_df.iterrows():

        employee_title = employee[title_column]

        employee_title_key = (
            str(employee_title).strip().lower() if pd.notna(employee_title) else ""
        )

        employee_rank = employee["seniority_rank"]

        # Exclude the employee from both candidate pools.
        known_pool = known_managers[
            known_managers["employee_id"] != employee["employee_id"]
        ].copy()

        missing_pool = unknown_title_candidates[
            unknown_title_candidates["employee_id"] != employee["employee_id"]
        ].copy()

        # Normal candidates must have higher estimated seniority.
        higher_seniority = known_pool["seniority_rank"] > employee_rank

        # Exception for Customer Success Manager.
        if employee_title_key in AMBIGUOUS_MANAGER_TITLES:
            equal_seniority = known_pool["seniority_rank"] == employee_rank
            eligible = higher_seniority | equal_seniority
        else:
            eligible = higher_seniority

        known_pool = known_pool[eligible].copy()

        # Fallback candidates require known, matching departments.
        if pd.notna(employee["department"]):

            missing_pool = missing_pool[
                missing_pool["department"].notna()
                & (missing_pool["department"] == employee["department"])
            ].copy()

        else:
            missing_pool = missing_pool.iloc[0:0].copy()

        # Prepare known-title candidates.
        if not known_pool.empty:

            known_pool["same_department"] = known_pool["department"].notna() & (
                known_pool["department"] == employee["department"]
            )

            known_pool["seniority_gap"] = known_pool["seniority_rank"] - employee_rank

            known_pool["same_seniority_allowed"] = (
                known_pool["seniority_rank"] == employee_rank
            )

            known_pool["candidate_manager_title_missing"] = False
            known_pool["candidate_type"] = "known_title"

            known_pool = known_pool.sort_values(
                by=["same_department", "seniority_gap"],
                ascending=[False, True],
            )

        # Prepare missing-title fallback candidates.
        if not missing_pool.empty:

            missing_pool["same_department"] = True

            # Seniority is unknown, not estimated.
            missing_pool["seniority_gap"] = pd.NA
            missing_pool["same_seniority_allowed"] = False

            missing_pool["candidate_manager_title_missing"] = True
            missing_pool["candidate_type"] = "missing_title_fallback"

            # Department match is the only evidence used to
            # order candidates within this fallback pool.
            missing_pool = missing_pool.sort_values(
                by=["department"],
                ascending=True,
            )

        # Keep known-title candidates as the primary pool.
        normal_slots = max_candidates - missing_title_slots

        selected_known = known_pool.head(normal_slots)

        remaining_slots = max_candidates - len(selected_known)

        # Reserve fallback slots, but use fewer if unavailable.
        fallback_slots = min(missing_title_slots, remaining_slots)

        selected_missing = missing_pool.head(fallback_slots)

        # If fallback candidates are unavailable, fill unused
        # slots with additional known-title candidates.
        additional_slots = remaining_slots - len(selected_missing)

        if additional_slots > 0:
            remaining_known = known_pool.iloc[len(selected_known) :]
            selected_known = pd.concat(
                [
                    selected_known,
                    remaining_known.head(additional_slots),
                ],
                ignore_index=False,
            )

        selected_candidates = pd.concat(
            [selected_known, selected_missing],
            ignore_index=False,
        )

        for _, manager in selected_candidates.iterrows():

            candidate_pairs.append(
                {
                    "company_id": company_id,
                    "employee_id": employee["employee_id"],
                    "candidate_manager_id": manager["employee_id"],
                    "employee_title": employee_title,
                    "candidate_manager_title": manager[title_column],
                    "employee_department": employee["department"],
                    "candidate_manager_department": manager["department"],
                    "employee_seniority_rank": employee_rank,
                    "manager_seniority_rank": manager["seniority_rank"],
                    "seniority_gap": manager["seniority_gap"],
                    "same_department": manager["same_department"],
                    "same_seniority_allowed": manager["same_seniority_allowed"],
                    "candidate_manager_title_missing": manager[
                        "candidate_manager_title_missing"
                    ],
                    "candidate_type": manager["candidate_type"],
                    "hire_date": employee.get("hire_date", pd.NaT),
                    "candidate_manager_hire_date": manager.get("hire_date", pd.NaT),
                }
            )

    return pd.DataFrame(candidate_pairs)


if __name__ == "__main__":

    employees = pd.read_parquet("data/processed/employees_cleaned.parquet")

    sample_company_id = employees["company_id"].iloc[0]

    candidates = generate_candidates_for_company(
        employees,
        company_id=sample_company_id,
        max_candidates=10,
        missing_title_slots=2,
    )

    print("Sample company:", sample_company_id)

    print(
        "Employees in company:",
        employees[employees["company_id"] == sample_company_id].shape[0],
    )

    print("Generated candidate pairs:", len(candidates))

    if not candidates.empty:

        print("\nCandidate type distribution:")
        print(candidates["candidate_type"].value_counts())

        print("\nSample candidate pairs:")
        print(candidates.head(15).to_string(index=False))

        print(
            "\nMaximum candidates per employee:",
            candidates.groupby("employee_id").size().max(),
        )

    else:
        print("No candidate pairs generated.")
