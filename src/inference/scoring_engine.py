import pandas as pd
import re

SCORING_WEIGHTS = {
    "department_score": 0.30,
    "seniority_score": 0.25,
    "title_score": 0.25,
    "manager_role_score": 0.20,
}


# Related terms are grouped into functional families.
ROLE_FAMILIES = {
    "technology_engineering": [
        "engineer",
        "engineering",
        "developer",
        "software",
        "technology",
        "technical",
        "devops",
        "platform",
        "infrastructure",
    ],
    "data": ["data", "analytics", "analyst", "database", "business intelligence", "bi"],
    "sales": ["sales", "revenue", "account", "business development"],
    "marketing": ["marketing", "brand", "growth", "communications"],
    "finance": ["finance", "accounting", "accountant", "treasury"],
    "human_resources": [
        "human resources",
        "hr",
        "people",
        "talent",
        "recruiting",
        "recruitment",
    ],
    "operations": ["operations", "operation", "logistics", "supply chain"],
    "product": ["product", "design", "ux"],
    "security": ["security", "cybersecurity"],
    "customer_success": ["customer success", "customer support", "support"],
}


def get_role_family(title):
    if pd.isna(title):
        return None

    title = str(title).lower()

    for family, keywords in ROLE_FAMILIES.items():
        for keyword in keywords:
            pattern = r"\b" + re.escape(keyword) + r"\b"

            if re.search(pattern, title):
                return family

    return None


def calculate_department_score(row):
    employee_department = row["employee_department"]
    manager_department = row["candidate_manager_department"]

    if pd.isna(employee_department) or pd.isna(manager_department):
        return 0.5

    return float(employee_department == manager_department)


def calculate_seniority_score(row):
    """
    Score seniority compatibility.

    Missing-title fallback candidates have unknown seniority.
    Their seniority evidence is neutral rather than negative.
    """

    gap = row["seniority_gap"]

    if pd.isna(gap):
        return 0.5

    if gap == 1:
        return 1.0
    elif gap == 2:
        return 0.8
    elif gap == 3:
        return 0.6

    return 0.3


def calculate_title_score(row):
    employee_family = get_role_family(row["employee_title"])
    manager_family = get_role_family(row["candidate_manager_title"])

    if employee_family is None or manager_family is None:
        return 0.5

    if employee_family == manager_family:
        return 1.0

    return 0.2


def calculate_manager_role_score(row):
    """
    Score the candidate's apparent management role.

    Missing titles receive neutral evidence rather than
    automatically being treated as unsuitable managers.
    """

    title = row["candidate_manager_title"]

    if pd.isna(title) or not str(title).strip():
        return 0.5

    title = str(title).lower()

    if "ceo" in title or "chief " in title:
        return 1.0

    if "vice president" in title or "vp " in title:
        return 0.95

    if "director" in title or "head of" in title:
        return 0.90

    if "manager" in title or "mgr" in title:
        return 0.85

    if "lead" in title:
        return 0.75

    return 0.30


def generate_explanation(row):
    reasons = []

    if row["candidate_manager_title_missing"]:
        reasons.append("Manager title missing; fallback based on matching department")

    if row["department_score"] == 1.0:
        reasons.append("Same department")

    elif row["department_score"] == 0.5:
        reasons.append("Department information incomplete")

    if row["seniority_score"] >= 0.8:
        reasons.append("Strong seniority compatibility")

    if row["title_score"] == 1.0:
        reasons.append("Related job-title function")

    elif row["title_score"] == 0.2:
        reasons.append("Different job-title functions")

    if row["manager_role_score"] >= 0.85:
        reasons.append("Suitable leadership title")

    if not reasons:
        reasons.append("Limited supporting evidence")

    return "; ".join(reasons)


def calculate_candidate_scores(candidates_df):

    required_columns = [
        "company_id",
        "employee_id",
        "candidate_manager_id",
        "employee_title",
        "candidate_manager_title",
        "employee_department",
        "candidate_manager_department",
        "seniority_gap",
    ]

    missing_columns = [
        column for column in required_columns if column not in candidates_df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if candidates_df.empty:
        return candidates_df.copy()

    scored = candidates_df.copy()

    scored["department_score"] = scored.apply(calculate_department_score, axis=1)

    scored["seniority_score"] = scored.apply(calculate_seniority_score, axis=1)

    scored["title_score"] = scored.apply(calculate_title_score, axis=1)

    scored["manager_role_score"] = scored.apply(calculate_manager_role_score, axis=1)

    scored["confidence_score"] = 0.0

    for score_column, weight in SCORING_WEIGHTS.items():
        scored["confidence_score"] += scored[score_column] * weight * 100

    scored["confidence_score"] = scored["confidence_score"].round(2)

    scored["candidate_rank"] = (
        scored.groupby(["company_id", "employee_id"])["confidence_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    scored["explanation"] = scored.apply(generate_explanation, axis=1)

    return scored


if __name__ == "__main__":

    from src.inference.candidate_generator import generate_candidates_for_company

    employees = pd.read_parquet("data/processed/employees_cleaned.parquet")

    candidates = generate_candidates_for_company(
        employees, employees["company_id"].iloc[0], max_candidates=10
    )

    scored = calculate_candidate_scores(candidates)

    print("\nScored candidate sample:")

    print(
        scored[
            [
                "employee_id",
                "candidate_manager_id",
                "department_score",
                "seniority_score",
                "title_score",
                "manager_role_score",
                "confidence_score",
                "candidate_rank",
                "explanation",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print("\nScore distribution:")
    print(scored["confidence_score"].describe())

    print("\nScores within 0-100:", scored["confidence_score"].between(0, 100).all())

    print("No missing scores:", scored["confidence_score"].notna().all())
