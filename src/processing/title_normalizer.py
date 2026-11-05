import re
import pandas as pd

# Standardize common abbreviations in job titles.
ABBREVIATIONS = {
    r"\bsr\.?\b": "Senior",
    r"\bjr\.?\b": "Junior",
    r"\beng\.?\b": "Engineer",
    r"\bmgr\.?\b": "Manager",
    r"\bdir\.?\b": "Director",
    r"\bvp\b": "Vice President",
    r"\bsvp\b": "Senior Vice President",
    r"\bevp\b": "Executive Vice President",
}


def normalize_title(title):
    """
    Standardize a job title without changing its underlying meaning.

    Examples:
        Sr. Data Eng. -> Senior Data Engineer
        Jr Software Engineer -> Junior Software Engineer
        Data Mgr. -> Data Manager
    """

    if pd.isna(title) or not str(title).strip():
        return None

    title = str(title).strip()

    # Standardize casing and whitespace.
    title = re.sub(r"\s+", " ", title)
    title = title.title()

    # Keep common connector words lowercase.
    title = re.sub(r"\b(Of|And|For|In|At|To)\b", lambda m: m.group(1).lower(), title)

    # Replace common abbreviations.
    for pattern, replacement in ABBREVIATIONS.items():
        title = re.sub(pattern, replacement, title, flags=re.IGNORECASE)

    # Remove trailing punctuation and extra whitespace.
    title = re.sub(r"[.,]+$", "", title)
    title = re.sub(r"\s+", " ", title).strip()

    return title


def normalize_employee_titles(employees_df):
    """
    Add a normalized_job_title column while preserving the original job_title.
    """

    if "job_title" not in employees_df.columns:
        raise ValueError("Input DataFrame must contain a 'job_title' column.")

    result_df = employees_df.copy()

    result_df["normalized_job_title"] = result_df["job_title"].apply(normalize_title)

    return result_df


if __name__ == "__main__":

    employees = pd.read_parquet("data/raw/employees.parquet")

    normalized_employees = normalize_employee_titles(employees)

    print("Total employees:", len(normalized_employees))

    print("Missing original titles:", normalized_employees["job_title"].isna().sum())

    print(
        "Missing normalized titles:",
        normalized_employees["normalized_job_title"].isna().sum(),
    )

    print("\nSample title transformations:")

    examples = (
        normalized_employees[normalized_employees["job_title"].notna()][
            ["job_title", "normalized_job_title"]
        ]
        .drop_duplicates()
        .head(20)
    )

    print(examples.to_string(index=False))
