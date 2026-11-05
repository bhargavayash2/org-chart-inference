import random

import pandas as pd


def corrupt_employee_data(
    employees_df: pd.DataFrame,
    seed: int = 42,
    missing_department_rate: float = 0.05,
    missing_job_title_rate: float = 0.03,
    missing_city_rate: float = 0.05,
    missing_email_rate: float = 0.02,
    inconsistent_title_rate: float = 0.15,
):
    """
    Introduce realistic data quality issues into employee records.

    Returns:
        corrupted_df: Employee data with simulated imperfections.
        corruption_summary: Summary of introduced corruption.
    """

    if employees_df.empty:
        raise ValueError("Employee dataframe cannot be empty.")

    rates = {
        "missing_department_rate": missing_department_rate,
        "missing_job_title_rate": missing_job_title_rate,
        "missing_city_rate": missing_city_rate,
        "missing_email_rate": missing_email_rate,
        "inconsistent_title_rate": inconsistent_title_rate,
    }

    for rate_name, rate in rates.items():
        if not 0 <= rate <= 1:
            raise ValueError(f"{rate_name} must be between 0 and 1.")

    rng = random.Random(seed)

    corrupted_df = employees_df.copy(deep=True)

    # Track the original values before corruption.
    original_df = employees_df.copy(deep=True)

    corruption_summary = []

    # --------------------------------------------------
    # 1. Missing departments
    # --------------------------------------------------

    department_mask = pd.Series(
        [rng.random() < missing_department_rate for _ in range(len(corrupted_df))],
        index=corrupted_df.index,
    )

    corrupted_df.loc[department_mask, "department"] = None

    corruption_summary.append(
        {"issue": "Missing department", "affected_records": int(department_mask.sum())}
    )

    # --------------------------------------------------
    # 2. Missing job titles
    # --------------------------------------------------

    title_mask = pd.Series(
        [rng.random() < missing_job_title_rate for _ in range(len(corrupted_df))],
        index=corrupted_df.index,
    )

    corrupted_df.loc[title_mask, "job_title"] = None

    corruption_summary.append(
        {"issue": "Missing job title", "affected_records": int(title_mask.sum())}
    )

    # --------------------------------------------------
    # 3. Inconsistent job titles
    # --------------------------------------------------

    title_variations = {
        "Senior": ["Sr.", "Sr", "Senior"],
        "Junior": ["Jr.", "Jr", "Junior"],
        "Engineer": ["Eng.", "Engineer"],
        "Manager": ["Mgr.", "Mgr", "Manager"],
        "Director": ["Dir.", "Dir", "Director"],
    }

    inconsistent_count = 0

    for index in corrupted_df.index:

        current_title = corrupted_df.at[index, "job_title"]

        # Do not modify missing titles.
        if pd.isna(current_title):
            continue

        if rng.random() < inconsistent_title_rate:

            modified_title = current_title

            for original, variations in title_variations.items():

                if original in modified_title:

                    replacement = rng.choice(variations)

                    modified_title = modified_title.replace(original, replacement, 1)

                    break

            if modified_title != current_title:
                corrupted_df.at[index, "job_title"] = modified_title
                inconsistent_count += 1

    corruption_summary.append(
        {"issue": "Inconsistent job title", "affected_records": inconsistent_count}
    )

    # --------------------------------------------------
    # 4. Missing cities
    # --------------------------------------------------

    city_mask = pd.Series(
        [rng.random() < missing_city_rate for _ in range(len(corrupted_df))],
        index=corrupted_df.index,
    )

    corrupted_df.loc[city_mask, "city"] = None

    corruption_summary.append(
        {"issue": "Missing city", "affected_records": int(city_mask.sum())}
    )

    # --------------------------------------------------
    # 5. Missing emails
    # --------------------------------------------------

    email_mask = pd.Series(
        [rng.random() < missing_email_rate for _ in range(len(corrupted_df))],
        index=corrupted_df.index,
    )

    corrupted_df.loc[email_mask, "email"] = None

    corruption_summary.append(
        {"issue": "Missing email", "affected_records": int(email_mask.sum())}
    )

    corruption_summary_df = pd.DataFrame(corruption_summary)

    # --------------------------------------------------
    # Final validations
    # --------------------------------------------------

    if not corrupted_df["employee_id"].equals(original_df["employee_id"]):
        raise AssertionError("Employee IDs were modified.")

    if not corrupted_df["company_id"].equals(original_df["company_id"]):
        raise AssertionError("Company IDs were modified.")

    if len(corrupted_df) != len(original_df):
        raise AssertionError("Employee record count changed.")

    return corrupted_df, corruption_summary_df


if __name__ == "__main__":

    from src.data_generation.company_generator import generate_companies

    from src.data_generation.hierarchy_generator import generate_all_hierarchies

    from src.data_generation.employee_generator import generate_employees

    companies = generate_companies()

    hierarchy = generate_all_hierarchies(companies)

    employees, ground_truth = generate_employees(hierarchy, companies)

    corrupted_employees, summary = corrupt_employee_data(employees)

    print("\nDATA CORRUPTION SUMMARY")
    print("-----------------------")
    print(summary.to_string(index=False))

    print("\nORIGINAL EMPLOYEE COUNT:", len(employees))
    print("CORRUPTED EMPLOYEE COUNT:", len(corrupted_employees))

    print("\nORIGINAL SAMPLE")
    print(employees.head(5).to_string(index=False))

    print("\nCORRUPTED SAMPLE")
    print(corrupted_employees.head(5).to_string(index=False))
