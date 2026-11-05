import random
import re

import pandas as pd
from faker import Faker


def generate_employees(
    hierarchy_df: pd.DataFrame, companies_df: pd.DataFrame, seed: int = 42
):
    """
    Generate synthetic employee profiles and a separate
    ground-truth reporting relationships dataset.

    Returns:
        employees_df:
            Employee information without manager relationships.

        ground_truth_edges_df:
            Actual employee-to-manager relationships.
            This dataset must only be used for evaluation.
    """

    if hierarchy_df.empty:
        raise ValueError("Hierarchy dataframe cannot be empty.")

    if companies_df.empty:
        raise ValueError("Companies dataframe cannot be empty.")

    required_hierarchy_columns = {
        "company_id",
        "position_id",
        "job_title",
        "department",
        "manager_position_id",
    }

    required_company_columns = {"company_id", "hq_city", "hq_country"}

    missing_hierarchy = required_hierarchy_columns - set(hierarchy_df.columns)

    missing_companies = required_company_columns - set(companies_df.columns)

    if missing_hierarchy:
        raise ValueError(f"Missing hierarchy columns: {missing_hierarchy}")

    if missing_companies:
        raise ValueError(f"Missing company columns: {missing_companies}")

    rng = random.Random(seed)

    fake = Faker()
    fake.seed_instance(seed)

    # Add company location information to each position.
    hierarchy = hierarchy_df.merge(
        companies_df[["company_id", "hq_city", "hq_country"]],
        on="company_id",
        how="left",
        validate="many_to_one",
    )

    if hierarchy["hq_country"].isna().any():
        raise ValueError("Some hierarchy records have no matching company.")

    employees = []

    # Map each position to its generated employee ID.
    position_to_employee = {}

    for index, row in hierarchy.iterrows():

        employee_id = f"E{index + 1:09d}"

        position_to_employee[row["position_id"]] = employee_id

    # Generate employee profiles.
    for index, row in hierarchy.iterrows():

        employee_id = position_to_employee[row["position_id"]]

        first_name = fake.first_name()
        last_name = fake.last_name()

        full_name = f"{first_name} {last_name}"

        # Employee ID ensures email uniqueness.
        email_name = re.sub(r"[^a-z0-9.]", "", f"{first_name}.{last_name}".lower())

        email = f"{email_name}.{employee_id.lower()}@example.com"

        # Most employees are located at their company HQ.
        # Some are assigned another synthetic city.
        if rng.random() < 0.70:
            city = row["hq_city"]
        else:
            city = fake.city()

        # Keep employment status realistic.
        employment_status = rng.choices(["Active", "On Leave"], weights=[96, 4], k=1)[0]

        hire_date = fake.date_between(start_date="-15y", end_date="today")

        employees.append(
            {
                "employee_id": employee_id,
                "company_id": row["company_id"],
                "first_name": first_name,
                "last_name": last_name,
                "full_name": full_name,
                "email": email,
                "job_title": row["job_title"],
                "department": row["department"],
                "city": city,
                "country": row["hq_country"],
                "hire_date": hire_date,
                "employment_status": employment_status,
            }
        )

    employees_df = pd.DataFrame(employees)

    # --------------------------------------------------
    # Generate ground-truth reporting relationships.
    # This is kept separate from employee input data.
    # --------------------------------------------------

    ground_truth_edges = []

    for _, row in hierarchy.iterrows():

        manager_position_id = row["manager_position_id"]

        # Root employee (CEO) has no manager.
        if pd.isna(manager_position_id):
            continue

        employee_id = position_to_employee[row["position_id"]]

        manager_employee_id = position_to_employee[manager_position_id]

        ground_truth_edges.append(
            {
                "company_id": row["company_id"],
                "employee_id": employee_id,
                "manager_employee_id": manager_employee_id,
            }
        )

    ground_truth_edges_df = pd.DataFrame(ground_truth_edges)

    # --------------------------------------------------
    # Final validations
    # --------------------------------------------------

    if employees_df["employee_id"].duplicated().any():
        raise AssertionError("Duplicate employee IDs found.")

    if employees_df["email"].duplicated().any():
        raise AssertionError("Duplicate employee emails found.")

    if employees_df["company_id"].isna().any():
        raise AssertionError("Employee company IDs cannot be null.")

    if employees_df["full_name"].isna().any():
        raise AssertionError("Employee names cannot be null.")

    if employees_df["job_title"].isna().any():
        raise AssertionError("Employee job titles cannot be null.")

    if (
        not ground_truth_edges_df["manager_employee_id"]
        .isin(employees_df["employee_id"])
        .all()
    ):
        raise AssertionError("Invalid manager employee IDs found.")

    return employees_df, ground_truth_edges_df


if __name__ == "__main__":

    from src.data_generation.company_generator import generate_companies

    from src.data_generation.hierarchy_generator import generate_all_hierarchies

    companies = generate_companies()

    hierarchy = generate_all_hierarchies(companies)

    employees, ground_truth = generate_employees(hierarchy, companies)

    print("\nEMPLOYEE DATASET")
    print("----------------")
    print("Total employees:", len(employees))
    print("Unique employee IDs:", employees.employee_id.nunique())
    print("Unique emails:", employees.email.nunique())
    print("Companies:", employees.company_id.nunique())

    print("\nGROUND-TRUTH EDGES")
    print("-----------------")
    print("Total reporting relationships:", len(ground_truth))
    print("Unique employees with managers:", ground_truth.employee_id.nunique())

    print("\nEMPLOYEE SAMPLE")
    print(employees.head(5).to_string(index=False))

    print("\nGROUND-TRUTH SAMPLE")
    print(ground_truth.head(5).to_string(index=False))
