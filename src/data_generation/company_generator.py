import random
import pandas as pd
from faker import Faker

INDUSTRIES = [
    "Technology",
    "Banking and Finance",
    "Healthcare",
    "Retail and E-commerce",
    "Telecommunications",
    "Manufacturing",
    "Consulting",
    "Logistics",
    "Education",
    "Media and Entertainment",
    "Energy",
    "Insurance",
]

COUNTRIES = [
    "United States",
    "United Kingdom",
    "India",
    "Germany",
    "Canada",
    "Australia",
    "Singapore",
    "Netherlands",
    "France",
    "Ireland",
]

COMPANY_TYPES = ["Private", "Public", "Non-Profit"]


def classify_company(size):
    """
    Classify companies based on employee count.
    """

    if size <= 150:
        return "Startup"

    elif size <= 250:
        return "Scale-up"

    elif size <= 350:
        return "Mid-Market"

    else:
        return "Enterprise"


def generate_companies(n_companies=500, min_employees=100, max_employees=500, seed=42):
    """
    Generate synthetic company records.

    Returns:
        pandas DataFrame containing company information.
    """

    if n_companies < 1:
        raise ValueError("n_companies must be at least 1")

    if min_employees < 100:
        raise ValueError("Minimum company size must be at least 100")

    if max_employees < min_employees:
        raise ValueError("max_employees must be >= min_employees")

    random.seed(seed)

    fake = Faker()
    fake.seed_instance(seed)

    companies = []

    for i in range(1, n_companies + 1):

        employee_count = random.randint(min_employees, max_employees)

        company = {
            "company_id": f"C{i:06d}",
            "company_name": fake.unique.company(),
            "industry": random.choice(INDUSTRIES),
            "employee_count": employee_count,
            "company_stage": classify_company(employee_count),
            "company_type": random.choice(COMPANY_TYPES),
            "hq_country": random.choice(COUNTRIES),
            "hq_city": fake.city(),
            "founded_year": random.randint(1985, 2025),
        }

        companies.append(company)

    df = pd.DataFrame(companies)

    return df


if __name__ == "__main__":

    companies_df = generate_companies(
        n_companies=500, min_employees=100, max_employees=500, seed=42
    )

    print("\nCOMPANY DATASET GENERATED")
    print("-" * 50)

    print("Total companies:", len(companies_df))

    print("Total target employees:", companies_df["employee_count"].sum())

    print("Minimum company size:", companies_df["employee_count"].min())

    print("Maximum company size:", companies_df["employee_count"].max())

    print("\nCompany distribution:")

    print(companies_df["company_stage"].value_counts())

    print("\nIndustry distribution:")

    print(companies_df["industry"].value_counts())

    print("\nSample companies:")

    print(companies_df.head(10).to_string(index=False))
