from pathlib import Path

from src.data_generation.company_generator import generate_companies
from src.data_generation.hierarchy_generator import generate_all_hierarchies
from src.data_generation.employee_generator import generate_employees
from src.data_generation.data_corruption import corrupt_employee_data


def generate_dataset():

    print("STEP 1: Generating companies...")
    companies = generate_companies()

    print("Companies generated:", len(companies))

    print("\nSTEP 2: Generating organizational hierarchies...")
    hierarchy = generate_all_hierarchies(companies)

    print("Positions generated:", len(hierarchy))

    print("\nSTEP 3: Generating employees...")
    employees, ground_truth = generate_employees(hierarchy, companies)

    print("Employees generated:", len(employees))
    print("Ground-truth relationships:", len(ground_truth))

    print("\nSTEP 4: Introducing data corruption...")
    corrupted_employees, corruption_summary = corrupt_employee_data(employees)

    print(corruption_summary.to_string(index=False))

    # Output directories.
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    ground_truth_dir = Path("data/ground_truth")

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    ground_truth_dir.mkdir(parents=True, exist_ok=True)

    print("\nSTEP 5: Saving Parquet files...")

    # Save corrupted employee data as inference input.
    corrupted_employees.to_parquet(raw_dir / "employees.parquet", index=False)

    # Save company metadata.
    companies.to_parquet(processed_dir / "companies.parquet", index=False)

    # Save actual relationships separately for evaluation.
    ground_truth.to_parquet(
        ground_truth_dir / "ground_truth_edges.parquet", index=False
    )

    print("\nDATASET GENERATION COMPLETED")
    print("----------------------------")
    print("Employee records:", len(corrupted_employees))
    print("Company records:", len(companies))
    print("Ground-truth edges:", len(ground_truth))

    print("\nSaved files:")
    print(raw_dir / "employees.parquet")
    print(processed_dir / "companies.parquet")
    print(ground_truth_dir / "ground_truth_edges.parquet")


if __name__ == "__main__":
    generate_dataset()
