import random
import pandas as pd

# --------------------------------------------------
# EXECUTIVE ROLES
# --------------------------------------------------

EXECUTIVE_ROLES = [
    {
        "title": "Chief Technology Officer",
        "department": "Technology",
        "key": "technology",
    },
    {"title": "Chief Financial Officer", "department": "Finance", "key": "finance"},
    {
        "title": "Chief Operating Officer",
        "department": "Operations",
        "key": "operations",
    },
    {"title": "Chief Revenue Officer", "department": "Sales", "key": "revenue"},
    {"title": "Chief Product Officer", "department": "Product", "key": "product"},
    {
        "title": "Chief Human Resources Officer",
        "department": "Human Resources",
        "key": "people",
    },
    {
        "title": "Chief Information Security Officer",
        "department": "Security",
        "key": "security",
    },
]


# --------------------------------------------------
# DEPARTMENT CONFIGURATION
# --------------------------------------------------

DEPARTMENTS = [
    {
        "name": "Engineering",
        "parent": "technology",
        "weight": 0.24,
        "leader": "VP Engineering",
    },
    {"name": "Data", "parent": "technology", "weight": 0.12, "leader": "Head of Data"},
    {"name": "Product", "parent": "product", "weight": 0.10, "leader": "VP Product"},
    {"name": "Sales", "parent": "revenue", "weight": 0.14, "leader": "VP Sales"},
    {
        "name": "Marketing",
        "parent": "revenue",
        "weight": 0.09,
        "leader": "Head of Marketing",
    },
    {
        "name": "Finance",
        "parent": "finance",
        "weight": 0.07,
        "leader": "Finance Director",
    },
    {
        "name": "Human Resources",
        "parent": "people",
        "weight": 0.06,
        "leader": "HR Director",
    },
    {
        "name": "Operations",
        "parent": "operations",
        "weight": 0.09,
        "leader": "Operations Director",
    },
    {
        "name": "Customer Success",
        "parent": "operations",
        "weight": 0.06,
        "leader": "Head of Customer Success",
    },
    {
        "name": "Security",
        "parent": "security",
        "weight": 0.03,
        "leader": "Head of Security",
    },
]


# --------------------------------------------------
# HELPER: GENERATE DEPARTMENT ALLOCATION
# --------------------------------------------------


def allocate_departments(total, departments):
    """
    Allocate employees across departments.

    Uses largest-remainder allocation so that
    department sizes add up exactly to total.
    """

    weights = [department["weight"] for department in departments]

    weight_sum = sum(weights)

    exact_allocations = [total * weight / weight_sum for weight in weights]

    allocations = [int(value) for value in exact_allocations]

    remaining = total - sum(allocations)

    remainders = sorted(
        range(len(departments)),
        key=lambda i: exact_allocations[i] - allocations[i],
        reverse=True,
    )

    for i in remainders[:remaining]:
        allocations[i] += 1

    return allocations


# --------------------------------------------------
# MAIN HIERARCHY GENERATOR
# --------------------------------------------------


def generate_company_hierarchy(company, seed=42):
    """
    Generate a realistic organizational hierarchy
    for one company.

    Returns:
        pandas DataFrame containing positions
        and their actual reporting relationships.
    """

    rng = random.Random(seed)

    company_id = company["company_id"]
    company_size = int(company["employee_count"])
    industry = company["industry"]

    positions = []

    position_counter = 0

    # ----------------------------------------------
    # HELPER: ADD POSITION
    # ----------------------------------------------

    def add_position(
        title, department, seniority, role_family, manager_id, is_people_manager, level
    ):

        nonlocal position_counter

        position_counter += 1

        position_id = f"{company_id}-P{position_counter:06d}"

        position = {
            "company_id": company_id,
            "position_id": position_id,
            "job_title": title,
            "department": department,
            "seniority": seniority,
            "role_family": role_family,
            "manager_position_id": manager_id,
            "is_people_manager": is_people_manager,
            "org_level": level,
        }

        positions.append(position)

        return position_id

    # ----------------------------------------------
    # 1. GENERATE CEO
    # ----------------------------------------------

    ceo_id = add_position(
        title="Chief Executive Officer",
        department="Executive",
        seniority="C-Suite",
        role_family="Executive Leadership",
        manager_id=None,
        is_people_manager=True,
        level=0,
    )

    # ----------------------------------------------
    # 2. SELECT EXECUTIVE LEADERSHIP
    # ----------------------------------------------

    if company_size <= 150:
        executive_count = 4

    elif company_size <= 300:
        executive_count = 5

    else:
        executive_count = 6

    selected_executives = rng.sample(EXECUTIVE_ROLES, executive_count)

    executive_map = {}

    for executive in selected_executives:

        executive_id = add_position(
            title=executive["title"],
            department=executive["department"],
            seniority="C-Suite",
            role_family="Executive Leadership",
            manager_id=ceo_id,
            is_people_manager=True,
            level=1,
        )

        executive_map[executive["key"]] = executive_id

    # ----------------------------------------------
    # 3. SELECT DEPARTMENTS
    # ----------------------------------------------

    selected_departments = DEPARTMENTS.copy()

    rng.shuffle(selected_departments)

    if company_size <= 150:
        department_count = 7

    elif company_size <= 300:
        department_count = 8

    else:
        department_count = 10

    selected_departments = selected_departments[:department_count]

    # ----------------------------------------------
    # 4. CALCULATE DEPARTMENT SIZES
    # ----------------------------------------------

    remaining_positions = company_size - 1 - executive_count

    allocations = allocate_departments(remaining_positions, selected_departments)

    # ----------------------------------------------
    # 5. GENERATE DEPARTMENT HIERARCHIES
    # ----------------------------------------------

    for department, team_size in zip(selected_departments, allocations):

        department_name = department["name"]

        # Identify executive manager.
        parent_key = department["parent"]

        executive_manager = executive_map.get(parent_key, ceo_id)

        # Department head.
        head_id = add_position(
            title=department["leader"],
            department=department_name,
            seniority="VP/Director/Head",
            role_family="Department Leadership",
            manager_id=executive_manager,
            is_people_manager=True,
            level=2,
        )

        # Remaining employees in department.
        remaining = team_size - 1

        # ------------------------------------------
        # MANAGER LAYER
        # ------------------------------------------

        manager_count = 0

        if team_size >= 20:

            manager_count = min(4, max(1, team_size // 25))

        managers = []

        for i in range(manager_count):

            manager_id = add_position(
                title=rng.choice(
                    [
                        f"{department_name} Manager",
                        f"Senior {department_name} Manager",
                        f"{department_name} Team Manager",
                    ]
                ),
                department=department_name,
                seniority="Manager",
                role_family="People Management",
                manager_id=head_id,
                is_people_manager=True,
                level=3,
            )

            managers.append(manager_id)

        remaining -= manager_count

        # ------------------------------------------
        # TEAM LEAD LAYER
        # ------------------------------------------

        lead_count = 0

        if team_size >= 35:

            lead_count = min(6, team_size // 18)

        leads = []

        if managers:

            for i in range(lead_count):

                manager_id = rng.choice(managers)

                lead_id = add_position(
                    title=rng.choice(
                        [
                            f"Senior {department_name} Team Lead",
                            f"{department_name} Lead",
                            f"Lead {department_name} Specialist",
                        ]
                    ),
                    department=department_name,
                    seniority="Team Lead",
                    role_family="Team Leadership",
                    manager_id=manager_id,
                    is_people_manager=True,
                    level=4,
                )

                leads.append(lead_id)

        remaining -= lead_count

        # ------------------------------------------
        # INDIVIDUAL CONTRIBUTORS
        # ------------------------------------------

        if leads:
            supervisors = leads

        elif managers:
            supervisors = managers

        else:
            supervisors = [head_id]

        ic_titles = {
            "Engineering": [
                "Software Engineer",
                "Senior Software Engineer",
                "Backend Engineer",
                "Frontend Engineer",
                "QA Engineer",
            ],
            "Data": [
                "Data Engineer",
                "Senior Data Engineer",
                "Data Analyst",
                "Analytics Engineer",
                "Data Scientist",
            ],
            "Product": [
                "Product Analyst",
                "Product Specialist",
                "Product Designer",
                "Associate Product Manager",
            ],
            "Sales": [
                "Account Executive",
                "Sales Representative",
                "Business Development Executive",
                "Sales Operations Analyst",
            ],
            "Marketing": [
                "Marketing Specialist",
                "Content Strategist",
                "SEO Analyst",
                "Digital Marketing Executive",
            ],
            "Finance": [
                "Financial Analyst",
                "Accountant",
                "Finance Associate",
                "Accounts Executive",
            ],
            "Human Resources": [
                "HR Executive",
                "Talent Acquisition Specialist",
                "HR Analyst",
                "Recruiter",
            ],
            "Operations": [
                "Operations Analyst",
                "Operations Executive",
                "Process Specialist",
                "Business Operations Associate",
            ],
            "Customer Success": [
                "Customer Success Manager",
                "Customer Success Associate",
                "Account Manager",
                "Support Specialist",
            ],
            "Security": [
                "Security Analyst",
                "Information Security Engineer",
                "Security Specialist",
                "Cybersecurity Analyst",
            ],
        }

        titles = ic_titles.get(department_name, ["Business Analyst", "Specialist"])

        for i in range(remaining):

            supervisor_id = supervisors[i % len(supervisors)]

            title = rng.choice(titles)

            add_position(
                title=title,
                department=department_name,
                seniority="Individual Contributor",
                role_family="Individual Contributor",
                manager_id=supervisor_id,
                is_people_manager=False,
                level=5 if leads else (4 if managers else 3),
            )

    # ----------------------------------------------
    # 6. VALIDATION
    # ----------------------------------------------

    df = pd.DataFrame(positions)

    assert len(df) == company_size, (
        f"Expected {company_size} positions, " f"generated {len(df)}"
    )

    assert df["position_id"].is_unique

    assert df["company_id"].nunique() == 1

    assert df["manager_position_id"].isna().sum() == 1

    return df


# --------------------------------------------------
# GENERATE HIERARCHIES FOR ALL COMPANIES
# --------------------------------------------------


def generate_all_hierarchies(companies_df, seed=42):
    """
    Generate organizational hierarchies
    for every company.
    """

    all_positions = []

    for index, company in companies_df.iterrows():

        company_dict = company.to_dict()

        hierarchy = generate_company_hierarchy(company_dict, seed=seed + index)

        all_positions.append(hierarchy)

    return pd.concat(all_positions, ignore_index=True)


# --------------------------------------------------
# EXECUTION
# --------------------------------------------------

if __name__ == "__main__":

    from src.data_generation.company_generator import generate_companies

    companies = generate_companies(n_companies=500, seed=42)

    hierarchy_df = generate_all_hierarchies(companies, seed=42)

    print("\nHIERARCHY GENERATION COMPLETED")
    print("-" * 50)

    print("Total positions:", len(hierarchy_df))

    print("Total companies:", hierarchy_df["company_id"].nunique())

    print("Unique positions:", hierarchy_df["position_id"].nunique())

    print("\nSeniority distribution:")

    print(hierarchy_df["seniority"].value_counts())

    print("\nDepartment distribution:")

    print(hierarchy_df["department"].value_counts())

    print("\nSample hierarchy:")

    sample = hierarchy_df[hierarchy_df["company_id"] == "C000001"]

    print(
        sample[["job_title", "department", "seniority", "manager_position_id"]]
        .head(20)
        .to_string(index=False)
    )
