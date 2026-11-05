import pandas as pd
import pytest

from src.inference.candidate_generator import (
    get_seniority_rank,
    generate_candidates_for_company,
)


@pytest.fixture
def employees():
    """Small synthetic dataset for candidate-generation tests."""

    return pd.DataFrame(
        [
            {
                "employee_id": "E001",
                "company_id": "C001",
                "department": "Engineering",
                "job_title": "Software Engineer",
            },
            {
                "employee_id": "E002",
                "company_id": "C001",
                "department": "Engineering",
                "job_title": "Engineering Lead",
            },
            {
                "employee_id": "E003",
                "company_id": "C001",
                "department": "Engineering",
                "job_title": "Engineering Manager",
            },
            {
                "employee_id": "E004",
                "company_id": "C001",
                "department": "Engineering",
                "job_title": None,
            },
            {
                "employee_id": "E005",
                "company_id": "C001",
                "department": "Sales",
                "job_title": None,
            },
            {
                "employee_id": "E006",
                "company_id": "C001",
                "department": "Sales",
                "job_title": "Sales Director",
            },
            {
                "employee_id": "E007",
                "company_id": "C001",
                "department": "Engineering",
                "job_title": "Software Engineer",
            },
            {
                "employee_id": "E008",
                "company_id": "C002",
                "department": "Engineering",
                "job_title": "Engineering Manager",
            },
        ]
    )


class TestSeniorityRank:

    @pytest.mark.parametrize(
        "title, expected_rank",
        [
            ("CEO", 6),
            ("Chief Technology Officer", 5),
            ("Vice President Engineering", 4),
            ("Engineering Director", 4),
            ("Head of Data", 4),
            ("Engineering Manager", 3),
            ("Engineering Lead", 2),
            ("Software Engineer", 1),
            ("Associate Product Manager", 1),
            ("Account Manager", 1),
            ("Customer Success Manager", 3),
            (None, 1),
            ("", 1),
        ],
    )
    def test_seniority_rank(self, title, expected_rank):
        assert get_seniority_rank(title) == expected_rank


class TestCandidateGeneration:

    def test_generates_candidates_for_company(self, employees):
        result = generate_candidates_for_company(
            employees,
            company_id="C001",
        )

        assert not result.empty
        assert set(result["company_id"]) == {"C001"}

    def test_does_not_include_employees_from_other_companies(self, employees):
        result = generate_candidates_for_company(
            employees,
            company_id="C001",
        )

        assert "E008" not in result["candidate_manager_id"].values

    def test_employee_cannot_be_their_own_manager(self, employees):
        result = generate_candidates_for_company(
            employees,
            company_id="C001",
        )

        assert (result["employee_id"] != result["candidate_manager_id"]).all()

    def test_known_title_candidates_have_higher_seniority(self, employees):
        result = generate_candidates_for_company(
            employees,
            company_id="C001",
        )

        known_candidates = result[result["candidate_type"] == "known_title"]

        invalid = known_candidates[
            known_candidates["manager_seniority_rank"]
            <= known_candidates["employee_seniority_rank"]
        ]

        assert invalid.empty

    def test_missing_title_fallback_requires_same_department(self, employees):
        result = generate_candidates_for_company(
            employees,
            company_id="C001",
        )

        fallback = result[result["candidate_type"] == "missing_title_fallback"]

        assert not fallback.empty

        assert (
            fallback["employee_department"] == fallback["candidate_manager_department"]
        ).all()

        assert fallback["candidate_manager_title_missing"].all()

    def test_missing_title_fallback_is_not_used_for_different_departments(
        self, employees
    ):
        result = generate_candidates_for_company(
            employees,
            company_id="C001",
        )

        fallback = result[result["candidate_type"] == "missing_title_fallback"]

        assert (
            fallback["employee_department"] == fallback["candidate_manager_department"]
        ).all()

    def test_maximum_candidates_per_employee(self, employees):
        result = generate_candidates_for_company(
            employees,
            company_id="C001",
            max_candidates=3,
            missing_title_slots=1,
        )

        counts = result.groupby("employee_id").size()

        assert counts.max() <= 3

    def test_known_candidates_are_prioritized(self, employees):
        result = generate_candidates_for_company(
            employees,
            company_id="C001",
            max_candidates=3,
            missing_title_slots=1,
        )

        employee_candidates = result[result["employee_id"] == "E001"]

        assert employee_candidates["candidate_type"].iloc[0] == "known_title"

    def test_known_candidates_are_sorted_by_seniority_gap(self, employees):
        result = generate_candidates_for_company(
            employees,
            company_id="C001",
            max_candidates=3,
            missing_title_slots=1,
        )

        employee_candidates = result[
            (result["employee_id"] == "E001")
            & (result["candidate_type"] == "known_title")
        ]

        gaps = employee_candidates["seniority_gap"].tolist()

        assert gaps == sorted(gaps)

    def test_missing_title_employee_can_receive_candidates(self, employees):
        result = generate_candidates_for_company(
            employees,
            company_id="C001",
        )

        employee_candidates = result[result["employee_id"] == "E004"]

        assert not employee_candidates.empty

    def test_empty_company_raises_error(self, employees):
        with pytest.raises(ValueError, match="No employees found"):
            generate_candidates_for_company(
                employees,
                company_id="C999",
            )

    def test_missing_required_column_raises_error(self, employees):
        invalid_df = employees.drop(columns=["department"])

        with pytest.raises(ValueError, match="Missing required columns"):
            generate_candidates_for_company(
                invalid_df,
                company_id="C001",
            )

    def test_invalid_max_candidates_raises_error(self, employees):
        with pytest.raises(ValueError, match="max_candidates"):
            generate_candidates_for_company(
                employees,
                company_id="C001",
                max_candidates=0,
            )

    def test_invalid_missing_title_slots_raises_error(self, employees):
        with pytest.raises(ValueError, match="missing_title_slots"):
            generate_candidates_for_company(
                employees,
                company_id="C001",
                max_candidates=5,
                missing_title_slots=6,
            )
