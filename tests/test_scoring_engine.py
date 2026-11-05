import pandas as pd
import pytest

from src.inference.scoring_engine import (
    SCORING_WEIGHTS,
    get_role_family,
    calculate_department_score,
    calculate_seniority_score,
    calculate_title_score,
    calculate_manager_role_score,
    generate_explanation,
    calculate_candidate_scores,
)


@pytest.fixture
def candidate():
    return {
        "company_id": "C001",
        "employee_id": "E001",
        "candidate_manager_id": "E002",
        "employee_title": "Software Engineer",
        "candidate_manager_title": "Engineering Manager",
        "employee_department": "Engineering",
        "candidate_manager_department": "Engineering",
        "seniority_gap": 2,
        "candidate_manager_title_missing": False,
        "candidate_type": "known_title",
    }


class TestRoleFamily:

    @pytest.mark.parametrize(
        "title, expected_family",
        [
            ("Software Engineer", "technology_engineering"),
            ("Backend Developer", "technology_engineering"),
            ("Data Analyst", "data"),
            ("Sales Representative", "sales"),
            ("Marketing Specialist", "marketing"),
            ("Financial Analyst", "data"),
            ("Accountant", "finance"),
            ("HR Executive", "human_resources"),
            ("Operations Manager", "operations"),
            ("Product Designer", "product"),
            ("Security Analyst", "data"),
            ("Customer Success Manager", "customer_success"),
            ("CEO", None),
            (None, None),
        ],
    )
    def test_role_family_classification(self, title, expected_family):
        assert get_role_family(title) == expected_family

    def test_data_engineer_current_classification(self):
        # The technology family is checked before the data family.
        assert get_role_family("Data Engineer") == "technology_engineering"


class TestDepartmentScore:

    def test_same_department(self, candidate):
        assert calculate_department_score(candidate) == 1.0

    def test_different_department(self, candidate):
        candidate["candidate_manager_department"] = "Sales"

        assert calculate_department_score(candidate) == 0.0

    @pytest.mark.parametrize(
        "employee_department, manager_department",
        [
            (None, "Engineering"),
            ("Engineering", None),
            (None, None),
        ],
    )
    def test_missing_department(
        self, candidate, employee_department, manager_department
    ):
        candidate["employee_department"] = employee_department
        candidate["candidate_manager_department"] = manager_department

        assert calculate_department_score(candidate) == 0.5


class TestSeniorityScore:

    @pytest.mark.parametrize(
        "gap, expected_score",
        [
            (1, 1.0),
            (2, 0.8),
            (3, 0.6),
            (4, 0.3),
            (5, 0.3),
            (None, 0.5),
        ],
    )
    def test_seniority_gap_score(self, candidate, gap, expected_score):
        candidate["seniority_gap"] = gap

        assert calculate_seniority_score(candidate) == expected_score


class TestTitleScore:

    def test_same_role_family(self, candidate):
        candidate["employee_title"] = "Software Engineer"
        candidate["candidate_manager_title"] = "Engineering Manager"

        assert calculate_title_score(candidate) == 1.0

    def test_different_role_families(self, candidate):
        candidate["employee_title"] = "Software Engineer"
        candidate["candidate_manager_title"] = "Sales Manager"

        assert calculate_title_score(candidate) == 0.2

    def test_missing_employee_title(self, candidate):
        candidate["employee_title"] = None

        assert calculate_title_score(candidate) == 0.5

    def test_missing_manager_title(self, candidate):
        candidate["candidate_manager_title"] = None

        assert calculate_title_score(candidate) == 0.5


class TestManagerRoleScore:

    @pytest.mark.parametrize(
        "title, expected_score",
        [
            ("CEO", 1.0),
            ("Chief Technology Officer", 1.0),
            ("Vice President Engineering", 0.95),
            ("Engineering Director", 0.90),
            ("Head of Data", 0.90),
            ("Engineering Manager", 0.85),
            ("Engineering Lead", 0.75),
            ("Software Engineer", 0.30),
            (None, 0.5),
            ("", 0.5),
        ],
    )
    def test_manager_role_score(self, candidate, title, expected_score):
        candidate["candidate_manager_title"] = title

        assert calculate_manager_role_score(candidate) == expected_score


class TestCandidateScoring:

    def test_scoring_weights_total_one(self):
        assert sum(SCORING_WEIGHTS.values()) == pytest.approx(1.0)

    def test_calculates_all_component_scores(self, candidate):
        df = pd.DataFrame([candidate])

        result = calculate_candidate_scores(df)

        expected_columns = [
            "department_score",
            "seniority_score",
            "title_score",
            "manager_role_score",
            "confidence_score",
            "candidate_rank",
            "explanation",
        ]

        for column in expected_columns:
            assert column in result.columns

    def test_weighted_confidence_score(self, candidate):
        df = pd.DataFrame([candidate])

        result = calculate_candidate_scores(df)

        # Department: 1.0
        # Seniority: 0.8
        # Title: 1.0
        # Manager role: 0.85
        #
        # 100 * (1.0*0.30 + 0.8*0.25 + 1.0*0.25 + 0.85*0.20)
        expected_score = 92.0

        assert result.iloc[0]["confidence_score"] == expected_score

    def test_confidence_score_is_between_zero_and_100(self, candidate):
        df = pd.DataFrame([candidate])

        result = calculate_candidate_scores(df)

        assert result["confidence_score"].between(0, 100).all()

    def test_candidate_rank_starts_at_one(self, candidate):
        df = pd.DataFrame([candidate])

        result = calculate_candidate_scores(df)

        assert result.iloc[0]["candidate_rank"] == 1

    def test_candidates_are_ranked_by_confidence(self):
        df = pd.DataFrame(
            [
                {
                    "company_id": "C001",
                    "employee_id": "E001",
                    "candidate_manager_id": "E002",
                    "employee_title": "Software Engineer",
                    "candidate_manager_title": "Engineering Manager",
                    "employee_department": "Engineering",
                    "candidate_manager_department": "Engineering",
                    "seniority_gap": 1,
                    "candidate_manager_title_missing": False,
                },
                {
                    "company_id": "C001",
                    "employee_id": "E001",
                    "candidate_manager_id": "E003",
                    "employee_title": "Software Engineer",
                    "candidate_manager_title": "Sales Representative",
                    "employee_department": "Engineering",
                    "candidate_manager_department": "Sales",
                    "seniority_gap": 2,
                    "candidate_manager_title_missing": False,
                },
            ]
        )

        result = calculate_candidate_scores(df)

        assert result.iloc[0]["candidate_manager_id"] == "E002"
        assert result.iloc[0]["candidate_rank"] == 1
        assert result.iloc[1]["candidate_rank"] == 2

    def test_candidate_ranks_are_separate_for_different_employees(self):
        df = pd.DataFrame(
            [
                {
                    "company_id": "C001",
                    "employee_id": "E001",
                    "candidate_manager_id": "E003",
                    "employee_title": "Software Engineer",
                    "candidate_manager_title": "Engineering Manager",
                    "employee_department": "Engineering",
                    "candidate_manager_department": "Engineering",
                    "seniority_gap": 1,
                    "candidate_manager_title_missing": False,
                },
                {
                    "company_id": "C001",
                    "employee_id": "E002",
                    "candidate_manager_id": "E003",
                    "employee_title": "Sales Representative",
                    "candidate_manager_title": "Sales Manager",
                    "employee_department": "Sales",
                    "candidate_manager_department": "Sales",
                    "seniority_gap": 1,
                    "candidate_manager_title_missing": False,
                },
            ]
        )

        result = calculate_candidate_scores(df)

        assert result["candidate_rank"].tolist() == [1, 1]

    def test_empty_dataframe_returns_empty_dataframe(self):
        columns = [
            "company_id",
            "employee_id",
            "candidate_manager_id",
            "employee_title",
            "candidate_manager_title",
            "employee_department",
            "candidate_manager_department",
            "seniority_gap",
        ]

        df = pd.DataFrame(columns=columns)

        result = calculate_candidate_scores(df)

        assert result.empty

    def test_missing_required_column_raises_error(self, candidate):
        df = pd.DataFrame([candidate]).drop(columns=["seniority_gap"])

        with pytest.raises(ValueError, match="Missing required columns"):
            calculate_candidate_scores(df)


class TestExplanation:

    def test_explanation_includes_matching_department(self, candidate):
        row = {
            **candidate,
            "department_score": 1.0,
            "seniority_score": 0.8,
            "title_score": 1.0,
            "manager_role_score": 0.85,
        }

        explanation = generate_explanation(row)

        assert "Same department" in explanation
        assert "Strong seniority compatibility" in explanation
        assert "Related job-title function" in explanation
        assert "Suitable leadership title" in explanation

    def test_explanation_includes_missing_title_fallback(self, candidate):
        row = {
            **candidate,
            "candidate_manager_title_missing": True,
            "department_score": 1.0,
            "seniority_score": 0.5,
            "title_score": 0.5,
            "manager_role_score": 0.5,
        }

        explanation = generate_explanation(row)

        assert "Manager title missing" in explanation
        assert "Same department" in explanation

    def test_explanation_for_limited_evidence(self, candidate):
        row = {
            **candidate,
            "candidate_manager_title_missing": False,
            "department_score": 0.0,
            "seniority_score": 0.3,
            "title_score": 0.5,
            "manager_role_score": 0.3,
        }

        explanation = generate_explanation(row)

        assert explanation == "Limited supporting evidence"
