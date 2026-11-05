import networkx as nx
import pandas as pd
import pytest

from src.inference.graph_builder import (
    build_org_graph,
    validate_org_graph,
)


@pytest.fixture
def employees():
    return pd.DataFrame(
        [
            {
                "employee_id": "E001",
                "company_id": "C001",
                "job_title": "Software Engineer",
                "normalized_job_title": "Software Engineer",
                "department": "Engineering",
            },
            {
                "employee_id": "E002",
                "company_id": "C001",
                "job_title": "Engineering Manager",
                "normalized_job_title": "Engineering Manager",
                "department": "Engineering",
            },
            {
                "employee_id": "E003",
                "company_id": "C001",
                "job_title": "Engineering Director",
                "normalized_job_title": "Engineering Director",
                "department": "Engineering",
            },
            {
                "employee_id": "E004",
                "company_id": "C001",
                "job_title": None,
                "normalized_job_title": None,
                "department": "Engineering",
            },
        ]
    )


@pytest.fixture
def candidate():
    return {
        "company_id": "C001",
        "employee_id": "E001",
        "candidate_manager_id": "E002",
        "confidence_score": 90.0,
        "explanation": "Same department; Strong seniority compatibility",
        "employee_seniority_rank": 1,
        "manager_seniority_rank": 3,
        "candidate_manager_title_missing": False,
        "employee_department": "Engineering",
        "candidate_manager_department": "Engineering",
        "candidate_type": "known_title",
    }


def build_graph(employees, candidates, min_confidence=65):

    candidate_columns = [
        "company_id",
        "employee_id",
        "candidate_manager_id",
        "confidence_score",
        "explanation",
        "employee_seniority_rank",
        "manager_seniority_rank",
        "candidate_manager_title_missing",
        "employee_department",
        "candidate_manager_department",
    ]

    candidates_df = pd.DataFrame(candidates)

    if candidates_df.empty:
        candidates_df = pd.DataFrame(columns=candidate_columns)

    return build_org_graph(
        employees,
        candidates_df,
        company_id="C001",
        min_confidence=min_confidence,
    )


class TestBasicGraphConstruction:

    def test_adds_all_company_employees_as_nodes(self, employees):
        graph, relationships = build_graph(employees, [])

        assert graph.number_of_nodes() == 4
        assert set(graph.nodes) == {"E001", "E002", "E003", "E004"}

    def test_creates_manager_to_employee_edge(self, employees, candidate):
        graph, relationships = build_graph(employees, [candidate])

        assert graph.has_edge("E002", "E001")
        assert not graph.has_edge("E001", "E002")

    def test_relationship_dataframe_contains_manager(self, employees, candidate):
        graph, relationships = build_graph(employees, [candidate])

        assert len(relationships) == 1
        assert relationships.iloc[0]["employee_id"] == "E001"
        assert relationships.iloc[0]["manager_employee_id"] == "E002"

    def test_preserves_confidence_and_explanation(self, employees, candidate):
        graph, relationships = build_graph(employees, [candidate])

        edge = graph["E002"]["E001"]

        assert edge["confidence_score"] == 90.0
        assert edge["explanation"] == candidate["explanation"]

    def test_preserves_candidate_type(self, employees, candidate):
        graph, relationships = build_graph(employees, [candidate])

        assert graph["E002"]["E001"]["candidate_type"] == "known_title"
        assert relationships.iloc[0]["candidate_type"] == "known_title"

    def test_defaults_candidate_type_when_missing(self, employees, candidate):
        candidate.pop("candidate_type")

        graph, relationships = build_graph(employees, [candidate])

        assert graph["E002"]["E001"]["candidate_type"] == "known_title"


class TestCandidateRejection:

    def test_rejects_self_reporting(self, employees, candidate):
        candidate["candidate_manager_id"] = "E001"

        graph, relationships = build_graph(employees, [candidate])

        assert graph.number_of_edges() == 0
        assert relationships.empty

    def test_rejects_candidate_manager_not_in_graph(self, employees, candidate):
        candidate["candidate_manager_id"] = "E999"

        graph, relationships = build_graph(employees, [candidate])

        assert graph.number_of_edges() == 0
        assert relationships.empty

    def test_rejects_employee_not_in_graph(self, employees, candidate):
        candidate["employee_id"] = "E999"

        graph, relationships = build_graph(employees, [candidate])

        assert graph.number_of_edges() == 0
        assert relationships.empty

    def test_rejects_low_confidence_candidate(self, employees, candidate):
        candidate["confidence_score"] = 64.99

        graph, relationships = build_graph(employees, [candidate])

        assert graph.number_of_edges() == 0
        assert relationships.empty

    def test_accepts_candidate_at_exact_threshold(self, employees, candidate):
        candidate["confidence_score"] = 65.0

        graph, relationships = build_graph(employees, [candidate], min_confidence=65)

        assert graph.has_edge("E002", "E001")

    def test_rejects_equal_seniority(self, employees, candidate):
        candidate["manager_seniority_rank"] = 1
        candidate["employee_seniority_rank"] = 1

        graph, relationships = build_graph(employees, [candidate])

        assert graph.number_of_edges() == 0

    def test_rejects_lower_manager_seniority(self, employees, candidate):
        candidate["manager_seniority_rank"] = 1
        candidate["employee_seniority_rank"] = 2

        graph, relationships = build_graph(employees, [candidate])

        assert graph.number_of_edges() == 0


class TestMissingTitleFallback:

    def test_accepts_fallback_with_matching_departments(self, employees, candidate):
        candidate["candidate_manager_id"] = "E004"
        candidate["candidate_manager_title_missing"] = True
        candidate["candidate_manager_department"] = "Engineering"
        candidate["candidate_type"] = "missing_title_fallback"

        # Fallback can bypass seniority.
        candidate["manager_seniority_rank"] = 1
        candidate["employee_seniority_rank"] = 3

        graph, relationships = build_graph(employees, [candidate])

        assert graph.has_edge("E004", "E001")
        assert relationships.iloc[0]["candidate_type"] == "missing_title_fallback"

    def test_rejects_fallback_with_different_departments(self, employees, candidate):
        candidate["candidate_manager_id"] = "E004"
        candidate["candidate_manager_title_missing"] = True
        candidate["candidate_manager_department"] = "Sales"

        graph, relationships = build_graph(employees, [candidate])

        assert graph.number_of_edges() == 0
        assert relationships.empty

    @pytest.mark.parametrize(
        "employee_department, manager_department",
        [
            (None, "Engineering"),
            ("Engineering", None),
            (None, None),
        ],
    )
    def test_rejects_fallback_with_missing_department(
        self,
        employees,
        candidate,
        employee_department,
        manager_department,
    ):
        candidate["candidate_manager_id"] = "E004"
        candidate["candidate_manager_title_missing"] = True
        candidate["employee_department"] = employee_department
        candidate["candidate_manager_department"] = manager_department

        graph, relationships = build_graph(employees, [candidate])

        assert graph.number_of_edges() == 0
        assert relationships.empty


class TestCandidateSelection:

    def test_chooses_highest_confidence_valid_candidate(self, employees, candidate):
        lower = candidate.copy()
        lower["candidate_manager_id"] = "E003"
        lower["confidence_score"] = 75.0

        higher = candidate.copy()
        higher["candidate_manager_id"] = "E002"
        higher["confidence_score"] = 95.0

        graph, relationships = build_graph(employees, [lower, higher])

        assert graph.has_edge("E002", "E001")
        assert not graph.has_edge("E003", "E001")

    def test_tries_next_candidate_if_highest_is_invalid(self, employees, candidate):
        invalid = candidate.copy()
        invalid["candidate_manager_id"] = "E003"
        invalid["confidence_score"] = 95.0
        invalid["manager_seniority_rank"] = 1
        invalid["employee_seniority_rank"] = 2

        valid = candidate.copy()
        valid["candidate_manager_id"] = "E002"
        valid["confidence_score"] = 85.0

        graph, relationships = build_graph(employees, [invalid, valid])

        assert graph.has_edge("E002", "E001")
        assert len(relationships) == 1

    def test_assigns_at_most_one_manager_per_employee(self, employees, candidate):
        second = candidate.copy()
        second["candidate_manager_id"] = "E003"
        second["confidence_score"] = 80.0

        graph, relationships = build_graph(employees, [candidate, second])

        assert len(relationships) == 1
        assert graph.in_degree("E001") == 1


class TestCyclePrevention:

    def test_rejects_cycle_creating_edge(self, employees):
        candidates = [
            {
                "company_id": "C001",
                "employee_id": "E001",
                "candidate_manager_id": "E002",
                "confidence_score": 95.0,
                "explanation": "First relationship",
                "employee_seniority_rank": 1,
                "manager_seniority_rank": 3,
                "candidate_manager_title_missing": False,
                "employee_department": "Engineering",
                "candidate_manager_department": "Engineering",
            },
            {
                "company_id": "C001",
                "employee_id": "E002",
                "candidate_manager_id": "E001",
                "confidence_score": 90.0,
                "explanation": "Would create a cycle",
                "employee_seniority_rank": 1,
                "manager_seniority_rank": 3,
                "candidate_manager_title_missing": False,
                "employee_department": "Engineering",
                "candidate_manager_department": "Engineering",
            },
        ]

        # Use a separate dataset with two employees.
        two_employees = employees[employees["employee_id"].isin(["E001", "E002"])]

        graph, relationships = build_org_graph(
            two_employees,
            pd.DataFrame(candidates),
            company_id="C001",
            min_confidence=65,
        )

        assert nx.is_directed_acyclic_graph(graph)
        assert graph.number_of_edges() == 1
        assert len(relationships) == 1


class TestCompanyIsolation:

    def test_ignores_other_companies(self, employees, candidate):
        other_employee = {
            "employee_id": "E100",
            "company_id": "C002",
            "job_title": "Sales Manager",
            "department": "Sales",
        }

        all_employees = pd.concat(
            [employees, pd.DataFrame([other_employee])],
            ignore_index=True,
        )

        other_candidate = candidate.copy()
        other_candidate["company_id"] = "C002"
        other_candidate["employee_id"] = "E100"
        other_candidate["candidate_manager_id"] = "E002"

        graph, relationships = build_org_graph(
            all_employees,
            pd.DataFrame([candidate, other_candidate]),
            company_id="C001",
        )

        assert "E100" not in graph
        assert graph.number_of_nodes() == 4
        assert len(relationships) == 1


class TestGraphValidation:

    def test_valid_graph_passes_all_structural_checks(self):
        graph = nx.DiGraph()
        graph.add_edges_from(
            [
                ("CEO", "Director"),
                ("Director", "Manager"),
                ("Manager", "Employee"),
            ]
        )

        result = validate_org_graph(graph)

        assert result["is_directed_acyclic"] is True
        assert result["no_employee_has_multiple_managers"] is True
        assert result["no_self_reporting"] is True
        assert result["all_nodes_in_graph"] is True
        assert result["number_of_nodes"] == 4
        assert result["number_of_relationships"] == 3
        assert result["number_of_roots"] == 1

    def test_detects_cycle(self):
        graph = nx.DiGraph()
        graph.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])

        result = validate_org_graph(graph)

        assert result["is_directed_acyclic"] is False

    def test_detects_multiple_managers(self):
        graph = nx.DiGraph()
        graph.add_edges_from(
            [
                ("Manager1", "Employee"),
                ("Manager2", "Employee"),
            ]
        )

        result = validate_org_graph(graph)

        assert result["no_employee_has_multiple_managers"] is False

    def test_detects_self_reporting(self):
        graph = nx.DiGraph()
        graph.add_edge("Employee", "Employee")

        result = validate_org_graph(graph)

        assert result["no_self_reporting"] is False

    def test_empty_graph_reports_no_nodes(self):
        graph = nx.DiGraph()

        result = validate_org_graph(graph)

        assert result["all_nodes_in_graph"] is False
        assert result["number_of_nodes"] == 0
        assert result["number_of_relationships"] == 0


class TestInputValidation:

    def test_invalid_confidence_threshold_raises_error(self, employees, candidate):
        with pytest.raises(ValueError, match="min_confidence"):
            build_graph(employees, [candidate], min_confidence=101)

    def test_negative_confidence_threshold_raises_error(self, employees, candidate):
        with pytest.raises(ValueError, match="min_confidence"):
            build_graph(employees, [candidate], min_confidence=-1)

    def test_missing_employee_column_raises_error(self, employees, candidate):
        incomplete = employees.drop(columns=["department"])

        with pytest.raises(ValueError, match="Missing employee columns"):
            build_graph(incomplete, [candidate])

    def test_missing_candidate_column_raises_error(self, employees, candidate):
        incomplete = pd.DataFrame([candidate]).drop(columns=["confidence_score"])

        with pytest.raises(ValueError, match="Missing candidate columns"):
            build_graph(employees, incomplete)

    def test_empty_company_raises_error(self, employees, candidate):
        with pytest.raises(ValueError, match="No employees found"):
            build_org_graph(
                employees,
                pd.DataFrame([candidate]),
                company_id="C999",
            )

    def test_duplicate_employee_ids_raise_error(self, employees, candidate):
        duplicate = pd.concat(
            [employees, employees.iloc[[0]]],
            ignore_index=True,
        )

        with pytest.raises(ValueError, match="Employee IDs must be unique"):
            build_graph(duplicate, [candidate])
