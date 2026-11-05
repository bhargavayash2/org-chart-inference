# Org Chart Inference — Methodology

## 1. Project Overview

The objective of this project is to infer employee reporting relationships and reconstruct organizational hierarchies from employee data that does not explicitly contain manager relationships.

The solution uses a multi-stage data engineering and graph-based inference pipeline. It generates a synthetic employee dataset, introduces realistic data quality issues, generates potential manager candidates, scores candidate relationships, and constructs organizational graphs.

The implementation also includes an evaluation framework that compares inferred relationships against separately stored ground-truth relationships.

### Key objectives

- Generate a realistic synthetic dataset containing multiple companies and organizational structures.
- Introduce missing and inconsistent employee attributes.
- Identify potential managers using employee attributes.
- Rank potential manager relationships using a weighted scoring approach.
- Construct valid organizational graphs without cycles or multiple direct managers.
- Evaluate inference quality using precision, recall, F1-score, and assignment coverage.
- Compare a greedy graph construction approach with a maximum spanning arborescence optimizer.

---

## 2. Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python 3.12 |
| Data processing | Pandas, NumPy |
| Synthetic data generation | Faker |
| Data storage | Parquet |
| Analytical database | DuckDB |
| Candidate and relationship graph | NetworkX |
| Streamlit application | Streamlit |
| Visualization | Plotly |
| Evaluation | Python, Pandas, scikit-learn |
| Testing | Pytest |

---

## 3. Synthetic Dataset Generation

A synthetic dataset was generated to simulate employee structures across multiple organizations.

### Dataset characteristics

| Metric | Value |
|---|---:|
| Companies | 500 |
| Total employees | 152,894 |
| Minimum employees per company | 100 |
| Maximum employees per company | 498 |
| Ground-truth manager relationships | 152,394 |

Each company contains a generated organizational hierarchy with roles such as:

- CEO
- C-level executives
- Vice Presidents
- Directors and Department Heads
- Managers
- Team Leads
- Individual Contributors

Departments include Engineering, Data, Product, Sales, Marketing, Finance, HR, Operations, Customer Success, and Security.

The hierarchy generator creates the original reporting structure. Employee records are then generated separately, and the corresponding manager relationships are stored in a separate ground-truth dataset.

### Data corruption

To simulate imperfect real-world employee data, selected attributes are randomly corrupted.

| Attribute | Configured corruption rate |
|---|---:|
| Department | 5% |
| Job title | 3% |
| City | 5% |
| Email | 2% |
| Inconsistent job title | 15% |

The inference process operates on the corrupted employee dataset.

### Data separation

Three primary datasets are generated:

- `data/raw/employees.parquet` — employee data used by the inference pipeline.
- `data/processed/companies.parquet` — company information.
- `data/ground_truth/ground_truth_edges.parquet` — actual employee-manager relationships used exclusively for evaluation.

**Ground truth is never used for candidate generation, scoring, or graph construction.** It is only accessed by evaluation and diagnostic scripts.

---

## 4. Data Processing and Cleaning

The raw employee dataset is processed before inference.

The cleaning process includes:

1. Validating employee and company identifiers.
2. Removing duplicate employee records.
3. Trimming whitespace from string fields.
4. Converting blank values into missing values.
5. Normalizing job titles.
6. Preserving missing attributes instead of replacing them with assumed values.

Job title normalization standardizes common abbreviations and variations, such as:

- Sr → Senior
- Jr → Junior
- Eng → Engineer
- Mgr → Manager
- Dir → Director
- VP → Vice President

The normalized title is stored separately from the original title so that the original source information remains available.

All 152,894 employee records were retained after cleaning, with no duplicate employee IDs.

---

## 5. Candidate Manager Generation

The first inference stage identifies potential managers for every employee.

Instead of comparing every employee against every other employee in a company, candidate generation applies filtering rules to reduce the search space.

### Candidate generation logic

For employees with known job titles:

- The employee cannot be their own manager.
- Potential managers must generally have a higher seniority rank.
- Employees from the same department are prioritized.
- Candidates are ordered using department alignment and seniority gap.
- A maximum of 10 candidates is retained per employee.

The seniority ranking considers title indicators such as:

| Role indicator | Seniority rank |
|---|---:|
| Individual Contributor / Default | 1 |
| Lead | 2 |
| Manager | 3 |
| Director / Head / Vice President | 4 |
| Chief | 5 |
| CEO | 6 |

Some titles require special handling because their wording can represent different organizational responsibilities. For example, Customer Success Manager can represent an individual contributor role in the generated dataset. Such cases are handled through title-specific overrides.

### Missing-title fallback

Job titles are not always available in real-world employee datasets.

For employees with missing titles, the pipeline uses department information to generate fallback candidates.

This fallback is intentionally more conservative and requires department alignment during graph construction.

Missing-title candidates are treated separately from candidates with known titles so their behavior can be evaluated independently.

### Candidate generation evaluation

The final candidate-generation diagnostic found:

| Metric | Value |
|---|---:|
| Actual manager found in candidate list | 144,449 |
| Actual manager not found | 7,945 |
| Candidate recall | 94.79% |

This indicates that the actual manager is available in the candidate set for approximately 94.79% of ground-truth relationships.

Candidate generation is therefore a major determinant of final inference quality: a manager excluded at this stage cannot be recovered by the scoring or graph construction stages.

---

## 6. Candidate Scoring Engine

After generating potential managers, the scoring engine calculates a confidence score for each employee-manager candidate pair.

The score combines four signals.

### Scoring weights

| Signal | Weight |
|---|---:|
| Department similarity | 30% |
| Seniority compatibility | 25% |
| Job title similarity | 25% |
| Manager role indicator | 20% |
| **Total** | **100%** |

### 6.1 Department score

- Same department: 1.0
- Missing department information: 0.5
- Different departments: 0.0

### 6.2 Seniority score

The score is based on the seniority difference between the employee and candidate manager.

- Seniority gap of 1: 1.0
- Gap of 2: 0.8
- Gap of 3: 0.6
- Larger gap: 0.3
- Missing seniority information: 0.5

### 6.3 Job title similarity

Job titles are mapped into role families, such as:

- Technology and Engineering
- Data
- Sales
- Marketing
- Finance
- Human Resources
- Operations
- Product
- Security
- Customer Success

The scoring logic rewards candidates belonging to the same role family.

- Same role family: 1.0
- Either title missing: 0.5
- Different role families: 0.2

### 6.4 Manager role indicator

Candidates with titles indicating leadership responsibilities receive higher scores.

Examples include:

- CEO or Chief: 1.0
- Vice President: 0.95
- Director or Head: 0.90
- Manager: 0.85
- Lead: 0.75
- Other roles: 0.30
- Missing manager title: 0.50

### Final confidence score

The final score is calculated as:

`Confidence Score = 100 × (0.30 × Department Score + 0.25 × Seniority Score + 0.25 × Title Score + 0.20 × Manager Role Score)`

The result is represented as a score between 0 and 100.

An explanation is also generated for each candidate to make the scoring decision easier to inspect.

---

## 7. Graph Construction Approaches

Two graph construction strategies were implemented and compared.

### 7.1 Greedy graph construction — baseline

The baseline approach processes candidate relationships in descending confidence-score order.

For each employee:

1. Sort potential managers by confidence score.
2. Check whether the candidate exists in the company graph.
3. Reject self-reporting relationships.
4. Reject candidates below the confidence threshold.
5. Apply seniority validation.
6. Apply department validation for missing-title fallback candidates.
7. Check whether adding the edge would introduce a cycle.
8. Accept the first valid candidate.

The graph is represented using a directed NetworkX graph.

The relationship direction is:

`Manager → Employee`

The baseline implementation enforces:

- No self-reporting relationships.
- At most one direct manager per employee.
- No directed cycles.
- Valid employee and manager nodes.

Although this approach is straightforward, it makes decisions locally. Once a manager is selected for an employee, the algorithm does not reconsider that assignment in the context of the entire company's organizational structure.

### 7.2 Experimental graph optimizer — maximum spanning arborescence

An alternative approach was implemented using NetworkX's maximum spanning arborescence algorithm.

The optimizer introduces a virtual root node connected to every employee.

Candidate manager relationships are represented as directed edges:

`Manager → Employee`

Each candidate edge receives its confidence score as its weight.

Virtual-root edges receive a weight slightly below the minimum confidence threshold. This allows the optimizer to leave an employee without an inferred manager when no suitable candidate edge is selected.

The maximum spanning arborescence algorithm then selects a globally optimized set of edges while respecting the graph's tree structure.

This approach differs from greedy construction because it evaluates the combination of selected relationships across the graph rather than accepting the first valid candidate independently for each employee.

The resulting graph is validated to ensure:

- No cycles.
- No self-reporting relationships.
- No employee has multiple direct managers.
- All expected employee nodes are retained.

The optimizer does not directly maximize ground-truth accuracy. It maximizes the total weight of the selected graph edges, subject to the graph constraints.

---

## 8. Evaluation Methodology

Ground-truth relationships are used only after inference to measure performance.

The following metrics are calculated:

### Precision

`Precision = TP / (TP + FP)`

Measures how many inferred manager relationships are correct.

### Recall

`Recall = TP / (TP + FN)`

Measures how many actual manager relationships were successfully inferred.

### F1-score

`F1 = 2 × (Precision × Recall) / (Precision + Recall)`

Balances precision and recall.

### Assignment coverage

`Assignment Coverage = Inferred Relationships / Total Employees`

Measures the proportion of employees assigned an inferred manager.

Assignment coverage is not an accuracy metric. A system can achieve high coverage while assigning incorrect managers.

---

## 9. Baseline vs Experimental Optimizer Results

Both approaches were evaluated against the same ground-truth dataset.

### Overall results

| Metric | Greedy baseline | Experimental optimizer |
|---|---:|---:|
| Inferred relationships | 146,471 | 146,478 |
| True positives | 82,669 | 84,784 |
| False positives | 63,802 | 61,694 |
| False negatives | 69,725 | 67,610 |
| Precision | 56.44% | 57.88% |
| Recall | 54.25% | 55.63% |
| F1-score | 55.32% | 56.74% |
| Assignment coverage | 96.11% | 96.12% |

### Improvements

The experimental optimizer achieved:

- 1.44 percentage-point improvement in precision.
- 1.38 percentage-point improvement in recall.
- 1.42 percentage-point improvement in F1-score.
- 2,115 additional correctly inferred relationships.
- 2,108 fewer false positives.

The number of inferred relationships remained almost identical, meaning the improvement was not simply caused by assigning more employees.

### Assignment comparison

Among employees assigned a manager by both approaches:

- 146,436 employees were assigned managers by both.
- 98,712 had the same inferred manager.
- 47,724 had different inferred managers.

The experimental optimizer therefore changed a substantial number of assignments. This is why the comparison includes not only overall metrics but also company-level and tie diagnostics.

---

## 10. Company-Level Evaluation

Overall metrics do not tell the complete story.

The company-level comparison showed:

| Outcome | Companies |
|---|---:|
| Improved F1-score | 299 |
| Unchanged F1-score | 71 |
| Regressed F1-score | 130 |
| **Total** | **500** |

The macro-average company metrics were:

| Metric | Greedy baseline | Experimental optimizer |
|---|---:|---:|
| Precision | 61.07% | 62.21% |
| Recall | 58.26% | 59.37% |
| F1-score | 59.62% | 60.75% |

The macro-average gives equal weight to each company, unlike the overall metrics, which are calculated across all employee relationships.

The experimental optimizer performs better overall, but it does not improve every company.

Some companies experience noticeable regressions. This indicates that a globally better objective does not guarantee better inference for every individual organization.

---

## 11. Tie and Ambiguity Analysis

A further diagnostic was performed to understand why the two graph construction approaches select different managers.

### Full-dataset findings

| Diagnostic | Result |
|---|---:|
| Companies analyzed | 500 |
| Employees | 152,894 |
| Employees with candidates | 152,372 |
| Employees with tied top candidates | 83,798 |
| Percentage with tied top candidates | 55.0% |
| Ties with identical component evidence | 83,462 |
| Aggregate-score collisions | 336 |
| Actual manager present among tied top candidates | 73,511 |
| Actual manager coverage within tied top candidates | 87.72% |

A tie occurs when multiple manager candidates receive the same highest confidence score for an employee.

The analysis distinguishes between:

1. **Identical evidence:** Candidates have the same component scores and therefore the same overall score.
2. **Aggregate-score collision:** Candidates have different component scores but arrive at the same weighted total.

### Changed assignments

Among the 47,724 assignments changed by the experimental optimizer:

| Diagnostic | Result |
|---|---:|
| Changed assignments with equal confidence scores | 47,724 |
| Identical-evidence ties | 47,631 |
| Aggregate-score collisions | 93 |
| Baseline correct assignments | 14,857 |
| Experimental correct assignments | 16,964 |

All changed assignments had equal selected confidence scores.

This indicates that the difference between the two graph construction approaches is largely driven by how they resolve equally scored candidates, rather than by one approach consistently selecting candidates with higher confidence scores.

The actual manager was present among tied top candidates for many employees. This suggests that the existing input attributes often do not contain enough information to distinguish between equally plausible candidates.

---

## 12. Additional Signal Analysis

Additional employee attributes were evaluated to determine whether they could improve manager selection.

The analysis included:

- Hire-date relationships.
- City similarity.
- Country similarity.

### Findings

| Signal | Actual manager | Inferred manager |
|---|---:|---:|
| Manager hired before employee | 49.66% | 49.96% |
| Same city | 43.45% | 43.63% |
| Same country | 100% | 100% |

The results show that these signals provide little separation between correct and incorrect manager assignments in the current synthetic dataset.

For example, the same-country signal is identical for both actual and inferred managers, so it does not help distinguish between candidates.

These signals were therefore not added to the scoring model.

This decision avoids adding complexity without demonstrated improvement.

---

## 13. Limitations and Trade-offs

### 13.1 Candidate generation limits

The actual manager is not present in the candidate list for every employee.

Candidate recall is 94.79%, so some relationships cannot be recovered by downstream scoring or graph optimization.

### 13.2 Ambiguous employee titles

Some job titles can represent either an individual contributor or a people manager.

For example, titles containing words such as Manager or Lead are not always sufficient to determine reporting responsibility.

Title-based seniority rules therefore require exceptions and may still produce incorrect candidate sets.

### 13.3 Missing employee attributes

Missing job titles and departments reduce the amount of evidence available for inference.

The fallback approach uses department alignment, but department similarity alone cannot reliably establish a reporting relationship.

### 13.4 Limited relationship signals

The current model primarily uses department, seniority, title similarity, and manager-role indicators.

Other potentially useful signals, such as reporting-line information, team membership, management-chain identifiers, or authoritative HR data, are not available in the synthetic input.

### 13.5 Equal-confidence candidates

A large proportion of employees have multiple candidates with identical top scores.

When candidates have identical evidence, the current model cannot reliably distinguish between them.

A deterministic tie-breaking rule can make output reproducible, but it cannot create missing information or guarantee that the selected candidate is correct.

### 13.6 Company-level regressions

Although the experimental optimizer improves overall F1-score, 130 companies regress.

This highlights the trade-off between optimizing the overall graph and optimizing every company's individual inference quality.

### 13.7 Synthetic data limitations

The dataset is generated using predefined organizational rules. Real companies may have more complex structures, matrix reporting, cross-functional management, contractors, dotted-line relationships, and inconsistent HR records.

Performance on synthetic data should therefore not be interpreted as a guarantee of performance on real-world company datasets.

---

## 14. Final Approach and Recommendation

Based on the evaluation results, the maximum spanning arborescence approach is preferred as the experimental final graph construction strategy.

It achieved:

- Higher precision.
- Higher recall.
- Higher F1-score.
- More correctly inferred relationships.
- Fewer false-positive relationships.

However, the results also show that graph optimization alone cannot resolve ambiguity when multiple candidates have identical evidence.

For a production implementation, the next improvements would focus on:

1. Improving candidate generation using additional employee and organizational signals.
2. Introducing richer manager-specific evidence.
3. Separating people-manager roles from similarly named individual contributor roles.
4. Calibrating confidence scores using a labeled validation dataset.
5. Introducing an explicit low-confidence or ambiguous outcome instead of forcing a manager assignment.
6. Monitoring performance at both company and overall dataset levels.
7. Validating the approach against real-world organizational data.

The preferred solution is therefore the experimental graph optimizer, supported by candidate-level diagnostics, graph validation, and company-level evaluation.

The objective is not simply to maximize the number of inferred relationships, but to produce a structurally valid organizational graph while being transparent about uncertainty and inference limitations.
