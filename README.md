# 🏢 Org Chart Inference Platform

An explainable organizational hierarchy inference platform that predicts employee-manager reporting relationships from employee and company information.

The platform combines synthetic data generation, data cleaning, candidate generation, weighted scoring, graph optimization and evaluation to reconstruct organizational structures from incomplete and imperfect employee records.

## Project Overview

Organizations often have employee data distributed across different systems, with incomplete job titles, missing department information and inconsistent naming conventions.

This project addresses the challenge of reconstructing organizational reporting structures when explicit manager relationships are unavailable.

The platform:

- Generates synthetic companies and employee records.
- Creates realistic organizational hierarchies for ground-truth evaluation.
- Introduces controlled data corruption to simulate imperfect real-world records.
- Generates potential manager candidates for each employee.
- Scores candidate relationships using explainable business rules.
- Uses graph optimization to produce valid organizational hierarchies.
- Evaluates inferred relationships against separately stored ground truth.
- Provides an interactive Streamlit dashboard to explore inferred organizational structures.

## Key Results

**Dataset**

| Metric | Result |
|---|---:|
| Companies | 500 |
| Employees | 152,894 |
| Ground-truth relationships | 152,394 |
| Inferred relationships | 146,478 |
| Employees without inferred managers | 6,416 |
| Assignment coverage | 95.80% |

**Experimental optimizer performance**

| Metric | Baseline | Experimental |
|---|---:|---:|
| True Positives | 82,669 | 84,784 |
| False Positives | 63,802 | 61,694 |
| False Negatives | 69,725 | 67,610 |
| Precision | 56.44% | 57.88% |
| Recall | 54.25% | 55.63% |
| F1 Score | 55.32% | 56.74% |

The experimental maximum-spanning-arborescence approach improved F1 by approximately **1.42 percentage points** compared with the baseline greedy graph builder.

The experimental optimizer also reduced false positives by 2,108 and increased correctly inferred relationships by 2,115.

These results are measured against the synthetic ground-truth dataset, not a real-world production benchmark.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core development |
| Pandas | Data processing and analysis |
| NumPy | Numerical operations |
| Faker | Synthetic data generation |
| DuckDB | Analytical database capability |
| PyArrow / Parquet | Columnar data storage |
| NetworkX | Graph construction and optimization |
| RapidFuzz | String similarity capability |
| Streamlit | Interactive application |
| Plotly | Organizational chart visualization |
| Scikit-learn | Evaluation and analytical utilities |
| Pytest | Automated testing |

## Architecture

```text
                 SYNTHETIC DATA GENERATION
                           |
                           v
                 Company Generation
                           |
                           v
               Organizational Hierarchy
                           |
                           v
                  Employee Generation
                           |
                           v
                   Data Corruption
                           |
              +------------+------------+
              |                         |
              v                         v
       Raw Employee Data          Ground Truth Edges
              |                    (Evaluation Only)
              v                         |
       Title Normalization              |
              |                         |
              v                         |
         Data Cleaning                  |
              |                         |
              v                         |
      Candidate Generation              |
              |                         |
              v                         |
        Candidate Scoring               |
              |                         |
              v                         |
        Graph Optimization              |
              |                         |
              v                         |
      Inferred Relationships            |
              |                         |
              +------------+------------+
                           |
                           v
                    Evaluation
                           |
                           v
                  Streamlit Dashboard
```

**Important:** Ground-truth relationships are kept separate from the inference pipeline. They are used for evaluation and diagnostic analysis only.

## Project Structure

```text
org-chart-inference/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── config/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── ground_truth/
│
├── docs/
│   └── methodology.md
│
├── src/
│   ├── data_generation/
│   │   ├── company_generator.py
│   │   ├── hierarchy_generator.py
│   │   ├── employee_generator.py
│   │   ├── data_corruption.py
│   │   └── generate_dataset.py
│   │
│   ├── processing/
│   │   ├── title_normalizer.py
│   │   └── data_cleaner.py
│   │
│   ├── inference/
│   │   ├── candidate_generator.py
│   │   ├── scoring_engine.py
│   │   ├── graph_builder.py
│   │   ├── experimental_graph_optimizer.py
│   │   ├── inference_pipeline.py
│   │   └── experimental_inference_pipeline.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── candidate_recall.py
│   │   ├── ranking_evaluation.py
│   │   ├── threshold_analysis.py
│   │   ├── manager_failure_diagnostics.py
│   │   ├── graph_failure_breakdown.py
│   │   ├── compare_experimental_optimizer.py
│   │   └── additional diagnostic scripts
│   │
│   └── database/
│
└── tests/
    ├── test_candidate_generator.py
    ├── test_scoring_engine.py
    ├── test_graph_builder.py
    └── test_experimental_graph_optimizer.py
```

## Getting Started

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd org-chart-inference
```

### 2. Create a virtual environment

Python 3.12 is the development environment used for this project.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Running the Pipeline

Run the following commands from the project root directory, in order.

### Step 1: Generate the synthetic dataset

```bash
python -m src.data_generation.generate_dataset
```

This generates:

- 500 synthetic companies.
- Employee records and organizational positions.
- Ground-truth manager relationships.
- Corrupted employee data for inference.

Output files:

```text
data/raw/employees.parquet
data/processed/companies.parquet
data/ground_truth/ground_truth_edges.parquet
```

### Step 2: Normalize and clean employee data

```bash
python -m src.processing.data_cleaner
```

This stage:

- Normalizes employee job titles.
- Standardizes whitespace.
- Converts blank strings into missing values.
- Removes duplicate employee IDs.
- Validates essential identifiers.
- Preserves records with incomplete information.

Output:

```text
data/processed/employees_cleaned.parquet
```

### Step 3: Run experimental inference

```bash
python -m src.inference.experimental_inference_pipeline
```

The inference pipeline:

1. Generates potential manager candidates.
2. Calculates candidate confidence scores.
3. Applies graph constraints.
4. Uses maximum-spanning-arborescence optimization.
5. Validates the resulting organizational graph.
6. Saves inferred relationships and company statistics.

Output files:

```text
data/processed/experimental_inferred_edges.parquet
data/processed/experimental_company_graph_stats.parquet
```

## Inference Methodology

### 1. Candidate Generation

Potential managers are identified using employee attributes such as:

- Job title.
- Department.
- Relative seniority.
- Leadership indicators.

The candidate-generation process limits the number of potential manager relationships considered for each employee.

A missing-title fallback is also included to handle incomplete employee records.

### 2. Explainable Candidate Scoring

Each potential manager relationship receives a weighted confidence score.

| Scoring component | Weight |
|---|---:|
| Department compatibility | 30% |
| Seniority compatibility | 25% |
| Job-title compatibility | 25% |
| Manager-role suitability | 20% |

The final score is calculated from the weighted component scores.

Each inferred relationship also includes an explanation describing the evidence supporting the prediction.

### 3. Graph Construction

Two approaches were implemented and evaluated:

**Baseline: Greedy Graph Builder**

- Processes candidate relationships using confidence scores.
- Selects the first valid manager candidate.
- Enforces seniority and graph constraints.
- Prevents cycles and multiple managers.

**Experimental: Maximum Spanning Arborescence**

- Constructs a directed candidate graph.
- Uses a virtual root to permit employees without inferred managers.
- Optimizes the global graph using NetworkX maximum-spanning-arborescence functionality.
- Enforces one incoming manager relationship per assigned employee.
- Validates the resulting graph for cycles and self-reporting.

The experimental optimizer produced better overall precision, recall and F1 on the synthetic evaluation dataset.

Further methodology and diagnostic findings are documented in [docs/methodology.md](docs/methodology.md).

## Evaluation

The evaluation framework compares inferred manager relationships with the separately stored ground-truth relationships.

The main metrics are:

- **Precision:** Proportion of inferred relationships that are correct.
- **Recall:** Proportion of actual manager relationships successfully inferred.
- **F1 Score:** Harmonic mean of precision and recall.
- **Assignment Coverage:** Proportion of employees assigned an inferred manager.

Additional analysis includes:

- Candidate recall and ranking performance.
- Confidence threshold analysis.
- Candidate limit analysis.
- Seniority-based candidate exclusions.
- Missing manager-title diagnostics.
- Graph construction failure analysis.
- Optimizer regression analysis.
- Tie and identical-evidence analysis.

### Graph Integrity

The inference pipeline validates:

- Directed acyclic graph structure.
- Maximum one manager per employee.
- No self-reporting relationships.
- Employee and manager membership in the same company.
- Correct company-level relationship counts.

All 500 experimental company graphs passed the graph validation checks.

## Running Tests

Run the automated test suite:

```bash
python -m pytest -v
```

The project currently has **148 passing automated tests** across:

- Candidate generation.
- Candidate scoring.
- Baseline graph construction.
- Experimental graph optimization.

## Launching the Dashboard

After generating and processing the data and running the experimental inference pipeline:

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

The dashboard provides:

- Overall company and employee metrics.
- Inference coverage and confidence statistics.
- Graph validation status.
- Company-level statistics and visualizations.
- Interactive organizational hierarchy exploration.
- Employee search by name, ID, job title and department.
- Employee and inferred manager details.
- Confidence scores and inference explanations.

## Data and Reproducibility

Generated Parquet and CSV datasets are intentionally excluded from Git to keep the repository lightweight.

The repository includes the synthetic data generator so the dataset can be recreated locally.

The ground-truth dataset is stored separately from inference input and must not be used by the inference algorithms.

To reproduce the complete workflow, run the three pipeline commands in the order described above.

## Limitations

- The dataset is synthetic and may not capture all complexities of real organizational structures.
- Job-title ambiguity can make manager identification difficult.
- Missing or inconsistent employee attributes reduce inference reliability.
- Equal candidate scores can produce ambiguous manager assignments.
- The experimental optimizer improves aggregate metrics but does not improve every individual company.
- Confidence scores represent model scoring evidence, not calibrated probabilities of correctness.
- Evaluation results on synthetic data should not be interpreted as guaranteed performance on real company data.

## Future Improvements

- Improve ambiguous job-title and leadership-role classification.
- Introduce richer candidate-ranking signals.
- Improve confidence calibration.
- Add department-specific and hierarchy-level exploration.
- Evaluate against real-world organizational datasets where permitted.
- Improve scalability for substantially larger organizations.

## Author

**Yash Bhargava**

Data Engineer | Data & AI Engineering

---

*Built as a data engineering and organizational hierarchy inference project, demonstrating synthetic data generation, data quality, explainable scoring, graph optimization, evaluation and interactive visualization.*