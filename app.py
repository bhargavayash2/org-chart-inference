from pathlib import Path

import pandas as pd
import streamlit as st
import networkx as nx
import plotly.graph_objects as go

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Org Chart Inference",
    page_icon="🏢",
    layout="wide",
)

st.title("🏢 Org Chart Inference Platform")

st.markdown("""
    Discover organizational structures using explainable AI
    and graph-based manager inference.

    **Powered by Python, NetworkX, Pandas and Streamlit**
    """)

st.divider()


# --------------------------------------------------
# DATA FILE PATHS
# --------------------------------------------------

DATA_DIR = Path("data/processed")

EMPLOYEES_PATH = DATA_DIR / "employees_cleaned.parquet"
COMPANIES_PATH = DATA_DIR / "companies.parquet"
EDGES_PATH = DATA_DIR / "experimental_inferred_edges.parquet"
STATS_PATH = DATA_DIR / "experimental_company_graph_stats.parquet"


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------


@st.cache_data
def load_data():

    employees = pd.read_parquet(EMPLOYEES_PATH)
    companies = pd.read_parquet(COMPANIES_PATH)
    edges = pd.read_parquet(EDGES_PATH)
    company_stats = pd.read_parquet(STATS_PATH)

    return employees, companies, edges, company_stats


required_files = [
    EMPLOYEES_PATH,
    COMPANIES_PATH,
    EDGES_PATH,
    STATS_PATH,
]

missing_files = [str(path) for path in required_files if not path.exists()]

if missing_files:

    st.error("Required dataset files are missing.")

    for file in missing_files:
        st.write(f"- {file}")

    st.info(
        "Run the data generation, cleaning and experimental "
        "inference pipelines before opening this dashboard."
    )

    st.stop()


try:
    employees, companies, edges, company_stats = load_data()

except Exception as error:

    st.error(f"Unable to load project data: {error}")
    st.stop()


# --------------------------------------------------
# CALCULATE DASHBOARD METRICS
# --------------------------------------------------

total_employees = len(employees)
total_companies = len(companies)
total_relationships = len(edges)

unassigned_employees = total_employees - total_relationships

assignment_coverage = (
    total_relationships / total_employees * 100 if total_employees else 0
)

average_confidence = edges["confidence_score"].mean() if not edges.empty else 0

valid_dags = int(company_stats["is_directed_acyclic"].sum())

valid_manager_assignments = int(company_stats["no_multiple_managers"].sum())

valid_self_reporting_checks = int(company_stats["no_self_reporting"].sum())


# --------------------------------------------------
# PLATFORM OVERVIEW
# --------------------------------------------------

st.subheader("📊 Platform Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Companies",
    f"{total_companies:,}",
)

col2.metric(
    "Employees",
    f"{total_employees:,}",
)

col3.metric(
    "Inferred Relationships",
    f"{total_relationships:,}",
)

col4.metric(
    "Assignment Coverage",
    f"{assignment_coverage:.2f}%",
)


# --------------------------------------------------
# INFERENCE PERFORMANCE
# --------------------------------------------------

st.divider()

st.subheader("🎯 Inference Performance")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Average Confidence",
    f"{average_confidence:.2f}%",
)

col2.metric(
    "Employees Without Inferred Managers",
    f"{unassigned_employees:,}",
)

col3.metric(
    "Valid Company Graphs",
    f"{valid_dags:,} / {total_companies:,}",
)


# --------------------------------------------------
# GRAPH VALIDATION
# --------------------------------------------------

st.divider()

st.subheader("✅ Graph Validation")

validation_col1, validation_col2, validation_col3 = st.columns(3)

with validation_col1:
    if valid_dags == total_companies:
        st.success("All company graphs are acyclic.")
    else:
        st.warning("Some company graphs have validation failures.")

with validation_col2:
    if valid_manager_assignments == total_companies:
        st.success("No employees have multiple managers.")
    else:
        st.warning("Multiple-manager validation failures detected.")

with validation_col3:
    if valid_self_reporting_checks == total_companies:
        st.success("No self-reporting relationships.")
    else:
        st.warning("Self-reporting validation failures detected.")


# --------------------------------------------------
# COMPANY STATISTICS
# --------------------------------------------------

st.divider()

st.subheader("🏢 Company Statistics")

left_col, right_col = st.columns(2)

with left_col:

    st.markdown("**Largest companies by employee count**")

    largest_companies = (
        companies[["company_name", "company_id", "employee_count", "industry"]]
        .sort_values("employee_count", ascending=False)
        .head(10)
        .set_index("company_name")
    )

    st.bar_chart(
        largest_companies["employee_count"],
        horizontal=True,
    )

with right_col:

    st.markdown("**Company stage distribution**")

    stage_distribution = (
        companies["company_stage"]
        .value_counts()
        .rename_axis("Company Stage")
        .reset_index(name="Companies")
    )

    st.bar_chart(stage_distribution.set_index("Company Stage"))


# --------------------------------------------------
# COMPANY DATA TABLE
# --------------------------------------------------

st.markdown("**Company-level inference details**")

company_overview = company_stats.merge(
    companies[["company_id", "company_name", "industry", "company_stage"]],
    on="company_id",
    how="left",
)

display_columns = [
    "company_id",
    "company_name",
    "industry",
    "company_stage",
    "employee_count",
    "inferred_relationship_count",
    "employees_without_manager",
    "root_count",
]

st.dataframe(
    company_overview[display_columns].sort_values(
        "employee_count",
        ascending=False,
    ),
    width="stretch",
    hide_index=True,
)


# --------------------------------------------------
# CANDIDATE TYPE DISTRIBUTION
# --------------------------------------------------

st.divider()

st.subheader("🧠 Inference Methodology")

candidate_distribution = (
    edges["candidate_type"]
    .value_counts()
    .rename_axis("Candidate Type")
    .reset_index(name="Relationships")
)

st.markdown("**Inferred relationship types**")

st.dataframe(
    candidate_distribution,
    width="stretch",
    hide_index=True,
)

st.caption(
    "Manager relationships are inferred using candidate generation, "
    "weighted scoring and maximum-spanning-arborescence graph optimization. "
    "Ground-truth relationships are not used by the inference pipeline."
)


# --------------------------------------------------
# INTERACTIVE ORGANIZATIONAL CHART
# --------------------------------------------------

st.divider()

st.subheader("🌳 Interactive Organizational Chart")

st.markdown(
    "Select a company, explore its reporting hierarchy, "
    "and inspect individual employee relationships."
)


# --------------------------------------------------
# COMPANY SELECTION
# --------------------------------------------------

company_options = companies[["company_id", "company_name"]].drop_duplicates()

company_options = company_options.sort_values("company_name")

selected_company_id = st.selectbox(
    "Select a company",
    options=company_options["company_id"].tolist(),
    format_func=lambda company_id: (
        company_options.loc[
            company_options["company_id"] == company_id,
            "company_name",
        ].iloc[0]
        + f" ({company_id})"
    ),
    key="org_chart_company",
)


# --------------------------------------------------
# FILTER SELECTED COMPANY DATA
# --------------------------------------------------

company_employees = employees[employees["company_id"] == selected_company_id].copy()

company_edges = edges[edges["company_id"] == selected_company_id].copy()

company_name = company_options.loc[
    company_options["company_id"] == selected_company_id,
    "company_name",
].iloc[0]

st.caption(
    f"Company: {company_name} | "
    f"Employees: {len(company_employees):,} | "
    f"Inferred relationships: {len(company_edges):,}"
)


# --------------------------------------------------
# BUILD COMPANY GRAPH
# --------------------------------------------------

org_graph = nx.DiGraph()

for _, employee in company_employees.iterrows():
    org_graph.add_node(employee["employee_id"])

for _, edge in company_edges.iterrows():
    org_graph.add_edge(
        edge["manager_employee_id"],
        edge["employee_id"],
    )


# --------------------------------------------------
# CALCULATE HIERARCHICAL NODE POSITIONS
# --------------------------------------------------


def calculate_tree_positions(graph, employee_lookup):

    positions = {}
    current_x = 0

    roots = sorted(
        [node for node in graph.nodes if graph.in_degree(node) == 0],
        key=lambda node: str(node),
    )

    def assign_position(node, depth):

        nonlocal current_x

        children = sorted(
            list(graph.successors(node)),
            key=lambda child: employee_lookup.get(child, {}).get(
                "full_name", str(child)
            ),
        )

        if not children:
            x = current_x
            current_x += 1
        else:
            child_positions = []

            for child in children:
                child_x = assign_position(child, depth + 1)
                child_positions.append(child_x)

            x = sum(child_positions) / len(child_positions)

        positions[node] = (x, -depth)

        return x

    for root in roots:
        assign_position(root, 0)
        current_x += 2

    return positions


employee_lookup = company_employees.set_index("employee_id").to_dict("index")

positions = calculate_tree_positions(
    org_graph,
    employee_lookup,
)


# --------------------------------------------------
# CREATE PLOTLY GRAPH
# --------------------------------------------------

edge_x = []
edge_y = []

for manager_id, employee_id in org_graph.edges():

    x0, y0 = positions[manager_id]
    x1, y1 = positions[employee_id]

    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])


edge_trace = go.Scatter(
    x=edge_x,
    y=edge_y,
    mode="lines",
    line=dict(
        width=1.2,
        color="#AAB4C3",
    ),
    hoverinfo="none",
    showlegend=False,
)


node_ids = list(org_graph.nodes())

node_x = []
node_y = []
node_names = []
node_hover = []
node_colors = []

department_colors = {
    department: color
    for department, color in zip(
        company_employees["department"].dropna().unique(),
        [
            "#2563EB",
            "#059669",
            "#D97706",
            "#7C3AED",
            "#DB2777",
            "#0891B2",
            "#65A30D",
            "#EA580C",
            "#4F46E5",
            "#0F766E",
        ]
        * 2,
    )
}

for employee_id in node_ids:

    employee = employee_lookup[employee_id]

    x, y = positions[employee_id]

    node_x.append(x)
    node_y.append(y)

    name = employee.get("full_name", "Unknown")
    title = employee.get("job_title", "Unknown")
    department = employee.get("department", "Unknown")

    node_names.append(name)

    node_hover.append(
        f"<b>{name}</b><br>"
        f"Employee ID: {employee_id}<br>"
        f"Job Title: {title}<br>"
        f"Department: {department}"
    )

    node_colors.append(department_colors.get(department, "#64748B"))


node_trace = go.Scatter(
    x=node_x,
    y=node_y,
    mode="markers",
    marker=dict(
        size=15,
        color=node_colors,
        line=dict(
            width=1.5,
            color="white",
        ),
    ),
    text=node_hover,
    customdata=node_ids,
    hovertemplate="%{text}<extra></extra>",
    selected=dict(
        marker=dict(
            size=23,
            color="#F59E0B",
            opacity=1,
        )
    ),
    unselected=dict(marker=dict(opacity=0.85)),
    showlegend=False,
)


fig = go.Figure(data=[edge_trace, node_trace])

fig.update_layout(
    title=f"Organizational Hierarchy — {company_name}",
    height=750,
    template="plotly_white",
    showlegend=False,
    hovermode="closest",
    dragmode="pan",
    margin=dict(l=30, r=30, t=65, b=30),
    xaxis=dict(
        visible=False,
        fixedrange=False,
    ),
    yaxis=dict(
        visible=False,
        fixedrange=False,
    ),
)


# --------------------------------------------------
# DISPLAY INTERACTIVE CHART
# --------------------------------------------------

st.caption(
    "Tip: Hover over an employee for details. "
    "Click an employee to inspect their reporting relationship. "
    "Use the chart controls to zoom and pan."
)

chart_event = st.plotly_chart(
    fig,
    width="stretch",
    on_select="rerun",
    selection_mode="points",
    key="org_chart_plot",
)


# --------------------------------------------------
# EMPLOYEE SEARCH
# --------------------------------------------------

st.markdown("### 🔍 Employee Search")

search_query = st.text_input(
    "Search by employee name, ID, job title or department",
    key="org_employee_search",
)

searchable_employees = company_employees.copy()

if search_query.strip():

    query = search_query.strip().lower()

    search_columns = [
        "employee_id",
        "full_name",
        "job_title",
        "department",
    ]

    mask = pd.Series(
        False,
        index=searchable_employees.index,
    )

    for column in search_columns:
        mask |= (
            searchable_employees[column]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.contains(query, regex=False)
        )

    searchable_employees = searchable_employees[mask]

st.dataframe(
    searchable_employees[
        [
            "employee_id",
            "full_name",
            "job_title",
            "department",
        ]
    ].head(100),
    width="stretch",
    hide_index=True,
)

if search_query.strip():
    st.caption(
        f"Found {len(searchable_employees):,} matching employees. "
        "The table displays up to 100 results."
    )


# --------------------------------------------------
# EMPLOYEE SELECTION
# --------------------------------------------------

selected_employee_id = None

selected_points = (
    chart_event.selection.points if chart_event and chart_event.selection else []
)

if selected_points:

    selected_employee_id = selected_points[-1].get("customdata")

if selected_employee_id not in employee_lookup:
    selected_employee_id = None


if not searchable_employees.empty:

    search_employee_options = searchable_employees["employee_id"].tolist()

    selected_by_search = st.selectbox(
        "Or select an employee from search results",
        options=[""] + search_employee_options,
        format_func=lambda employee_id: (
            "Select an employee"
            if employee_id == ""
            else (employee_lookup[employee_id]["full_name"] + f" ({employee_id})")
        ),
        key="org_employee_selector",
    )

    if selected_by_search:
        selected_employee_id = selected_by_search


# --------------------------------------------------
# EMPLOYEE AND MANAGER DETAILS
# --------------------------------------------------

if selected_employee_id:

    st.divider()

    st.markdown("### 👤 Employee Details")

    employee = employee_lookup[selected_employee_id]

    manager_relationship = company_edges[
        company_edges["employee_id"] == selected_employee_id
    ]

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("**Selected Employee**")

        st.write("Name:", employee["full_name"])
        st.write("Employee ID:", selected_employee_id)
        st.write("Job Title:", employee["job_title"])
        st.write("Department:", employee["department"])

    with col2:

        st.markdown("**Inferred Manager**")

        if not manager_relationship.empty:

            relationship = manager_relationship.iloc[0]

            manager_id = relationship["manager_employee_id"]
            manager = employee_lookup[manager_id]

            st.write("Name:", manager["full_name"])
            st.write("Employee ID:", manager_id)
            st.write("Job Title:", manager["job_title"])

            st.metric(
                "Inference Confidence",
                f"{relationship['confidence_score']:.2f}%",
            )

            st.info(relationship["explanation"])

        else:

            st.warning(
                "No manager was inferred for this employee. "
                "This employee is currently a root in the inferred graph."
            )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Org Chart Inference Platform | Experimental Graph Optimizer | "
    "Synthetic Dataset | 500 Companies"
)
