import streamlit as st

st.set_page_config(page_title="Org Chart Inference", page_icon="🏢", layout="wide")

st.title("🏢 Org Chart Inference Platform")

st.markdown("""
    ### Organizational Intelligence

    Discover reporting relationships across companies
    using explainable AI and graph-based inference.

    **Current Status:** Project initialization complete.
    """)

st.divider()

st.subheader("Platform Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Companies", "0")

with col2:
    st.metric("Employees", "0")

with col3:
    st.metric("Inferred Relationships", "0")

st.info(
    "Dataset generation and org chart inference "
    "will be implemented in the upcoming phases."
)
