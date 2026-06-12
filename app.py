import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from etl import extract_text, extract_findings, save_extracted_text
from embeddings import build_dataframe
from agents import set_dataframe, run_single_agent, run_workflow

os.environ["AWS_PROFILE"] = "GSB570-BedrockOnly-490332585640"
os.environ["AWS_DEFAULT_REGION"] = "us-west-2"

st.set_page_config(page_title="Compliance Monitoring System", layout="wide")
st.title("Labor Compliance Monitoring System")
st.caption("Garment Factory Audit Analysis powered by Generative AI")

st.sidebar.header("Data Management")

# load the default dataset
if st.sidebar.button("Load Default Compliance Data"):
    with st.spinner("loading and embedding compliance data..."):
        df = build_dataframe("data/compliance_data.txt")
        set_dataframe(df)
        st.session_state["df"] = df
        st.session_state["data_loaded"] = True
    st.sidebar.success(f"loaded {len(df)} chunks!")

# or upload your own
st.sidebar.markdown("---")
st.sidebar.subheader("Or Upload Your Own Data")
uploaded_file = st.sidebar.file_uploader("upload audit file", type=["txt", "pdf"])

if uploaded_file is not None:
    if st.sidebar.button("Process Uploaded File"):
        temp_path = f"data/{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("extracting text from file..."):
            raw_text = extract_text(temp_path)
            st.session_state["raw_text"] = raw_text
            st.sidebar.info(f"extracted {len(raw_text)} characters")

        with st.spinner("using AI to extract structured findings..."):
            findings = extract_findings(raw_text)
            st.session_state["findings"] = findings

        extracted_path = "data/extracted_audit.txt"
        save_extracted_text(raw_text, extracted_path)

        with st.spinner("embedding chunks..."):
            df = build_dataframe(extracted_path)
            set_dataframe(df)
            st.session_state["df"] = df
            st.session_state["data_loaded"] = True

        st.sidebar.success(f"done - {len(df)} chunks ready!")

        if "findings" in st.session_state:
            with st.expander("ETL - Extracted Findings"):
                st.text(st.session_state["findings"])

# pull severity numbers out of the report text
def parse_severity_counts(report_text):
    critical, moderate, minor = 0, 0, 0
    crit = re.search(r"Critical Violations:\s*(\d+)", report_text)
    mod = re.search(r"Moderate Violations:\s*(\d+)", report_text)
    min_ = re.search(r"Minor Violations:\s*(\d+)", report_text)
    if crit:
        critical = int(crit.group(1))
    if mod:
        moderate = int(mod.group(1))
    if min_:
        minor = int(min_.group(1))
    return critical, moderate, minor

if st.session_state.get("data_loaded"):
    st.markdown("---")
    st.subheader("Ask a Compliance Question")

    question = st.text_input("Enter your question:", placeholder="e.g. Which factories had overtime violations?")

    col1, col2 = st.columns(2)
    with col1:
        run_single = st.button("Quick Answer (Single Agent)")
    with col2:
        run_multi = st.button("Full Report (3-Agent Workflow)")

    if run_single and question:
        with st.spinner("searching compliance data..."):
            result = run_single_agent(question)
        st.markdown("### Quick Answer")
        st.caption("Complete list of factories with relevant findings, ordered by relevance score. For detailed violation analysis and severity breakdown, use Full Report.")
        st.markdown(result)

    if run_multi and question:
        with st.spinner("running 3-agent compliance workflow..."):
            results = run_workflow(question)
        st.markdown("### Compliance Report")
        st.caption("Deep analysis of the most relevant findings. Focuses on top matches for detailed severity categorization and recommendations.")

        with st.expander("Step 1 - Raw Findings (Intake Agent)"):
            st.text(results["intake"])

        with st.expander("Step 2 - Violation Analysis (Analyst Agent)"):
            st.text(results["analysis"])

        st.markdown("#### Final Report")
        st.text(results["report"])

        # show severity chart if we can parse the numbers
        critical, moderate, minor = parse_severity_counts(results["report"])
        if critical + moderate + minor > 0:
            st.markdown("#### Violation Severity Breakdown")
            fig, ax = plt.subplots(figsize=(6, 3))
            bars = ax.bar(
                ["Critical", "Moderate", "Minor"],
                [critical, moderate, minor],
                color=["#d32f2f", "#f57c00", "#388e3c"]
            )
            ax.set_ylabel("Number of Factories")
            ax.set_title("Factories by Violation Severity")
            for bar, val in zip(bars, [critical, moderate, minor]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(val), ha="center", va="bottom", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=False)
            c1, c2, c3 = st.columns(3)
            c1.metric("Critical", critical)
            c2.metric("Moderate", moderate)
            c3.metric("Minor", minor)

        st.download_button(
            label="Download Report",
            data=results["report"],
            file_name="compliance_report.txt",
            mime="text/plain"
        )

else:
    st.info("Please load compliance data from the sidebar to get started")
    st.markdown("""
    ### How to use this system
    1. Click **Load Default Compliance Data** in the sidebar
    2. Or upload your own audit file (txt or pdf)
    3. Type a compliance question
    4. **Quick Answer** gives a fast list of affected factories
    5. **Full Report** gives a detailed breakdown with severity chart and recommendations
    6. Download the report if needed
    """)