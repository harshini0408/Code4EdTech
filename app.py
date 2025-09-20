import streamlit as st
import pandas as pd
from datetime import datetime

# --- Final Imports ---
from src.resume_parser import extract_text_from_pdf, extract_text_from_docx, parse_resume
from src.jd_parser import parse_job_description
from src.scoring_engine import calculate_advanced_relevance
from src.cluster_analyzer import analyze_clusters

def main():
    st.set_page_config(page_title="AI Resume Analyzer", layout="wide", initial_sidebar_state="expanded")
    st.markdown("<h1 style='text-align: center; color: #4B0082;'>🚀 AI-Powered Resume Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>An advanced tool integrating keyword, semantic, and LLM-based analysis for unparalleled accuracy.</p>", unsafe_allow_html=True)
    
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'cluster_names' not in st.session_state:
        st.session_state.cluster_names = {}

    col1, col2 = st.columns([2, 3])

    with col1:
        st.header("📋 Upload Documents")
        jd_file = st.file_uploader("1. Upload the Job Description", type=['pdf', 'docx', 'txt'])
        resume_files = st.file_uploader("2. Upload Candidate Resumes", type=['pdf', 'docx'], accept_multiple_files=True)

        if st.button("✨ Analyze Now", type="primary") and jd_file and resume_files:
            run_analysis(jd_file, resume_files)

    with col2:
        st.header("📊 Analysis Dashboard")
        if not st.session_state.results:
            st.info("Upload documents and click 'Analyze Now' to see the results here.")
        else:
            display_dashboard()

def run_analysis(jd_file, resume_files):
    with st.spinner('Performing advanced AI analysis... Please wait.'):
        jd_data = parse_job_description(process_uploaded_file(jd_file))

        parsed_resumes = []
        for resume_file in resume_files:
            text = process_uploaded_file(resume_file)
            if text:
                resume_data = parse_resume(text)
                resume_data['filename'] = resume_file.name
                parsed_resumes.append(resume_data)

        clustered_resumes, cluster_names = analyze_clusters(parsed_resumes)
        st.session_state.cluster_names = cluster_names

        final_results = []
        for resume_data in clustered_resumes:
            analysis_result = calculate_advanced_relevance(resume_data, jd_data)
            result_entry = {
                'Filename': resume_data['filename'],
                'Score': analysis_result['overall_score'],
                'Verdict': analysis_result['verdict'],
                'Missing Elements': ", ".join(analysis_result.get('missing_elements', [])),
                'Suggestions': analysis_result.get('suggestions', ''),
                'Cluster': resume_data.get('cluster_name', 'N/A')
            }
            final_results.append(result_entry)

        st.session_state.results = final_results
    st.success("✅ Analysis and Clustering complete!")
    st.rerun()

def display_dashboard():
    df = pd.DataFrame(st.session_state.results)
    if df.empty: return

    st.subheader("Results Overview")
    filtered_df = df
    if 'cluster_names' in st.session_state and st.session_state.cluster_names:
        cluster_options = list(st.session_state.cluster_names.values())
        selected_clusters = st.multiselect("Filter by Profile:", options=cluster_options, default=cluster_options)
        if selected_clusters:
            filtered_df = df[df['Cluster'].isin(selected_clusters)]

    df_sorted = filtered_df.sort_values(by="Score", ascending=False).reset_index(drop=True)
    st.dataframe(df_sorted[['Filename', 'Score', 'Verdict', 'Cluster']], use_container_width=True)
    
    st.subheader("Detailed Candidate Breakdown")
    for index, row in df_sorted.iterrows():
        with st.expander(f"**{index + 1}. {row['Filename']}** - Score: {row['Score']}% ({row['Verdict']}) | Profile: **{row['Cluster']}**"):
            st.markdown(f"**Verdict:** {row['Verdict']}")
            st.markdown("**Critical Missing Elements:**")
            st.info(row['Missing Elements'] or "No critical elements were found missing. Great fit!")
            st.markdown("**Personalized Improvement Suggestions:**")
            st.success(row['Suggestions'] or "No specific suggestions were generated.")

def process_uploaded_file(file):
    try:
        file.seek(0)
        file_type = file.type
        if file_type == "application/pdf":
            return extract_text_from_pdf(file)
        elif "word" in file_type:
            return extract_text_from_docx(file)
        return str(file.read(), "utf-8")
    except Exception as e:
        st.error(f"Error processing {file.name}: {e}")
        return ""

if __name__ == "__main__":
    main()
