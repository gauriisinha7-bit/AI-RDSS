import streamlit as st


st.set_page_config(
    page_title="AI-RDSS",
    layout="wide"
)


st.title(
    "AI-RDSS Candidate Recommendation System"
)


st.write(
    "AI-based Resume Screening and Candidate Ranking System"
)


resume_files = st.file_uploader(
    "Upload Resume Files",
    type=["pdf"],
    accept_multiple_files=True
)


job_description = st.file_uploader(
    "Upload Job Description",
    type=["pdf"]
)


if st.button("Analyze Candidates"):

    if resume_files and job_description:

        st.success(
            "Files uploaded successfully"
        )

    else:

        st.warning(
            "Please upload resumes and job description"
        )
