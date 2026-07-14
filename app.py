import streamlit as st
import pandas as pd
import json
import fitz
import re

import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="AI-RDSS Candidate Recommendation System",
    page_icon="🤖",
    layout="wide"
)


st.title("🤖 AI-RDSS Candidate Recommendation System")
st.caption("AI Powered Resume Screening and Candidate Recommendation")


# ======================================================
# LOAD AI MODEL
# ======================================================

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()


# ======================================================
# LOAD KNOWLEDGE BASE
# ======================================================

with open("recruitment_knowledge_base.json","r") as f:
    job_database = json.load(f)


job_role = st.sidebar.selectbox(
    "Select Job Role",
    list(job_database.keys())
)

job_data = job_database[job_role]

weights = job_data["weights"]


# ======================================================
# JOB DESCRIPTION
# ======================================================

job_description_file = st.sidebar.file_uploader(
    "Upload Job Description",
    type=["pdf"]
)


def extract_pdf_text(file):

    text = ""

    pdf = fitz.open(
        stream=file.read(),
        filetype="pdf"
    )

    for page in pdf:
        text += page.get_text()

    return text


if job_description_file:

    jd_text = extract_pdf_text(job_description_file)

    st.sidebar.success("Job Description Loaded")

else:

    jd_text = ""


# ======================================================
# RESUME TEXT EXTRACTION
# ======================================================

def extract_resume_text(file):

    text = ""

    pdf = fitz.open(
        stream=file.read(),
        filetype="pdf"
    )

    for page in pdf:

        text += page.get_text()

    return text


# ======================================================
# SKILL EXTRACTION
# ======================================================

MASTER_SKILL_LIST = [

    "Python",
    "SQL",
    "Git",
    "Docker",
    "AWS",
    "Linux",
    "REST API",
    "Java",
    "C++",
    "JavaScript",
    "Machine Learning",
    "Deep Learning",
    "TensorFlow",
    "PyTorch",
    "Pandas",
    "NumPy",
    "Power BI",
    "Tableau",
    "Excel",
    "Data Structures",
    "Algorithms",
    "Flask",
    "Django",
    "React",
    "Node.js"

]


def extract_skills(text):

    found = []

    text = text.lower()

    for skill in MASTER_SKILL_LIST:

        if skill.lower() in text:

            found.append(skill)

    return list(set(found))


# ======================================================
# AI SEMANTIC SIMILARITY
# ======================================================

def calculate_ai_similarity(

        resume_text,

        jd_text

):

    if jd_text == "":

        return 0

    resume_embedding = model.encode(resume_text)

    jd_embedding = model.encode(jd_text)

    similarity = cosine_similarity(

        [resume_embedding],

        [jd_embedding]

    )[0][0]

    return round(similarity * 100,2)
    ###############################################################
# RESUME ANALYSIS ENGINE
###############################################################

def analyze_resume(
    resume_text,
    jd_text,
    knowledge_base,
    role_weights
):

    ###########################################################
    # Extract Skills
    ###########################################################

    resume_skills = extract_skills(
        resume_text,
        knowledge_base
    )

    jd_skills = extract_skills(
        jd_text,
        knowledge_base
    )

    matched_skills = sorted(
        list(
            set(resume_skills).intersection(
                set(jd_skills)
            )
        )
    )

    missing_skills = sorted(
        list(
            set(jd_skills) -
            set(resume_skills)
        )
    )

    ###########################################################
    # Skill Score
    ###########################################################

    if len(jd_skills) == 0:

        skill_score = 0

    else:

        skill_score = round(

            (
                len(matched_skills)
                /
                len(jd_skills)
            )
            * 100,

            2

        )

    ###########################################################
    # Experience Score
    ###########################################################

    resume_experience = extract_experience(
        resume_text
    )

    jd_experience = extract_experience(
        jd_text
    )

    if jd_experience == 0:

        experience_score = 100

    else:

        experience_score = min(

            round(

                (
                    resume_experience
                    /
                    jd_experience
                )
                * 100,

                2

            ),

            100

        )

    ###########################################################
    # Education Score
    ###########################################################

    education_score = compare_education(

        resume_text,

        jd_text

    )

    ###########################################################
    # Project Score
    ###########################################################

    project_keywords = [

        "project",
        "developed",
        "implemented",
        "built",
        "designed",
        "system",
        "application",
        "dashboard",
        "web application",
        "machine learning",
        "deep learning",
        "ai",
        "nlp",
        "research"

    ]

    resume_lower = resume_text.lower()

    project_count = 0

    for keyword in project_keywords:

        if keyword in resume_lower:

            project_count += 1

    project_score = min(

        project_count * 10,

        100

    )

    ###########################################################
    # Certification Score
    ###########################################################

    certification_keywords = [

        "certification",
        "certificate",
        "coursera",
        "udemy",
        "nptel",
        "edx",
        "aws",
        "azure",
        "google",
        "oracle",
        "ibm",
        "microsoft"

    ]

    certification_count = 0

    for keyword in certification_keywords:

        if keyword in resume_lower:

            certification_count += 1

    certification_score = min(

        certification_count * 15,

        100

    )

    ###########################################################
    # Soft Skills Score
    ###########################################################

    soft_skill_keywords = [

        "communication",
        "leadership",
        "teamwork",
        "problem solving",
        "critical thinking",
        "adaptability",
        "decision making",
        "time management",
        "collaboration",
        "creativity",
        "presentation",
        "analytical"

    ]

    soft_skill_count = 0

    for keyword in soft_skill_keywords:

        if keyword in resume_lower:

            soft_skill_count += 1

    soft_skill_score = min(

        soft_skill_count * 8,

        100

    )

    ###########################################################
    # Semantic Similarity
    ###########################################################

    semantic_score = calculate_semantic_similarity(

        resume_text,

        jd_text

    )

    ###########################################################
    # Final AI Weighted Score
    ###########################################################

    final_score = (

        semantic_score *
        role_weights["semantic"]

        +

        skill_score *
        role_weights["skills"]

        +

        experience_score *
        role_weights["experience"]

        +

        education_score *
        role_weights["education"]

        +

        project_score *
        role_weights["projects"]

        +

        certification_score *
        role_weights["certifications"]

        +

        soft_skill_score *
        role_weights["soft_skills"]

    )

    final_score = round(

        final_score,

        2

    )

    ###########################################################
    # Return Results
    ###########################################################

    return {

        "Semantic Score": semantic_score,

        "Skill Score": skill_score,

        "Experience Score": experience_score,

        "Education Score": education_score,

        "Project Score": project_score,

        "Certification Score": certification_score,

        "Soft Skill Score": soft_skill_score,

        "Matched Skills": matched_skills,

        "Missing Skills": missing_skills,

        "Resume Experience": resume_experience,

        "Required Experience": jd_experience,

        "Final Score": final_score

    }

###############################################################
# DETAILED RESUME ANALYSIS ENGINE
###############################################################

def generate_resume_analysis(
    resume_text,
    jd_text,
    knowledge_base,
    role_weights
):

    results = analyze_resume(

        resume_text,
        jd_text,
        knowledge_base,
        role_weights

    )

    ###########################################################
    # Candidate Strengths
    ###########################################################

    strengths = []

    if results["Skill Score"] >= 80:
        strengths.append(
            "Excellent technical skill alignment."
        )

    if results["Experience Score"] >= 80:
        strengths.append(
            "Experience meets or exceeds job requirement."
        )

    if results["Education Score"] >= 80:
        strengths.append(
            "Educational qualification is highly suitable."
        )

    if results["Project Score"] >= 70:
        strengths.append(
            "Relevant industry/research projects identified."
        )

    if results["Certification Score"] >= 70:
        strengths.append(
            "Relevant professional certifications available."
        )

    if results["Soft Skill Score"] >= 70:
        strengths.append(
            "Strong evidence of communication and teamwork."
        )

    ###########################################################
    # Weaknesses
    ###########################################################

    weaknesses = []

    if results["Skill Score"] < 60:

        weaknesses.append(
            "Several required skills are missing."
        )

    if results["Experience Score"] < 60:

        weaknesses.append(
            "Experience is below the required level."
        )

    if results["Education Score"] < 60:

        weaknesses.append(
            "Education does not fully satisfy requirements."
        )

    if results["Project Score"] < 50:

        weaknesses.append(
            "Few relevant projects detected."
        )

    if results["Certification Score"] < 50:

        weaknesses.append(
            "Professional certifications are limited."
        )

    if results["Soft Skill Score"] < 50:

        weaknesses.append(
            "Limited evidence of soft skills."
        )

    ###########################################################
    # Recommendation
    ###########################################################

    final_score = results["Final Score"]

    if final_score >= 90:

        recommendation = "Highly Recommended"

    elif final_score >= 80:

        recommendation = "Recommended"

    elif final_score >= 70:

        recommendation = "Recommended with Reservations"

    elif final_score >= 60:

        recommendation = "Average Candidate"

    else:

        recommendation = "Not Recommended"

    ###########################################################
    # Hiring Confidence
    ###########################################################

    if final_score >= 90:

        confidence = "Very High"

    elif final_score >= 80:

        confidence = "High"

    elif final_score >= 70:

        confidence = "Moderate"

    elif final_score >= 60:

        confidence = "Low"

    else:

        confidence = "Very Low"

    ###########################################################
    # Resume Summary
    ###########################################################

    summary = f"""
Candidate achieved an overall AI recruitment score of
{final_score}%.

The resume demonstrates
{len(results["Matched Skills"])} matched skills,
{results["Resume Experience"]} years of experience
and an education score of
{results["Education Score"]}%.
"""

    ###########################################################
    # Return
    ###########################################################

    analysis = {

        "Summary": summary,

        "Strengths": strengths,

        "Weaknesses": weaknesses,

        "Recommendation": recommendation,

        "Hiring Confidence": confidence,

        "Matched Skills": results["Matched Skills"],

        "Missing Skills": results["Missing Skills"],

        "Detailed Scores": results

    }

    return analysis
    ###############################################################
# ATS RECRUITER DASHBOARD
###############################################################

st.title("🤖 Explainable AI Recruitment Decision Support System")

st.markdown(
"""
Upload a candidate resume and job description.
The system automatically evaluates the candidate using
semantic AI matching and explainable scoring.
"""
)

st.divider()

###############################################################
# FILE UPLOADS
###############################################################

resume_file = st.file_uploader(
    "📄 Upload Resume (PDF)",
    type=["pdf"],
    key="resume"
)

jd_file = st.file_uploader(
    "📄 Upload Job Description (PDF)",
    type=["pdf"],
    key="jd"
)

###############################################################
# ANALYZE BUTTON
###############################################################

analyze_button = st.button(
    "🚀 Analyze Candidate",
    use_container_width=True
)

###############################################################
# START ANALYSIS
###############################################################

if analyze_button:

    if resume_file is None:

        st.error("Please upload a Resume.")

    elif jd_file is None:

        st.error("Please upload a Job Description.")

    else:

        with st.spinner("Analyzing Resume..."):

            ###################################################
            # Extract PDF Text
            ###################################################

            resume_text = extract_text_from_pdf(
                resume_file
            )

            jd_text = extract_text_from_pdf(
                jd_file
            )

            ###################################################
            # AI Analysis
            ###################################################

            analysis = generate_resume_analysis(

                resume_text,

                jd_text,

                knowledge_base,

                role_weights

            )

            scores = analysis["Detailed Scores"]

            st.success("Analysis Completed Successfully")

            st.divider()

            ###################################################
            # FINAL SCORE
            ###################################################

            st.metric(

                label="Final AI Recruitment Score",

                value=f'{scores["Final Score"]}%'

            )

            ###################################################
            # SCORE CARDS
            ###################################################

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Skill Score",
                    f'{scores["Skill Score"]}%'
                )

                st.metric(
                    "Experience Score",
                    f'{scores["Experience Score"]}%'
                )

            with col2:

                st.metric(
                    "Education Score",
                    f'{scores["Education Score"]}%'
                )

                st.metric(
                    "Project Score",
                    f'{scores["Project Score"]}%'
                )

            with col3:

                st.metric(
                    "Certification Score",
                    f'{scores["Certification Score"]}%'
                )

                st.metric(
                    "Soft Skill Score",
                    f'{scores["Soft Skill Score"]}%'
                )

            st.divider()

            ###################################################
            # RECOMMENDATION
            ###################################################

            st.subheader("Recruitment Recommendation")

            recommendation = analysis["Recommendation"]

            if recommendation == "Highly Recommended":

                st.success(recommendation)

            elif recommendation == "Recommended":

                st.success(recommendation)

            elif recommendation == "Recommended with Reservations":

                st.warning(recommendation)

            elif recommendation == "Average Candidate":

                st.warning(recommendation)

            else:

                st.error(recommendation)

            ###################################################
            # HIRING CONFIDENCE
            ###################################################

            st.info(
                f'Hiring Confidence : {analysis["Hiring Confidence"]}'
            )

            st.divider()

            ###################################################
            # SUMMARY
            ###################################################

            st.subheader("Candidate Summary")

            st.write(
                analysis["Summary"]
            )

            ###################################################
            # STRENGTHS
            ###################################################

            st.subheader("Candidate Strengths")

            for strength in analysis["Strengths"]:

                st.success(strength)

            ###################################################
            # WEAKNESSES
            ###################################################

            st.subheader("Candidate Weaknesses")

            for weakness in analysis["Weaknesses"]:

                st.warning(weakness)

            ###################################################
            # MATCHED SKILLS
            ###################################################

            st.subheader("Matched Skills")

            st.write(
                analysis["Matched Skills"]
            )

            ###################################################
            # MISSING SKILLS
            ###################################################

            st.subheader("Missing Skills")

            st.write(
                analysis["Missing Skills"]
            )
            ###############################################################
# IMPORTS
###############################################################

import plotly.graph_objects as go
import pandas as pd

###############################################################
# DASHBOARD VISUALIZATION
###############################################################

def display_dashboard(scores):

    ###########################################################
    # Circular Gauge
    ###########################################################

    fig = go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=scores["Final Score"],

            title={

                "text":"Final AI Score"

            },

            gauge={

                "axis":{"range":[0,100]},

                "bar":{"color":"green"},

                "steps":[

                    {
                        "range":[0,50],
                        "color":"#ffcccc"
                    },

                    {
                        "range":[50,70],
                        "color":"#ffe699"
                    },

                    {
                        "range":[70,85],
                        "color":"#b6d7a8"
                    },

                    {
                        "range":[85,100],
                        "color":"#6aa84f"
                    }

                ]

            }

        )

    )

    fig.update_layout(

        height=350

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    ###########################################################
    # Radar Chart
    ###########################################################

    radar_labels = [

        "Skill",

        "Experience",

        "Education",

        "Projects",

        "Certification",

        "Soft Skills"

    ]

    radar_values = [

        scores["Skill Score"],

        scores["Experience Score"],

        scores["Education Score"],

        scores["Project Score"],

        scores["Certification Score"],

        scores["Soft Skill Score"]

    ]

    radar = go.Figure()

    radar.add_trace(

        go.Scatterpolar(

            r=radar_values,

            theta=radar_labels,

            fill="toself",

            name="Candidate"

        )

    )

    radar.update_layout(

        polar={

            "radialaxis":{

                "visible":True,

                "range":[0,100]

            }

        },

        height=500

    )

    st.plotly_chart(

        radar,

        use_container_width=True

    )

    ###########################################################
    # Score Table
    ###########################################################

    score_table = pd.DataFrame({

        "Criterion":[

            "Skill",

            "Experience",

            "Education",

            "Projects",

            "Certification",

            "Soft Skills",

            "Semantic"

        ],

        "Score":[

            scores["Skill Score"],

            scores["Experience Score"],

            scores["Education Score"],

            scores["Project Score"],

            scores["Certification Score"],

            scores["Soft Skill Score"],

            scores["Semantic Score"]

        ]

    })

    st.dataframe(

        score_table,

        use_container_width=True

    )

    ###########################################################
    # Horizontal Bar Chart
    ###########################################################

    bar = go.Figure()

    bar.add_trace(

        go.Bar(

            x=score_table["Score"],

            y=score_table["Criterion"],

            orientation="h",

            text=score_table["Score"],

            textposition="outside"

        )

    )

    bar.update_layout(

        xaxis=dict(

            range=[0,100]

        ),

        height=500

    )

    st.plotly_chart(

        bar,

        use_container_width=True

    )
    ###############################################################
# MULTIPLE RESUME UPLOAD
###############################################################

resume_files = st.file_uploader(

    "Upload Candidate Resumes",

    type=["pdf"],

    accept_multiple_files=True

)

jd_file = st.file_uploader(

    "Upload Job Description",

    type=["pdf"]

)
###############################################################
# FEATURE CONTRIBUTION
###############################################################

def calculate_feature_contributions(scores, role_weights):

    contributions = {

        "Semantic Matching": round(
            scores["Semantic Score"] *
            role_weights["semantic"], 2
        ),

        "Skill Matching": round(
            scores["Skill Score"] *
            role_weights["skills"], 2
        ),

        "Experience": round(
            scores["Experience Score"] *
            role_weights["experience"], 2
        ),

        "Education": round(
            scores["Education Score"] *
            role_weights["education"], 2
        ),

        "Projects": round(
            scores["Project Score"] *
            role_weights["projects"], 2
        ),

        "Certifications": round(
            scores["Certification Score"] *
            role_weights["certifications"], 2
        ),

        "Soft Skills": round(
            scores["Soft Skill Score"] *
            role_weights["soft_skills"], 2
        )

    }

    return contributions
    ###############################################################
# FEATURE CONTRIBUTION
###############################################################

def calculate_feature_contributions(scores, role_weights):

    contributions = {

        "Semantic Matching": round(
            scores["Semantic Score"] *
            role_weights["semantic"], 2
        ),

        "Skill Matching": round(
            scores["Skill Score"] *
            role_weights["skills"], 2
        ),

        "Experience": round(
            scores["Experience Score"] *
            role_weights["experience"], 2
        ),

        "Education": round(
            scores["Education Score"] *
            role_weights["education"], 2
        ),

        "Projects": round(
            scores["Project Score"] *
            role_weights["projects"], 2
        ),

        "Certifications": round(
            scores["Certification Score"] *
            role_weights["certifications"], 2
        ),

        "Soft Skills": round(
            scores["Soft Skill Score"] *
            role_weights["soft_skills"], 2
        )

    }

    return contributions
    ###############################################################
# DISPLAY XAI TABLE
###############################################################

st.subheader("Explainable AI Report")

xai = generate_explanation(

    scores,

    role_weights

)

xai_df = pd.DataFrame(

    xai

)

st.dataframe(

    xai_df,

    use_container_width=True
)
###############################################################
# FEATURE CONTRIBUTION CHART
###############################################################

fig = go.Figure()

fig.add_trace(

    go.Bar(

        x=xai_df["Contribution"],

        y=xai_df["Feature"],

        orientation="h",

        text=xai_df["Contribution"],

        textposition="outside"

    )

)

fig.update_layout(

    title="Feature Contribution to Final AI Score",

    xaxis_title="Contribution",

    yaxis_title="Feature",

    height=500

)

st.plotly_chart(

    fig,

    use_container_width=True
)
###############################################################
# NATURAL LANGUAGE EXPLANATION
###############################################################

st.subheader("AI Decision Explanation")

top_feature = max(
    xai,
    key=lambda x: x["Contribution"]
)

lowest_feature = min(
    xai,
    key=lambda x: x["Contribution"]
)

st.info(f"""
The candidate achieved a Final AI Score of **{scores['Final Score']}%**.

The strongest factor influencing the decision was **{top_feature['Feature']}**, contributing **{top_feature['Contribution']}** points.

The weakest area was **{lowest_feature['Feature']}**, contributing only **{lowest_feature['Contribution']}** points.

Improving the weakest area would likely increase the candidate's overall suitability for this role.
""")
