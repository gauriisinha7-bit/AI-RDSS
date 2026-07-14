import streamlit as st
import pandas as pd
import json
import fitz
import re

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="AI-RDSS",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI-RDSS Candidate Recommendation System")
st.caption("AI Powered Resume Screening & Candidate Recommendation")


# ============================================
# LOAD AI MODEL
# ============================================

@st.cache_resource
def load_model():
    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

model = load_model()


# ============================================
# LOAD KNOWLEDGE BASE
# ============================================

with open(
    "recruitment_knowledge_base.json",
    "r"
) as f:

    job_database = json.load(f)


job_role = st.sidebar.selectbox(

    "Select Job Role",

    list(job_database.keys())

)

job_data = job_database[job_role]

weights = job_data["weights"]


# ============================================
# JOB DESCRIPTION
# ============================================

jd_file = st.sidebar.file_uploader(

    "Upload Job Description",

    type=["pdf"]

)


def extract_pdf(file):

    text = ""

    pdf = fitz.open(

        stream=file.read(),

        filetype="pdf"

    )

    for page in pdf:

        text += page.get_text()

    return text


if jd_file:

    jd_text = extract_pdf(jd_file)

    st.sidebar.success(

        "Job Description Loaded"

    )

else:

    jd_text = ""


# ============================================
# MASTER SKILL DATABASE
# ============================================

MASTER_SKILLS = [

    "Python",

    "Java",

    "C++",

    "JavaScript",

    "SQL",

    "Git",

    "Linux",

    "AWS",

    "Docker",

    "REST API",

    "React",

    "Node.js",

    "Machine Learning",

    "Deep Learning",

    "TensorFlow",

    "PyTorch",

    "Power BI",

    "Tableau",

    "Pandas",

    "NumPy",

    "Data Structures",

    "Algorithms",

    "Flask",

    "Django",

    "MongoDB",

    "MySQL"

]


# ============================================
# RESUME EXTRACTION
# ============================================

def extract_resume_text(file):

    text = ""

    pdf = fitz.open(

        stream=file.read(),

        filetype="pdf"

    )

    for page in pdf:

        text += page.get_text()

    return text


# ============================================
# SKILL EXTRACTION
# ============================================

def extract_skills(text):

    text = text.lower()

    found = []

    for skill in MASTER_SKILLS:

        if skill.lower() in text:

            found.append(skill)

    return sorted(

        list(set(found))

    )


# ============================================
# AI SEMANTIC MATCHING
# ============================================

def calculate_ai_similarity(

    resume_text,

    jd_text

):

    if jd_text == "":

        return 0

    resume_embedding = model.encode(

        resume_text

    )

    jd_embedding = model.encode(

        jd_text

    )

    similarity = cosine_similarity(

        [resume_embedding],

        [jd_embedding]

    )[0][0]

    return round(

        similarity * 100,

        2

    )
