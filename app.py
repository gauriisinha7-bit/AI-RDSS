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

# ============================================
# SKILL SCORE
# ============================================

def calculate_skill_score(candidate_skills):

    required = job_data["required_skills"]

    matched = []

    missing = []

    for skill in required:

        if skill in candidate_skills:

            matched.append(skill)

        else:

            missing.append(skill)

    if len(required) == 0:

        score = 0

    else:

        score = (

            len(matched) /

            len(required)

        ) * 100

    return round(score,2), matched, missing


# ============================================
# EXPERIENCE SCORE
# ============================================

def calculate_experience_score(text):

    matches = re.findall(

        r'(\d+)\+?\s*(?:years|year|yrs)',

        text.lower()

    )

    if matches:

        years = max(

            [int(x) for x in matches]

        )

    else:

        years = 0

    required = job_data["minimum_experience"]

    if years >= required:

        return 100

    elif years == 0:

        return 0

    else:

        return round(

            (years/required)*100,

            2

        )


# ============================================
# EDUCATION SCORE
# ============================================

def calculate_education_score(text):

    text = text.lower()

    for edu in job_data["education"]:

        if edu.lower() in text:

            return 100

    if "bachelor" in text:

        return 80

    return 0


# ============================================
# CERTIFICATION SCORE
# ============================================

CERTIFICATIONS = [

    "aws",

    "azure",

    "oracle",

    "google",

    "coursera",

    "udemy",

    "ibm",

    "microsoft"

]


def calculate_certification_score(text):

    text = text.lower()

    score = 0

    for cert in CERTIFICATIONS:

        if cert in text:

            score += 15

    return min(score,100)


# ============================================
# PROJECT SCORE
# ============================================

PROJECT_KEYWORDS = [

    "project",

    "developed",

    "implemented",

    "github",

    "designed",

    "built"

]


def calculate_project_score(text):

    text = text.lower()

    score = 0

    for word in PROJECT_KEYWORDS:

        if word in text:

            score += 20

    return min(score,100)


# ============================================
# SOFT SKILL SCORE
# ============================================

SOFT_SKILLS = [

    "leadership",

    "communication",

    "teamwork",

    "problem solving",

    "critical thinking",

    "adaptability"

]


def calculate_softskill_score(text):

    text = text.lower()

    score = 0

    for word in SOFT_SKILLS:

        if word in text:

            score += 15

    return min(score,100)


# ============================================
# FINAL SCORE
# ============================================

def calculate_final_score(

        skill,

        experience,

        education,

        ai,

        certification,

        project,

        softskills

):

    final = (

        skill * 0.30 +

        experience * 0.15 +

        education * 0.10 +

        ai * 0.20 +

        certification * 0.10 +

        project * 0.10 +

        softskills * 0.05

    )

    return round(final,2)
