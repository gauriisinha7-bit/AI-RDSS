##############################################################
# AI RECRUITMENT DECISION SUPPORT SYSTEM
##############################################################

import streamlit as st
import pandas as pd
import numpy as np
import json
import re

from pathlib import Path

from PyPDF2 import PdfReader

from sentence_transformers import SentenceTransformer

from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt

##############################################################
# PAGE CONFIGURATION
##############################################################

st.set_page_config(

    page_title="AI Recruitment Decision Support System",

    page_icon="🤖",

    layout="wide"

)

##############################################################
# TITLE
##############################################################

st.title("🤖 Explainable AI Recruitment Decision Support System")

st.markdown("""

Upload resumes and a Job Description.

The system evaluates candidates using

- Semantic AI Matching

- Role-Based Evaluation

- Explainable AI

- Automatic Candidate Ranking

""")

st.divider()

##############################################################
# LOAD SENTENCE TRANSFORMER
##############################################################

@st.cache_resource
def load_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

model = load_model()

##############################################################
# LOAD KNOWLEDGE BASE
##############################################################

@st.cache_data
def load_knowledge_base():

    path = Path(
        "recruitment_knowledge_base.json"
    )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)

knowledge_base = load_knowledge_base()

##############################################################
# ROLE WEIGHTS
##############################################################

ROLE_WEIGHTS = {

    "Software Engineer":{

        "semantic":0.20,

        "skills":0.30,

        "experience":0.20,

        "education":0.10,

        "projects":0.10,

        "certifications":0.05,

        "soft_skills":0.05

    },

    "Data Scientist":{

        "semantic":0.25,

        "skills":0.25,

        "experience":0.15,

        "education":0.10,

        "projects":0.15,

        "certifications":0.05,

        "soft_skills":0.05

    },

    "HR Manager":{

        "semantic":0.15,

        "skills":0.20,

        "experience":0.20,

        "education":0.15,

        "projects":0.05,

        "certifications":0.05,

        "soft_skills":0.20

    }

}

##############################################################
# ROLE SELECTION
##############################################################

selected_role = st.selectbox(

    "Select Job Role",

    list(ROLE_WEIGHTS.keys())

)

role_weights = ROLE_WEIGHTS[
    selected_role
]

st.divider()

##############################################################
# FILE UPLOAD
##############################################################

resume_files = st.file_uploader(

    "Upload Candidate Resumes",

    type=["pdf"],

    accept_multiple_files=True

)

jd_file = st.file_uploader(

    "Upload Job Description",

    type=["pdf"]

)

analyze_button = st.button(

    "Analyze Candidates",

    use_container_width=True

)

st.divider()

##############################################################
# PDF TEXT EXTRACTION
##############################################################

def extract_text_from_pdf(uploaded_file):
    """
    Extract text from a PDF uploaded through Streamlit.
    Returns an empty string if extraction fails.
    """

    if uploaded_file is None:
        return ""

    text = ""

    try:

        reader = PdfReader(uploaded_file)

        if reader.is_encrypted:

            try:
                reader.decrypt("")
            except Exception:
                return ""

        for page in reader.pages:

            try:

                page_text = page.extract_text()

                if page_text:

                    text += page_text + "\n"

            except Exception:

                continue

    except Exception as e:

        st.error(f"Error reading PDF: {e}")

        return ""

    return text.strip()


##############################################################
# TEXT CLEANING
##############################################################

def clean_text(text):

    if not text:
        return ""

    text = text.lower()

    text = re.sub(r"\n", " ", text)

    text = re.sub(r"\t", " ", text)

    text = re.sub(r"\s+", " ", text)

    text = re.sub(r"[^a-z0-9+#./ ]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


##############################################################
# SAFE PDF LOADER
##############################################################

def load_resume_text(uploaded_file):

    raw_text = extract_text_from_pdf(uploaded_file)

    return clean_text(raw_text)


def load_jd_text(uploaded_file):

    raw_text = extract_text_from_pdf(uploaded_file)

    return clean_text(raw_text)

##############################################################
# PDF TEXT EXTRACTION
##############################################################

def extract_text_from_pdf(uploaded_file):
    """
    Extract text from a PDF uploaded through Streamlit.
    Returns an empty string if extraction fails.
    """

    if uploaded_file is None:
        return ""

    text = ""

    try:

        reader = PdfReader(uploaded_file)

        if reader.is_encrypted:

            try:
                reader.decrypt("")
            except Exception:
                return ""

        for page in reader.pages:

            try:

                page_text = page.extract_text()

                if page_text:

                    text += page_text + "\n"

            except Exception:

                continue

    except Exception as e:

        st.error(f"Error reading PDF: {e}")

        return ""

    return text.strip()


##############################################################
# TEXT CLEANING
##############################################################

def clean_text(text):

    if not text:
        return ""

    text = text.lower()

    text = re.sub(r"\n", " ", text)

    text = re.sub(r"\t", " ", text)

    text = re.sub(r"\s+", " ", text)

    text = re.sub(r"[^a-z0-9+#./ ]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()

##############################################################
# SAFE PDF LOADER
##############################################################

def load_resume_text(uploaded_file):

    raw_text = extract_text_from_pdf(uploaded_file)

    return clean_text(raw_text)


def load_jd_text(uploaded_file):

    raw_text = extract_text_from_pdf(uploaded_file)

    return clean_text(raw_text)

##############################################################
# SKILL EXTRACTION ENGINE
##############################################################

def get_all_skills(knowledge_base):
    """
    Collect all skills from the knowledge base.
    Supports both:
    {
        "skills": [...]
    }
    and nested categories:
    {
        "technical_skills": [...],
        "soft_skills": [...]
    }
    """

    all_skills = set()

    if not isinstance(knowledge_base, dict):
        return []

    for key, value in knowledge_base.items():

        if isinstance(value, list):

            for item in value:

                if isinstance(item, str):

                    all_skills.add(item.lower())

        elif isinstance(value, dict):

            for _, sublist in value.items():

                if isinstance(sublist, list):

                    for item in sublist:

                        if isinstance(item, str):

                            all_skills.add(item.lower())

    return sorted(all_skills)


##############################################################
# EXTRACT SKILLS
##############################################################

def extract_skills(text, knowledge_base):

    text = clean_text(text)

    skills_found = set()

    all_skills = get_all_skills(knowledge_base)

    for skill in all_skills:

        pattern = r"\b" + re.escape(skill) + r"\b"

        if re.search(pattern, text):

            skills_found.add(skill)

    return sorted(skills_found)


##############################################################
# MATCH SKILLS
##############################################################

def compare_skills(resume_text, jd_text, knowledge_base):

    resume_skills = extract_skills(

        resume_text,

        knowledge_base

    )

    jd_skills = extract_skills(

        jd_text,

        knowledge_base

    )

    matched = sorted(

        list(

            set(resume_skills)

            &

            set(jd_skills)

        )

    )

    missing = sorted(

        list(

            set(jd_skills)

            -

            set(resume_skills)

        )

    )

    extra = sorted(

        list(

            set(resume_skills)

            -

            set(jd_skills)

        )

    )

    if len(jd_skills) == 0:

        score = 0.0

    else:

        score = round(

            (len(matched) / len(jd_skills)) * 100,

            2

        )

    return {

        "resume_skills": resume_skills,

        "jd_skills": jd_skills,

        "matched_skills": matched,

        "missing_skills": missing,

        "extra_skills": extra,

        "skill_score": score

    }

##############################################################
# EXPERIENCE EXTRACTION ENGINE
##############################################################

def extract_experience_years(text):
    """
    Extract the maximum years of experience mentioned
    in a resume or job description.
    """

    if not text:
        return 0

    text = clean_text(text)

    patterns = [

        r'(\d+)\+?\s*years?',
        r'(\d+)\+?\s*yrs?',
        r'(\d+)\+?\s*yr',
        r'experience\s*[:\-]?\s*(\d+)',
        r'(\d+)\s*years?\s*experience',
        r'minimum\s*(\d+)\s*years?',
        r'at least\s*(\d+)\s*years?'

    ]

    values = []

    for pattern in patterns:

        matches = re.findall(pattern, text)

        for match in matches:

            try:

                values.append(int(match))

            except ValueError:

                continue

    if len(values) == 0:

        return 0

    return max(values)


##############################################################
# EXPERIENCE COMPARISON
##############################################################

def compare_experience(
    resume_text,
    jd_text
):

    candidate_exp = extract_experience_years(
        resume_text
    )

    required_exp = extract_experience_years(
        jd_text
    )

    if required_exp == 0:

        score = 100.0

    else:

        score = min(

            round(

                (candidate_exp / required_exp) * 100,

                2

            ),

            100

        )

    if candidate_exp > required_exp:

        status = "Exceeds Requirement"

    elif candidate_exp == required_exp:

        status = "Matches Requirement"

    elif candidate_exp == 0:

        status = "Experience Not Found"

    else:

        status = "Below Requirement"

    return {

        "candidate_experience": candidate_exp,

        "required_experience": required_exp,

        "experience_score": score,

        "experience_status": status

    }

##############################################################
# EDUCATION MATCHING ENGINE
##############################################################

EDUCATION_LEVELS = {

    "high school": 1,

    "intermediate": 2,

    "12th": 2,

    "diploma": 3,

    "iti": 3,

    "b.a": 4,
    "ba": 4,

    "b.sc": 4,
    "bsc": 4,

    "b.com": 4,
    "bcom": 4,

    "bca": 4,

    "b.e": 4,
    "be": 4,

    "b.tech": 4,
    "btech": 4,

    "llb": 4,

    "m.a": 5,
    "ma": 5,

    "m.sc": 5,
    "msc": 5,

    "m.com": 5,
    "mcom": 5,

    "mba": 5,

    "mca": 5,

    "m.tech": 5,
    "mtech": 5,

    "llm": 5,

    "phd": 6,

    "doctorate": 6
}


##############################################################
# EXTRACT EDUCATION
##############################################################

def extract_education_level(text):

    if not text:

        return 0, "Not Found"

    text = clean_text(text)

    highest_level = 0

    highest_degree = "Not Found"

    for degree, level in EDUCATION_LEVELS.items():

        pattern = r"\b" + re.escape(degree) + r"\b"

        if re.search(pattern, text):

            if level > highest_level:

                highest_level = level

                highest_degree = degree

    return highest_level, highest_degree


##############################################################
# EDUCATION COMPARISON
##############################################################

def compare_education(

    resume_text,

    jd_text

):

    resume_level, resume_degree = extract_education_level(

        resume_text

    )

    jd_level, jd_degree = extract_education_level(

        jd_text

    )

    ##########################################################

    if jd_level == 0:

        score = 100.0

        status = "No Education Requirement"

    elif resume_level >= jd_level:

        score = 100.0

        status = "Requirement Satisfied"

    else:

        score = round(

            (resume_level / jd_level) * 100,

            2

        )

        status = "Below Requirement"

    ##########################################################

    return {

        "candidate_degree": resume_degree,

        "required_degree": jd_degree,

        "education_score": score,

        "education_status": status

    }

##############################################################
# PROJECT DETECTION
##############################################################

PROJECT_KEYWORDS = [

    "project",

    "developed",

    "implemented",

    "built",

    "designed",

    "created",

    "system",

    "application",

    "dashboard",

    "website",

    "api",

    "machine learning",

    "deep learning",

    "artificial intelligence",

    "research",

    "nlp",

    "computer vision"

]


def calculate_project_score(resume_text):

    text = clean_text(resume_text)

    count = 0

    found_projects = []

    for keyword in PROJECT_KEYWORDS:

        if keyword.lower() in text:

            count += 1

            found_projects.append(keyword)

    score = min(

        count * 10,

        100

    )

    return {

        "project_score": score,

        "projects_found": sorted(

            list(set(found_projects))

        )

    }


##############################################################
# CERTIFICATION DETECTION
##############################################################

CERTIFICATION_KEYWORDS = [

    "certification",

    "certificate",

    "coursera",

    "udemy",

    "nptel",

    "aws",

    "azure",

    "oracle",

    "google",

    "ibm",

    "microsoft",

    "salesforce",

    "cisco"

]


def calculate_certification_score(resume_text):

    text = clean_text(resume_text)

    found = []

    count = 0

    for cert in CERTIFICATION_KEYWORDS:

        if cert.lower() in text:

            count += 1

            found.append(cert)

    score = min(

        count * 15,

        100

    )

    return {

        "certification_score": score,

        "certifications_found": sorted(

            list(set(found))

        )

    }


##############################################################
# SOFT SKILLS
##############################################################

SOFT_SKILLS = [

    "communication",

    "leadership",

    "teamwork",

    "problem solving",

    "critical thinking",

    "adaptability",

    "decision making",

    "time management",

    "analytical",

    "creativity",

    "collaboration",

    "presentation",

    "negotiation"

]


def calculate_soft_skill_score(resume_text):

    text = clean_text(resume_text)

    found = []

    count = 0

    for skill in SOFT_SKILLS:

        if skill.lower() in text:

            count += 1

            found.append(skill)

    score = min(

        count * 8,

        100

    )

    return {

        "soft_skill_score": score,

        "soft_skills_found": sorted(

            list(set(found))

        )

    }

##############################################################
# KNOWLEDGE BASE HELPERS
##############################################################

def get_role_data(role_name, knowledge_base):
    """
    Return configuration for the selected role.
    """

    return knowledge_base.get(role_name, {})


def get_required_skills(role_name, knowledge_base):

    role = get_role_data(role_name, knowledge_base)

    return role.get("required_skills", [])


def get_preferred_skills(role_name, knowledge_base):

    role = get_role_data(role_name, knowledge_base)

    return role.get("preferred_skills", [])


def get_required_education(role_name, knowledge_base):

    role = get_role_data(role_name, knowledge_base)

    return role.get("education", [])


def get_required_experience(role_name, knowledge_base):

    role = get_role_data(role_name, knowledge_base)

    return role.get("minimum_experience", 0)


def get_role_weights(role_name, knowledge_base):

    role = get_role_data(role_name, knowledge_base)

    return role.get("weights", {})

##############################################################
# EXPERIENCE SCORE
##############################################################

def calculate_experience_score(

    resume_text,

    role_name,

    knowledge_base

):

    candidate = extract_experience_years(

        resume_text

    )

    required = get_required_experience(

        role_name,

        knowledge_base

    )

    if required == 0:

        score = 100

    else:

        score = min(

            round(

                (candidate / required) * 100,

                2

            ),

            100

        )

    return {

        "candidate_experience": candidate,

        "required_experience": required,

        "experience_score": score

    }

##############################################################
# EDUCATION SCORE
##############################################################

def calculate_education_score(

    resume_text,

    role_name,

    knowledge_base

):

    candidate_level,

    candidate_degree = extract_education_level(

        resume_text

    )

    required = get_required_education(

        role_name,

        knowledge_base

    )

    if len(required) == 0:

        return {

            "education_score":100,

            "candidate_degree":candidate_degree

        }

    required_levels = []

    for degree in required:

        level = EDUCATION_LEVELS.get(

            degree.lower(),

            0

        )

        required_levels.append(level)

    required_level = max(required_levels)

    if candidate_level >= required_level:

        score = 100

    else:

        score = round(

            (candidate_level / required_level) * 100,

            2

        )

    return {

        "education_score":score,

        "candidate_degree":candidate_degree,

        "required_degree":required

    }

##############################################################
# FINAL AI RECRUITMENT SCORING ENGINE
##############################################################

def analyze_candidate(
    resume_text,
    role_name,
    knowledge_base
):

    ##########################################################
    # Load Role Configuration
    ##########################################################

    role = get_role_data(
        role_name,
        knowledge_base
    )

    weights = role.get("weights", {})

    ##########################################################
    # Individual Scores
    ##########################################################

    skill_result = calculate_skill_score(
        resume_text,
        role_name,
        knowledge_base
    )

    experience_result = calculate_experience_score(
        resume_text,
        role_name,
        knowledge_base
    )

    education_result = calculate_education_score(
        resume_text,
        role_name,
        knowledge_base
    )

    project_result = calculate_project_score(
        resume_text
    )

    certification_result = calculate_certification_score(
        resume_text
    )

    softskill_result = calculate_soft_skill_score(
        resume_text
    )

    ##########################################################
    # Individual Values
    ##########################################################

    skill_score = skill_result["skill_score"]

    experience_score = experience_result["experience_score"]

    education_score = education_result["education_score"]

    project_score = project_result["project_score"]

    certification_score = certification_result["certification_score"]

    softskill_score = softskill_result["soft_skill_score"]

    ##########################################################
    # Weighted Score
    ##########################################################

    final_score = (

        skill_score *
        (weights.get("Skills",0)/100)

        +

        experience_score *
        (weights.get("Experience",0)/100)

        +

        education_score *
        (weights.get("Education",0)/100)

        +

        project_score *
        (weights.get("Projects",0)/100)

        +

        certification_score *
        (weights.get("Certifications",0)/100)

        +

        softskill_score *
        (weights.get("Soft_Skills",0)/100)

    )

    final_score = round(
        final_score,
        2
    )

    ##########################################################
    # Recommendation
    ##########################################################

    if final_score >= 90:

        recommendation = "Highly Recommended"

    elif final_score >= 80:

        recommendation = "Recommended"

    elif final_score >= 70:

        recommendation = "Consider"

    elif final_score >= 60:

        recommendation = "Hold"

    else:

        recommendation = "Reject"

    ##########################################################
    # Return Complete Analysis
    ##########################################################

    return {

        "Final Score": final_score,

        "Recommendation": recommendation,

        "Skill Score": skill_score,

        "Experience Score": experience_score,

        "Education Score": education_score,

        "Project Score": project_score,

        "Certification Score": certification_score,

        "Soft Skill Score": softskill_score,

        "Matched Required Skills":
            skill_result["matched_required"],

        "Missing Required Skills":
            skill_result["missing_required"],

        "Matched Preferred Skills":
            skill_result["matched_preferred"],

        "Candidate Experience":
            experience_result["candidate_experience"],

        "Required Experience":
            experience_result["required_experience"],

        "Candidate Degree":
            education_result["candidate_degree"],

        "Required Degree":
            education_result.get(
                "required_degree",
                []
            ),

        "Projects":
            project_result["projects_found"],

        "Certifications":
            certification_result["certifications_found"],

        "Soft Skills":
            softskill_result["soft_skills_found"]

    }

##############################################################
# SEMANTIC RESUME – JOB MATCHING
##############################################################

@st.cache_data(show_spinner=False)
def calculate_semantic_similarity(
    resume_text,
    jd_text
):
    """
    Computes semantic similarity between
    Resume and Job Description using
    Sentence Transformers.
    """

    if not resume_text or not jd_text:
        return 0.0

    try:

        resume_embedding = model.encode(
            resume_text,
            convert_to_numpy=True
        )

        jd_embedding = model.encode(
            jd_text,
            convert_to_numpy=True
        )

        similarity = cosine_similarity(
            [resume_embedding],
            [jd_embedding]
        )[0][0]

        score = round(
            similarity * 100,
            2
        )

        score = max(
            0,
            min(score,100)
        )

        return score

    except Exception:

        return 0.0
