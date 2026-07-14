import streamlit as st
import pandas as pd
import json
import fitz
import re
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="AI-RDSS",
    layout="wide"
)


st.title("AI-RDSS Candidate Recommendation System")


# Load Knowledge Base

with open("recruitment_knowledge_base.json","r") as f:

    job_database = json.load(f)



job_role = st.sidebar.selectbox(

    "Select Job Role",

    list(job_database.keys())

)


job_data = job_database[job_role]


weights = job_data["weights"]



# Upload Job Description

job_description_file = st.sidebar.file_uploader(

    "Upload Job Description PDF",

    type=["pdf"]

)



def extract_job_description(file):

    text = ""

    pdf = fitz.open(

        stream=file.read(),

        filetype="pdf"

    )


    for page in pdf:

        text += page.get_text()


    return text



if job_description_file:

    jd_text = extract_job_description(job_description_file)

else:

    jd_text = ""
def extract_resume_text(file):

    text = ""

    pdf = fitz.open(

        stream=file.read(),

        filetype="pdf"

    )


    for page in pdf:

        text += page.get_text()


    return text



# ==========================================================
# Skill Extraction
# ==========================================================


def extract_skills(text):

    skills = [

        "Python",
        "SQL",
        "Git",
        "Data Structures",
        "Algorithms",
        "Docker",
        "AWS",
        "Linux",
        "REST API"

    ]


    found = []


    text = text.lower()


    for skill in skills:

        if skill.lower() in text:

            found.append(skill)


    return found



# ==========================================================
# Score Calculation
# ==========================================================


def calculate_skill_score(candidate_skills):

    required = job_data["required_skills"]


    matched = []


    for skill in required:

        if skill in candidate_skills:

            matched.append(skill)



    score = (

        len(matched) / len(required)

    ) * 100



    return round(score,2), matched



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



    required_years = job_data["minimum_experience"]


    if years >= required_years:

        return 100

    elif years > 0:

        return 50

    else:

        return 0



def calculate_education_score(text):

    text = text.lower()


    for edu in job_data["education"]:

        if edu.lower() in text:

            return 100


    if "bachelor" in text:

        return 100


    return 0
   

               
