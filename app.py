import streamlit as st
import pandas as pd
import json
import fitz
import re


st.set_page_config(
    page_title="AI-RDSS",
    layout="wide"
)


st.title("AI-RDSS Candidate Recommendation System")


# Load knowledge base

with open("recruitment_knowledge_base.json","r") as f:

    job_database = json.load(f)


job_role = "Software Engineer"

job_data = job_database[job_role]


weights = job_data["weights"]



# Resume text extraction

def extract_resume_text(file):

    text = ""

    pdf = fitz.open(stream=file.read(), filetype="pdf")


    for page in pdf:

        text += page.get_text()


    return text



# Basic skill extraction

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
