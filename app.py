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

   

               
