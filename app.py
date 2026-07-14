import streamlit as st
import pandas as pd
import json
import fitz
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="AI-RDSS",
    layout="wide"
)


st.title("AI-RDSS Candidate Recommendation System")
@st.cache_resource
def load_ai_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


model = load_ai_model()

# ==========================
# LOAD KNOWLEDGE BASE
# ==========================

with open("recruitment_knowledge_base.json", "r") as f:
    job_database = json.load(f)


job_role = st.sidebar.selectbox(
    "Select Job Role",
    list(job_database.keys())
)


job_data = job_database[job_role]

weights = job_data["weights"]


# ==========================
# JOB DESCRIPTION
# ==========================

job_description_file = st.sidebar.file_uploader(
    "Upload Job Description PDF",
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

    st.sidebar.success(
        "Job Description Loaded"
    )

else:

    jd_text = ""



# ==========================
# RESUME EXTRACTION
# ==========================

def extract_resume_text(file):

    text = ""

    pdf = fitz.open(
        stream=file.read(),
        filetype="pdf"
    )


    for page in pdf:

        text += page.get_text()


    return text



# ==========================
# SKILL EXTRACTION
# ==========================

def extract_skills(text):

    skill_list = [

        "Python",
        "SQL",
        "Git",
        "Data Structures",
        "Algorithms",
        "Docker",
        "AWS",
        "Linux",
        "REST API",
        "Java",
        "C++",
        "JavaScript"

    ]


    found = []


    text = text.lower()


    for skill in skill_list:

        if skill.lower() in text:

            found.append(skill)


    return found



# ==========================
# SCORE FUNCTIONS
# ==========================

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

            len(matched) / len(required)

        ) * 100



    return round(score,2), matched, missing



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



def calculate_final_score(

        skill,

        experience,

        education

):

    score = (

        skill * weights["Skills"]/100

        +

        experience * weights["Experience"]/100

        +

        education * weights["Education"]/100

    )


    return round(score,2)


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
# ==========================
# RESUME ANALYSIS
# ==========================


uploaded_resumes = st.file_uploader(

    "Upload Candidate Resumes",

    type=["pdf"],

    accept_multiple_files=True

)



if st.button("Analyze Candidates"):


    if uploaded_resumes:


        results = []


        for resume in uploaded_resumes:


            resume_text = extract_resume_text(resume)


            candidate_name = resume.name.replace(

                ".pdf",

                ""

            )


            skills = extract_skills(

                resume_text

            )


            skill_score, matched, missing = calculate_skill_score(

                skills

            )


            experience_score = calculate_experience_score(

                resume_text

            )


            education_score = calculate_education_score(

                resume_text

            )

ai_score = calculate_ai_similarity(

    resume_text,

    jd_text

)
        final_score = calculate_final_score(

                skill_score,

                experience_score,

                education_score

            )



            results.append(

                {

                    "Candidate Name": candidate_name,

                    "Skill Score": skill_score,

                    "Experience Score": experience_score,

                    "Education Score": education_score,

                    "Final Score": final_score,

                    "Matched Skills": matched,

                    "Missing Skills": missing

                }

            )



        result_df = pd.DataFrame(results)



        result_df = result_df.sort_values(

            by="Final Score",

            ascending=False

        )


        st.session_state["result_df"] = result_df



    else:


        st.warning(

            "Please upload resumes"

        )



# ==========================
# LOAD RESULTS
# ==========================


if "result_df" in st.session_state:


    result_df = st.session_state["result_df"]



    # ==========================
    # RANKING TABLE
    # ==========================


    st.subheader(

        "AI-RDSS Candidate Ranking"

    )


    ranking_df = result_df.copy()


    st.dataframe(

        ranking_df.drop(

            columns=[

                "Matched Skills",

                "Missing Skills"

            ]

        ),

        use_container_width=True

    )



    # ==========================
    # DECISION
    # ==========================


    st.subheader(

        "Final Recruitment Decision"

    )


    def get_decision(score):

        if score >= 70:

            return "Recommended"

        elif score >= 50:

            return "Consider"

        else:

            return "Not Recommended"



    decision_df = result_df.copy()


    decision_df["Decision"] = decision_df[

        "Final Score"

    ].apply(

        get_decision

    )


    st.dataframe(

        decision_df[

            [

                "Candidate Name",

                "Final Score",

                "Decision"

            ]

        ],

        use_container_width=True

    )



    # ==========================
    # CANDIDATE PROFILE
    # ==========================


    st.subheader(

        "Candidate Detailed Analysis"

    )


    selected_candidate_name = st.selectbox(

        "Select Candidate",

        result_df["Candidate Name"].tolist(),

        key="profile_candidate"

    )


    candidate = result_df[

        result_df["Candidate Name"] == selected_candidate_name

    ].iloc[0]



    col1, col2, col3, col4 = st.columns(4)


    col1.metric(

        "Skill Score",

        candidate["Skill Score"]

    )


    col2.metric(

        "Experience Score",

        candidate["Experience Score"]

    )


    col3.metric(

        "Education Score",

        candidate["Education Score"]

    )


    col4.metric(

        "Final Score",

        candidate["Final Score"]

    )
    # ==========================
    # SKILL ANALYSIS
    # ==========================


    st.subheader(

        "Matched Skills"

    )


    for skill in candidate["Matched Skills"]:

        st.success(

            "✓ " + skill

        )



    st.subheader(

        "Missing Skills"

    )


    for skill in candidate["Missing Skills"]:

        st.error(

            "✗ " + skill

        )



    # ==========================
    # SHAP STYLE EXPLANATION
    # ==========================


    st.subheader(

        "AI Decision Explanation"

    )


    contribution_df = pd.DataFrame(

        {

            "Feature": [

                "Skills",

                "Experience",

                "Education"

            ],


            "Contribution": [

                candidate["Skill Score"] * weights["Skills"] / 100,

                candidate["Experience Score"] * weights["Experience"] / 100,

                candidate["Education Score"] * weights["Education"] / 100

            ]

        }

    )



    st.dataframe(

        contribution_df,

        use_container_width=True

    )


    st.bar_chart(

        contribution_df.set_index(

            "Feature"

        )

    )



    # ==========================
    # CANDIDATE COMPARISON
    # ==========================


    if len(result_df) > 1:


        st.subheader(

            "Candidate Comparison"

        )


        comparison = result_df[

            [

                "Candidate Name",

                "Skill Score",

                "Experience Score",

                "Education Score",

                "Final Score"

            ]

        ]


        st.dataframe(

            comparison,

            use_container_width=True

        )


        st.bar_chart(

            comparison.set_index(

                "Candidate Name"

            )

        )



    # ==========================
    # DOWNLOAD REPORT
    # ==========================


    st.subheader(

        "Download Recruitment Report"

    )


    report_df = result_df.copy()


    report_df["Decision"] = report_df[

        "Final Score"

    ].apply(

        get_decision

    )


    csv_file = report_df.to_csv(

        index=False

    )


    st.download_button(

        label="Download CSV Report",

        data=csv_file,

        file_name="AI_RDSS_Candidate_Report.csv",

        mime="text/csv"

    )
