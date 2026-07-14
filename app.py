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

st.title(" AI-RDSS Candidate Recommendation System")
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

# ============================================
# RESUME UPLOAD
# ============================================

uploaded_resumes = st.file_uploader(

    "Upload Candidate Resumes",

    type=["pdf"],

    accept_multiple_files=True

)


# ============================================
# ANALYZE RESUMES
# ============================================

if st.button("Analyze Candidates"):

    if uploaded_resumes:

        results = []

        with st.spinner("Analyzing resumes..."):

            for resume in uploaded_resumes:

                resume_text = extract_resume_text(resume)

                candidate_name = resume.name.replace(

                    ".pdf",

                    ""

                )

                candidate_skills = extract_skills(

                    resume_text

                )

                skill_score, matched, missing = calculate_skill_score(

                    candidate_skills

                )

                experience_score = calculate_experience_score(

                    resume_text

                )

                education_score = calculate_education_score(

                    resume_text

                )

                certification_score = calculate_certification_score(

                    resume_text

                )

                project_score = calculate_project_score(

                    resume_text

                )

                softskill_score = calculate_softskill_score(

                    resume_text

                )

                ai_score = calculate_ai_similarity(

                    resume_text,

                    jd_text

                )

                final_score = calculate_final_score(

                    skill_score,

                    experience_score,

                    education_score,

                    ai_score,

                    certification_score,

                    project_score,

                    softskill_score

                )

                if final_score >= 80:

                    decision = "Recommended"

                elif final_score >= 60:

                    decision = "Consider"

                else:

                    decision = "Not Recommended"

                results.append(

                    {

                        "Candidate Name": candidate_name,

                        "Skill Score": skill_score,

                        "Experience Score": experience_score,

                        "Education Score": education_score,

                        "Certification Score": certification_score,

                        "Project Score": project_score,

                        "Soft Skill Score": softskill_score,

                        "AI Match Score": ai_score,

                        "Final Score": final_score,

                        "Decision": decision,

                        "Matched Skills": matched,

                        "Missing Skills": missing

                    }

                )

        result_df = pd.DataFrame(results)

        result_df = result_df.sort_values(

            by="Final Score",

            ascending=False

        ).reset_index(drop=True)

        st.session_state["result_df"] = result_df

    else:

        st.warning(

            "Please upload resumes."

        )


# ============================================
# SHOW RANKING
# ============================================

if "result_df" in st.session_state:

    result_df = st.session_state["result_df"]

    st.subheader("Candidate Ranking")

    st.dataframe(

        result_df[

            [

                "Candidate Name",

                "AI Match Score",

                "Skill Score",

                "Experience Score",

                "Final Score",

                "Decision"

            ]

        ],

        use_container_width=True

    )

# ============================================
# RECRUITER DASHBOARD
# ============================================

if "result_df" in st.session_state:

    result_df = st.session_state["result_df"]

    st.divider()

    st.header("Recruiter Dashboard")

    selected_candidate = st.selectbox(

        "Select Candidate",

        result_df["Candidate Name"],

        key="candidate_dashboard"

    )

    candidate = result_df[

        result_df["Candidate Name"] == selected_candidate

    ].iloc[0]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(

        "Final Score",

        f"{candidate['Final Score']}%"

    )

    col2.metric(

        "AI Match",

        f"{candidate['AI Match Score']}%"

    )

    col3.metric(

        "Skill Score",

        f"{candidate['Skill Score']}%"

    )

    col4.metric(

        "Decision",

        candidate["Decision"]

    )

    st.divider()

    left, right = st.columns(2)

    with left:

        st.subheader("Matched Skills")

        if len(candidate["Matched Skills"]) > 0:

            for skill in candidate["Matched Skills"]:

                st.success(skill)

        else:

            st.warning("No skills matched")

    with right:

        st.subheader("Missing Skills")

        if len(candidate["Missing Skills"]) > 0:

            for skill in candidate["Missing Skills"]:

                st.error(skill)

        else:

            st.success("No missing skills")

    st.divider()

    st.subheader("Feature Contribution")

    contribution_df = pd.DataFrame(

        {

            "Feature":[

                "Skills",

                "Experience",

                "Education",

                "AI Match",

                "Certification",

                "Projects",

                "Soft Skills"

            ],

            "Contribution":[

                candidate["Skill Score"]*0.30,

                candidate["Experience Score"]*0.15,

                candidate["Education Score"]*0.10,

                candidate["AI Match Score"]*0.20,

                candidate["Certification Score"]*0.10,

                candidate["Project Score"]*0.10,

                candidate["Soft Skill Score"]*0.05

            ]

        }

    )

    st.bar_chart(

        contribution_df.set_index(

            "Feature"

        )

    )

# ============================================
# CANDIDATE FEEDBACK
# ============================================

def generate_candidate_feedback(candidate):

    feedback = []

    if candidate["Decision"] == "Recommended":

        feedback.append(
            "Excellent profile matching the job requirements."
        )

    elif candidate["Decision"] == "Consider":

        feedback.append(
            "Good profile with some improvement areas."
        )

    else:

        feedback.append(
            "Currently not suitable for this role."
        )

    if len(candidate["Matched Skills"]) > 0:

        feedback.append(

            "Strong Skills: "

            +

            ", ".join(candidate["Matched Skills"])

        )

    if len(candidate["Missing Skills"]) > 0:

        feedback.append(

            "Need to improve: "

            +

            ", ".join(candidate["Missing Skills"])

        )

    if candidate["Experience Score"] < 100:

        feedback.append(

            "Gain more relevant industry experience."

        )

    if candidate["Certification Score"] < 50:

        feedback.append(

            "Adding professional certifications will strengthen your profile."

        )

    if candidate["Project Score"] < 60:

        feedback.append(

            "Build more real-world projects and upload them on GitHub."

        )

    return feedback

if "result_df" in st.session_state:

    result_df = st.session_state["result_df"]

    selected_candidate = st.selectbox(

        "Select Candidate",

        result_df["Candidate Name"],

        key="feedback_candidate"

    )

    candidate = result_df[

        result_df["Candidate Name"] == selected_candidate

    ].iloc[0]
if "result_df" in st.session_state:

    result_df = st.session_state["result_df"]

    

    # ============================================
    # RECRUITER EXPLANATION
    # ============================================

    st.divider()

    st.subheader("Recruiter Explanation")

    if candidate["Decision"] == "Recommended":

        st.success("Candidate strongly matches the job requirements.")

    elif candidate["Decision"] == "Consider":

        st.warning("Candidate partially satisfies the requirements.")

    else:

        st.error("Candidate does not satisfy the minimum hiring criteria.")

    # ============================================
    # CANDIDATE FEEDBACK
    # ============================================

    st.divider()

    st.subheader("Candidate Feedback")

    feedback = generate_candidate_feedback(candidate)

    for item in feedback:

        st.info(item)

    # ============================================
    # EMAIL
    # ============================================

    st.divider()

    st.subheader("Email Feedback")

    email = f"""
Dear {candidate['Candidate Name']},

Thank you for applying for the {job_role} position.

Final Decision:
{candidate['Decision']}

Final Score:
{candidate['Final Score']}%

Matched Skills:
{", ".join(candidate['Matched Skills'])}

Missing Skills:
{", ".join(candidate['Missing Skills'])}

Regards,
Recruitment Team
"""

    st.text_area(
        "Email",
        email,
        height=250
    )

    # ============================================
    # COMPARISON
    # ============================================

    st.divider()

    st.subheader("Candidate Comparison")

    comparison = result_df[
        [
            "Candidate Name",
            "Skill Score",
            "Experience Score",
            "AI Match Score",
            "Final Score"
        ]
    ]

    st.dataframe(
        comparison,
        use_container_width=True
    )

    st.bar_chart(
        comparison.set_index("Candidate Name")
    )

    # ============================================
    # DOWNLOAD REPORT
    # ============================================

    csv = result_df.to_csv(index=False)

    st.download_button(
        "Download Recruitment Report",
        csv,
        "AI_RDSS_Report.csv",
        "text/csv"
    )

    st.divider()

    st.caption(
        "AI-RDSS Candidate Recommendation System | Explainable AI Recruitment Platform"
    )
