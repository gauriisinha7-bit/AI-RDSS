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



def calculate_final_score(

    skill,

    experience,

    education

):

    final = (

        skill * weights["Skills"]/100

        +

        experience * weights["Experience"]/100

        +

        education * weights["Education"]/100

    )


    return round(final,2)

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


            candidate_name = resume.name.replace(".pdf","")


            skills = extract_skills(resume_text)


            skill_score, matched = calculate_skill_score(skills)


            exp_score = calculate_experience_score(

                resume_text

            )


            edu_score = calculate_education_score(

                resume_text

            )


            final_score = calculate_final_score(

                skill_score,

                exp_score,

                edu_score

            )


            results.append({


                "Candidate Name":

                candidate_name,


                "Skill Score":

                skill_score,


                "Experience Score":

                exp_score,


                "Education Score":

                edu_score,


                "Final Score":

                final_score,


                "Matched Skills":

                ", ".join(matched)


            })



        result_df = pd.DataFrame(results)


        result_df = result_df.sort_values(

            by="Final Score",

            ascending=False

        )


        st.subheader(

            "AI-RDSS Candidate Ranking"

        )


        st.dataframe(

            result_df,

            use_container_width=True

        )


    else:


        st.warning(

            "Please upload resumes"

        )



st.subheader("AI-RDSS Decision Explanation")


if 'result_df' in locals():

    for index, row in result_df.iterrows():

        st.write("### Candidate:", row["Candidate Name"])

        reasons = []


        if row["Skill Score"] >= 70:

            reasons.append(
                "Strong skill alignment with job requirements"
            )

        else:

            reasons.append(
                "Skill gap detected"
            )


        if row["Experience Score"] >= 70:

            reasons.append(
                "Experience requirement satisfied"
            )


        else:

            reasons.append(
                "Experience requirement not fully satisfied"
            )


        if row["Education Score"] >= 70:

            reasons.append(
                "Education requirement satisfied"
            )


        else:

            reasons.append(
                "Education requirement not matched"
            )


        for r in reasons:

            st.write("✓", r)


        st.divider()



st.subheader("Final Recruitment Decision")


if 'result_df' in locals():

    decision_df = result_df.copy()


    def get_decision(score):

        if score >= 70:

            return "Recommended"

        elif score >= 50:

            return "Consider"

        else:

            return "Not Recommended"



    decision_df["Decision"] = decision_df["Final Score"].apply(

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
