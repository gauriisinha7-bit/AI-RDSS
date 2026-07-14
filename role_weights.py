###############################################################
# ROLE WEIGHTS
###############################################################

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

    },

    "Business Analyst":{

        "semantic":0.20,

        "skills":0.25,

        "experience":0.20,

        "education":0.10,

        "projects":0.10,

        "certifications":0.05,

        "soft_skills":0.10

    },

    "AI Engineer":{

        "semantic":0.25,

        "skills":0.30,

        "experience":0.15,

        "education":0.10,

        "projects":0.10,

        "certifications":0.05,

        "soft_skills":0.05

    }

}
from role_weights import ROLE_WEIGHTS
###############################################################
# JOB ROLE
###############################################################

selected_role = st.selectbox(

    "Select Job Role",

    list(ROLE_WEIGHTS.keys())

)
role_weights = ROLE_WEIGHTS[
    selected_role
]
###############################################################
# ROLE WEIGHTS
###############################################################

st.subheader("Role-Based Evaluation Criteria")

weight_df = pd.DataFrame({

    "Criterion":[

        "Semantic",

        "Skills",

        "Experience",

        "Education",

        "Projects",

        "Certifications",

        "Soft Skills"

    ],

    "Weight":[

        role_weights["semantic"],

        role_weights["skills"],

        role_weights["experience"],

        role_weights["education"],

        role_weights["projects"],

        role_weights["certifications"],

        role_weights["soft_skills"]

    ]

})

st.dataframe(

    weight_df,

    use_container_width=True

)
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
###############################################################
# ROLE EXPLANATION
###############################################################

st.info(

f"""
The candidate was evaluated using
**{selected_role}**
criteria.

Different job roles assign different importance
to skills, experience, education, projects,
soft skills and semantic similarity.

This ensures fair role-specific recruitment.
"""

)
