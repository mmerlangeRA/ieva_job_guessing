# Idéé

Let's assess some job characteristics from description

## run

### Create virtual env

python3.9 -m venv venv
source venv/bin/activate

### Install dependencies

pip install -r requirements.txt
streamlit run app.py


## Rules for inference
Check prompt-template.txt, copy below
Below is a job description.
Analyze it and assess the following fields:

- domain
- expertise_level  
- management_level
- experience_level
- study_level
- language_skills

There are domains:

- Drilling and Well Engineer
- information systems
- industrial projects

For expertise_level, there are 4 levels: 0 for no expertise required, 1 for a small team referent, 2 for 10 to 100 team referent, 3 for entreprise expert referent
For management_level, there are 3 levels : 0 for no management, 1 for less than 10 internal people, 2 for more than 10.
For experience_level, there are 3 levels : 0 for less than 5 years experience, 1 for more than 5 years, 2 for more than 10
For study_level, there are 4 levels : 0 for special study required, 1 for bachelor level, 2 for enginneer, 3 for phD
For language_skills, there are 3 levels : 0 no special skills required (only one language), 1 for working knowledge in at least a second language, 2 for fluency in at least one other language

Answer with a formalized json. For each of these 6 fields, returns your "assessed" level and "details"
If you are not confident, set "assessed" as -1 and justify in the "details" subfield
Example :
{{
    "domain":{{
        "assessed":"information systems",
        "details":"backend Development"
    }}
    "expertise_level":{{
        "assessed":1,
        "details":"seems like basic expertise"
    }},
    "management_level":{{
        "assessed":0,
        "details":"no information to assess, I estimate that is not required"
    }},
    "experience_level":{{
        "assessed":1,
        "details":"it is precised that at least 7 years experience is required"
    }},
    "study_level":{{
        "assessed":-1,
        "details":"I could not find any indication"
    }},
    "language_skills":{{
        "assessed":0,
        "details":"no information to assess, I estimate that is not required"
    }}

}}

{file_content}
