import streamlit as st
import json
import re
from docx import Document
import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher

nlp = spacy.load("en_core_web_trf")

master_technical_skills = {
    # Programming Languages
    "python", "java", "c", "c++", "c#", "go", "rust", "kotlin", "swift",
    "javascript", "typescript", "php", "ruby", "r", "matlab", "scala",
    "perl", "bash", "shell scripting",

    # Web Technologies
    "html", "css", "sass", "bootstrap", "tailwind css",
    "react", "angular", "vue.js", "next.js", "nuxt.js",
    "node.js", "express.js", "django", "flask", "fastapi",
    "spring", "spring boot", "laravel", "asp.net",

    # Databases
    "mysql", "postgresql", "sqlite", "oracle", "sql server",
    "mongodb", "cassandra", "dynamodb", "redis", "firebase",
    "elasticsearch", "neo4j",

    # Data Science & Analytics
    "numpy", "pandas", "scipy", "matplotlib", "seaborn",
    "plotly", "power bi", "tableau", "excel", "statistics",
    "data analysis", "data visualization",

    # Machine Learning & AI
    "machine learning", "deep learning", "artificial intelligence",
    "tensorflow", "keras", "pytorch", "scikit-learn", "xgboost",
    "opencv", "nlp", "computer vision", "speech recognition",
    "transformers", "huggingface", "langchain", "llm",

    # Big Data
    "hadoop", "spark", "pyspark", "kafka", "hive", "pig",
    "hbase", "flink", "airflow", "databricks",

    # Cloud & DevOps
    "aws", "azure", "google cloud", "gcp",
    "docker", "kubernetes", "terraform", "ansible",
    "jenkins", "github actions", "ci/cd",
    "linux", "unix",

    # APIs & Backend
    "rest api", "graphql", "grpc", "soap",
    "microservices", "jwt", "oauth",

    # Testing
    "unit testing", "integration testing", "system testing",
    "pytest", "unittest", "junit", "selenium", "cypress",
    "postman",

    # Mobile Development
    "android", "ios", "flutter", "react native",
    "swiftui", "kotlin multiplatform",

    # Cybersecurity
    "network security", "application security", "penetration testing",
    "ethical hacking", "cryptography", "owasp",
    "siem", "firewall",

    # Operating Systems
    "windows", "linux", "macos",

    # Version Control & Tools
    "git", "github", "gitlab", "bitbucket",
    "jira", "confluence", "trello",

    # Other Technical Skills
    "data structures", "algorithms", "object oriented programming",
    "design patterns", "system design",
    "software development lifecycle", "agile", "scrum"
}

master_soft_skills = {
    # Communication
    "communication", "verbal communication", "written communication",
    "public speaking", "presentation skills", "active listening",
    "business communication", "storytelling",

    # Interpersonal Skills
    "teamwork", "collaboration", "interpersonal skills",
    "relationship building", "empathy", "emotional intelligence",
    "conflict resolution", "negotiation",

    # Leadership
    "leadership", "people management", "team leadership",
    "decision making", "delegation", "mentoring", "coaching",
    "influencing", "strategic thinking",

    # Problem Solving & Thinking
    "problem solving", "critical thinking", "analytical thinking",
    "logical reasoning", "creative thinking", "innovation",
    "root cause analysis", "troubleshooting",

    # Time & Work Management
    "time management", "prioritization", "multitasking",
    "work ethic", "self discipline", "accountability",
    "goal setting", "organizational skills",

    # Adaptability & Learning
    "adaptability", "flexibility", "resilience",
    "learning agility", "continuous learning",
    "open mindedness", "growth mindset",

    # Professionalism
    "professionalism", "integrity", "ethical behavior",
    "reliability", "punctuality", "confidentiality",

    # Creativity & Innovation
    "creativity", "idea generation", "design thinking",
    "innovation mindset", "curiosity",

    # Emotional & Personal Skills
    "stress management", "self awareness", "self motivation",
    "confidence", "positive attitude", "emotional control",

    # Customer & Service Orientation
    "customer focus", "customer service",
    "client management", "stakeholder management",
    "user empathy", "service mindset",

    # Collaboration & Culture
    "cross functional collaboration", "cultural awareness",
    "diversity and inclusion", "remote collaboration",
    "team alignment",

    # Conflict & Crisis Handling
    "conflict management", "crisis management",
    "handling pressure", "decision making under pressure",

    # Work Style
    "independent work", "collaborative work",
    "attention to detail", "quality focus",
    "result oriented", "ownership",

    # Ethics & Values
    "honesty", "trustworthiness", "respect",
    "fairness", "social responsibility"
}

def loadDocx(file):
    """
    This function will load the document file and return the content as string
    args : file -> uploaded docx file (Streamlit)
    return-type : string
    """
    content = ""
    word = []
    document = Document(file)

    for line in document.paragraphs:
        text = line.text.strip()
        if text:
            word.append(text)

    content = "\n".join(word)
    return content

def cleanText(content):
    """
    This function will clean the content
    args : content -> string
    return-type : string
    """
    clean = content.lower()
    clean = re.sub(r"[^a-zA-Z0-9/:\s-]", "", clean)
    clean = re.sub(r"\s+", " ", clean)
    return clean.strip()

def extract_technical_skills(doc) :
    found_technical_skills = set()
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp(skill) for skill in master_technical_skills]
    matcher.add("SKILLS", patterns)
    matches = matcher(doc)
    for id, start, end in matches :
        found_technical_skills.add(doc[start:end])
    return list(found_technical_skills)

def extract_soft_skills(doc) :
    found_soft_skills = set()
    tokens = {token.text.lower() for token in doc if not token.is_punct}
    for skill in master_soft_skills :
        skill_tokens = set(skill.split())
        if skill_tokens.issubset(tokens) : 
            found_soft_skills.add(skill)
    return list(found_soft_skills)

def extract_skills(text) :
    doc = nlp(text)
    return {
        "techinical_skills" : extract_technical_skills(doc),
        "soft_skills" : extract_soft_skills(doc)
    }

def save_skills_to_json(skills, filename="extracted_skills.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(skills, f, indent=4)

st.set_page_config(page_title="Skill Extraction using NLP", layout="wide")

st.title("Skill Extraction using NLP")

menu = st.sidebar.radio(
    "Select Input Type",
    ["Resume", "Job Description", "Compare"]
)

if menu == "Resume":
    st.subheader("Resume Skills Extraction")

    resume_text = st.text_area("Paste Resume Text")

    if st.button("Extract Resume Skills"):
        resume_skills = extract_skills(resume_text)

        st.markdown("### Technical Skills")
        st.write(resume_skills["technical_skills"])

        st.markdown("### Soft Skills")
        st.write(resume_skills["soft_skills"])

elif menu == "Job Description":
    st.subheader("Job Description Skills Extraction")

    jd_text = st.text_area("Paste Job Description Text")

    if st.button("Extract JD Skills"):
        jd_skills = extract_skills(jd_text)

        st.markdown("### Technical Skills")
        st.write(jd_skills["technical_skills"])

        st.markdown("### Soft Skills")
        st.write(jd_skills["soft_skills"])

elif menu == "Compare":
    st.subheader("Resume vs Job Description Skill Comparison")

    resume_text = st.text_area("Paste Resume Text")
    jd_text = st.text_area("Paste Job Description Text")

    if st.button("Compare Skills"):
        resume = extract_skills(resume_text)
        jd = extract_skills(jd_text)

        # Convert to sets
        resume_tech = set(resume["technical_skills"])
        jd_tech = set(jd["technical_skills"])

        resume_soft = set(resume["soft_skills"])
        jd_soft = set(jd["soft_skills"])

        # Matching and missing
        matched_tech = resume_tech & jd_tech
        missing_tech = jd_tech - resume_tech

        matched_soft = resume_soft & jd_soft
        missing_soft = jd_soft - resume_soft

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Matching Technical Skills")
            st.write(list(matched_tech))

            st.markdown("### Missing Technical Skills")
            st.write(list(missing_tech))

        with col2:
            st.markdown("### Matching Soft Skills")
            st.write(list(matched_soft))

            st.markdown("### Missing Soft Skills")
            st.write(list(missing_soft))

data = {
    "Category": ["Matched Skills", "Missing Skills"],
    "Count": [
        len(matched_tech) + len(matched_soft),
        len(missing_tech) + len(missing_soft)
    ]
}

df = pd.DataFrame(data)
st.bar_chart(df.set_index("Category"))
