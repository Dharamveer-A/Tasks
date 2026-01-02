import streamlit as st
import pandas as pd
import spacy
import json
import re
from spacy.matcher import PhraseMatcher
from docx import Document
import pdfplumber

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

def load_file(uploaded_file):
    """
    Unified loader for TXT, DOCX, and PDF files
    """
    if uploaded_file is None:
        return ""

    filename = uploaded_file.name.lower()

    if filename.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")

    elif filename.endswith(".docx"):
        doc = Document(uploaded_file)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    elif filename.endswith(".pdf"):
        text = ""
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text

    else:
        st.error("Unsupported file format")
        return ""

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_technical_skills(doc):
    found = set()
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    matcher.add("TECH", [nlp(skill) for skill in master_technical_skills])

    for _, start, end in matcher(doc):
        found.add(doc[start:end].text.lower())

    return list(found)

def extract_soft_skills(doc):
    found = set()
    tokens = {token.text.lower() for token in doc if not token.is_punct}

    for skill in master_soft_skills:
        if set(skill.split()).issubset(tokens):
            found.add(skill)

    return list(found)

def extract_skills(text):
    doc = nlp(clean_text(text))
    return {
        "technical_skills": extract_technical_skills(doc),
        "soft_skills": extract_soft_skills(doc)
    }

def save_to_json(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

st.set_page_config(page_title="Skill Extraction using NLP", layout="wide")
st.title("Skill Extraction using NLP")

menu = st.sidebar.radio("Select Option", ["Resume", "Job Description", "Compare"])

if menu == "Resume":
    st.subheader("Resume Skill Extraction")
    resume_file = st.file_uploader(
        "Upload Resume (TXT / DOCX / PDF)",
        type=["txt", "docx", "pdf"]
    )

    if st.button("Extract Resume Skills"):
        resume_text = load_file(resume_file)
        skills = extract_skills(resume_text)

        save_to_json(skills, "resume_skills.json")

        st.success("Resume skills saved to resume_skills.json")
        st.write("### Technical Skills", skills["technical_skills"])
        st.write("### Soft Skills", skills["soft_skills"])

elif menu == "Job Description":
    st.subheader("Job Description Skill Extraction")
    jd_file = st.file_uploader(
        "Upload Job Description (TXT / DOCX / PDF)",
        type=["txt", "docx", "pdf"]
    )

    if st.button("Extract JD Skills"):
        jd_text = load_file(jd_file)
        skills = extract_skills(jd_text)

        save_to_json(skills, "jd_skills.json")

        st.success("JD skills saved to jd_skills.json")
        st.write("### Technical Skills", skills["technical_skills"])
        st.write("### Soft Skills", skills["soft_skills"])

elif menu == "Compare":
    st.subheader("Resume vs Job Description Comparison")

    resume_file = st.file_uploader(
        "Upload Resume",
        type=["txt", "docx", "pdf"]
    )
    jd_file = st.file_uploader(
        "Upload Job Description",
        type=["txt", "docx", "pdf"]
    )

    if st.button("Compare Skills"):
        resume = extract_skills(load_file(resume_file))
        jd = extract_skills(load_file(jd_file))

        matched_tech = set(resume["technical_skills"]) & set(jd["technical_skills"])
        missing_tech = set(jd["technical_skills"]) - set(resume["technical_skills"])

        matched_soft = set(resume["soft_skills"]) & set(jd["soft_skills"])
        missing_soft = set(jd["soft_skills"]) - set(resume["soft_skills"])

        col1, col2 = st.columns(2)

        with col1:
            st.write("### Matching Technical Skills", list(matched_tech))
            st.write("### Missing Technical Skills", list(missing_tech))

        with col2:
            st.write("### Matching Soft Skills", list(matched_soft))
            st.write("### Missing Soft Skills", list(missing_soft))

        comparison = {
            "resume": resume,
            "job_description": jd,
            "matched": {
                "technical": list(matched_tech),
                "soft": list(matched_soft)
            },
            "missing": {
                "technical": list(missing_tech),
                "soft": list(missing_soft)
            }
        }

        save_to_json(comparison, "comparison_skills.json")
        st.success("Comparison saved to comparison_skills.json")

        df = pd.DataFrame({
            "Category": ["Matched Skills", "Missing Skills"],
            "Count": [
                len(matched_tech) + len(matched_soft),
                len(missing_tech) + len(missing_soft)
            ]
        })

        st.bar_chart(df.set_index("Category"))
