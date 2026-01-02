import streamlit as st
import pandas as pd
import spacy
import json
from spacy.matcher import PhraseMatcher
import re

# -------------------- Load spaCy Model --------------------
nlp = spacy.load("en_core_web_trf")

# -------------------- Master Skill Lists --------------------
master_technical_skills = {
    "python", "java", "c", "c++", "c#", "javascript", "typescript",
    "html", "css", "react", "angular", "node.js", "express.js",
    "django", "flask", "fastapi",
    "sql", "mysql", "postgresql", "mongodb",
    "machine learning", "deep learning", "artificial intelligence",
    "tensorflow", "keras", "pytorch", "scikit-learn",
    "nlp", "computer vision",
    "aws", "azure", "gcp", "docker", "kubernetes",
    "git", "github", "linux",
    "data analysis", "data visualization", "power bi", "tableau"
}

master_soft_skills = {
    "communication", "teamwork", "leadership",
    "problem solving", "critical thinking",
    "time management", "adaptability",
    "collaboration", "decision making",
    "analytical thinking", "creativity"
}

# -------------------- Utility Functions --------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# -------------------- Skill Extraction --------------------
def extract_technical_skills(doc):
    found_skills = set()
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp(skill) for skill in master_technical_skills]
    matcher.add("TECH_SKILLS", patterns)

    for _, start, end in matcher(doc):
        found_skills.add(doc[start:end].text.lower())

    return list(found_skills)

def extract_soft_skills(doc):
    found_skills = set()
    tokens = {token.text.lower() for token in doc if not token.is_punct}

    for skill in master_soft_skills:
        if set(skill.split()).issubset(tokens):
            found_skills.add(skill)

    return list(found_skills)

def extract_skills(text):
    text = clean_text(text)
    doc = nlp(text)
    return {
        "technical_skills": extract_technical_skills(doc),
        "soft_skills": extract_soft_skills(doc)
    }

# -------------------- JSON Save --------------------
def save_to_json(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Skill Extraction using NLP", layout="wide")
st.title("Skill Extraction using NLP")

menu = st.sidebar.radio("Select Option", ["Resume", "Job Description", "Compare"])

# -------------------- Resume --------------------
if menu == "Resume":
    st.subheader("Resume Skill Extraction")
    resume_text = st.text_area("Paste Resume Text")

    if st.button("Extract Resume Skills"):
        resume_skills = extract_skills(resume_text)
        save_to_json(resume_skills, "resume_skills.json")

        st.success("Resume skills saved to resume_skills.json")

        st.markdown("### Technical Skills")
        st.write(resume_skills["technical_skills"])

        st.markdown("### Soft Skills")
        st.write(resume_skills["soft_skills"])

# -------------------- Job Description --------------------
elif menu == "Job Description":
    st.subheader("Job Description Skill Extraction")
    jd_text = st.text_area("Paste Job Description Text")

    if st.button("Extract JD Skills"):
        jd_skills = extract_skills(jd_text)
        save_to_json(jd_skills, "jd_skills.json")

        st.success("JD skills saved to jd_skills.json")

        st.markdown("### Technical Skills")
        st.write(jd_skills["technical_skills"])

        st.markdown("### Soft Skills")
        st.write(jd_skills["soft_skills"])

# -------------------- Compare --------------------
elif menu == "Compare":
    st.subheader("Resume vs Job Description Comparison")

    resume_text = st.text_area("Paste Resume Text")
    jd_text = st.text_area("Paste Job Description Text")

    if st.button("Compare Skills"):
        resume = extract_skills(resume_text)
        jd = extract_skills(jd_text)

        resume_tech = set(resume["technical_skills"])
        jd_tech = set(jd["technical_skills"])
        resume_soft = set(resume["soft_skills"])
        jd_soft = set(jd["soft_skills"])

        matched_tech = resume_tech & jd_tech
        missing_tech = jd_tech - resume_tech
        matched_soft = resume_soft & jd_soft
        missing_soft = jd_soft - resume_soft

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### ✅ Matching Technical Skills")
            st.write(list(matched_tech))

            st.markdown("### ❌ Missing Technical Skills")
            st.write(list(missing_tech))

        with col2:
            st.markdown("### ✅ Matching Soft Skills")
            st.write(list(matched_soft))

            st.markdown("### ❌ Missing Soft Skills")
            st.write(list(missing_soft))

        # Save comparison JSON
        comparison_data = {
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

        save_to_json(comparison_data, "comparison_skills.json")
        st.success("Comparison saved to comparison_skills.json")

        # Visualization
        df = pd.DataFrame({
            "Category": ["Matched Skills", "Missing Skills"],
            "Count": [
                len(matched_tech) + len(matched_soft),
                len(missing_tech) + len(missing_soft)
            ]
        })

        st.bar_chart(df.set_index("Category"))
