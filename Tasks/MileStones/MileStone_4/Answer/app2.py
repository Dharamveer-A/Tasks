import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import spacy
from spacy.matcher import PhraseMatcher
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Skill Gap Visual Insights", layout="wide")

# ================= LOAD MODELS =================
@st.cache_resource
def load_models():
    nlp = spacy.load("en_core_web_sm")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return nlp, embed_model

nlp, model = load_models()

# ================= TITLE =================
st.markdown("""
<div>
    <h2 style="color:white;">Milestone 4: Skill Gap Report & Visual Insights</h2>
</div>
""", unsafe_allow_html=True)

# ================= FILE UPLOAD =================
c1, c2 = st.columns(2)
with c1:
    resume_file = st.file_uploader("Upload Resume (.txt)", type=["txt"])
with c2:
    jd_file = st.file_uploader("Upload Job Description (.txt)", type=["txt"])

def read_file(file):
    return file.read().decode("utf-8").lower() if file else ""

resume_text = read_file(resume_file)
jd_text = read_file(jd_file)

if not resume_text or not jd_text:
    st.info("Upload both Resume and Job Description to continue")
    st.stop()

# ================= SKILL LISTS =================
TECHNICAL_SKILLS = [
    "python","java","c","c++","javascript","typescript","html","css",
    "react","angular","node.js","django","flask","fastapi","spring boot",
    "mysql","postgresql","mongodb","redis","aws","azure","gcp",
    "docker","kubernetes","git","linux","machine learning","deep learning",
    "nlp","computer vision","tensorflow","pytorch","scikit-learn",
    "data analysis","power bi","tableau","spark","hadoop","rest api"
]

SOFT_SKILLS = [
    "communication","teamwork","collaboration","leadership",
    "problem solving","critical thinking","time management",
    "adaptability","creativity","decision making",
    "emotional intelligence","conflict resolution",
    "presentation skills","customer focus","work ethic"
]

ALL_SKILLS = TECHNICAL_SKILLS + SOFT_SKILLS

# ================= PHRASE MATCHER =================
def phrase_extract(text, skills):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    matcher.add("SKILLS", [nlp.make_doc(s) for s in skills])
    doc = nlp(text)
    return sorted(set(doc[start:end].text.lower() for _, start, end in matcher(doc)))

resume_skills = phrase_extract(resume_text, ALL_SKILLS)
jd_skills = phrase_extract(jd_text, ALL_SKILLS)

# ================= EMBEDDINGS (FOR HEATMAP ONLY) =================
resume_emb = model.encode(resume_skills) if resume_skills else np.array([])
jd_emb = model.encode(jd_skills) if jd_skills else np.array([])
similarity = cosine_similarity(jd_emb, resume_emb) if len(resume_emb) else np.zeros((len(jd_skills), 1))

# ================= RULE + PARTIAL MATCH LOGIC =================
def classify_skills(jd_skills, resume_skills):
    matched, partial, missing = [], [], []

    for jd in jd_skills:
        if jd in resume_skills:
            matched.append(jd)
        else:
            jd_doc = nlp(jd)
            sims = [jd_doc.similarity(nlp(rs)) for rs in resume_skills]
            if sims and max(sims) >= 0.65:
                partial.append(jd)
            else:
                missing.append(jd)

    return matched, partial, missing

matched, partial, missing = classify_skills(jd_skills, resume_skills)

# ================= SPLIT TECH & SOFT =================
def split_type(skills):
    tech = [s for s in skills if s in TECHNICAL_SKILLS]
    soft = [s for s in skills if s in SOFT_SKILLS]
    return tech, soft

matched_tech, matched_soft = split_type(matched)
partial_tech, partial_soft = split_type(partial)
missing_tech, missing_soft = split_type(missing)

# ================= METRICS =================
overall_match = int((len(matched) / len(jd_skills)) * 100) if jd_skills else 0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Overall Match", f"{overall_match}%")
k2.metric("Matched Skills", len(matched))
k3.metric("Partial Matches", len(partial))
k4.metric("Missing Skills", len(missing))

# ================= TABS =================
tab1, tab2, tab3 = st.tabs([
    "Similarity Heatmap",
    "Skill Gap Report",
    "Missing Skill Ranking"
])

# ================= TAB 1 : HEATMAP (UNCHANGED) =================
with tab1:
    st.subheader("Skill Similarity Heatmap")
    fig = go.Figure(data=go.Heatmap(
        z=similarity,
        x=resume_skills,
        y=jd_skills
    ))
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# ================= TAB 2 : ONLY UI CHANGE =================
with tab2:
    c1, c2 = st.columns(2)

    with c1:
        st.success("Matched Skills – Technical")
        st.write(matched_tech)

        st.warning("Partially Matched – Technical")
        st.write(partial_tech)

    with c2:
        st.success("Matched Skills – Soft")
        st.write(matched_soft)

        st.warning("Partially Matched – Soft")
        st.write(partial_soft)

# ================= TAB 3 : RANKING =================
with tab3:
    if missing:
        rank_df = pd.DataFrame({
            "Skill": missing,
            "Criticality": [round(1 - max(similarity[i]) if similarity.size else 1, 2)
                            for i in range(len(missing))]
        }).sort_values("Criticality", ascending=False)

        st.dataframe(rank_df, use_container_width=True)

        st.download_button(
            "Download Skill Gap Report",
            rank_df.to_csv(index=False),
            file_name="skill_gap_report.csv",
            mime="text/csv"
        )
    else:
        st.success("No critical skill gaps found 🎉")
