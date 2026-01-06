import streamlit as st
import plotly.graph_objects as go
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(layout="wide")

# ---------------- LOAD BERT MODEL ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ---------------- HEADER ----------------
st.markdown("""
<div style="background:#5B3CE4;padding:22px;border-radius:10px;margin-bottom:20px">
    <h2 style="color:white;">Milestone 3: Skill Gap Analysis and Similarity Matching</h2>
    <p style="color:#e0e0ff;">Simple BERT Embeddings + Cosine Similarity</p>
</div>
""", unsafe_allow_html=True)

st.markdown("## Skill Gap Analysis Interface")

# ---------------- FILE UPLOAD ----------------
c1, c2 = st.columns(2)
with c1:
    resume_file = st.file_uploader("Upload Resume (.txt)", type=["txt"])
with c2:
    jd_file = st.file_uploader("Upload Job Description (.txt)", type=["txt"])

def read_file(file):
    if file:
        return file.read().decode("utf-8").lower()
    return ""

resume_text = read_file(resume_file)
jd_text = read_file(jd_file)

if not resume_text or not jd_text:
    st.info("Please upload both Resume and Job Description files")
    st.stop()

# ---------------- SKILL LIST ----------------
SKILLS = [
    "python", "java", "sql", "mysql", "mongodb",
    "spring boot", "react", "node.js",
    "communication", "leadership"
]

def extract_skills(text):
    found = set()
    for skill in SKILLS:
        if re.search(rf"\b{re.escape(skill)}\b", text):
            found.add(skill)
    return list(found)

resume_skills = extract_skills(resume_text)
jd_skills = extract_skills(jd_text)

if not jd_skills:
    st.warning("No skills detected in Job Description")
    st.stop()

# ---------------- BERT EMBEDDINGS ----------------
resume_emb = model.encode(resume_skills)
jd_emb = model.encode(jd_skills)

similarity_matrix = cosine_similarity(jd_emb, resume_emb)

# ---------------- MATCH CLASSIFICATION ----------------
matched, partial, missing = [], [], []

for i, jd_skill in enumerate(jd_skills):
    max_sim = similarity_matrix[i].max() if len(resume_skills) else 0

    if max_sim >= 0.8:
        matched.append(jd_skill)
    elif max_sim >= 0.5:
        partial.append(jd_skill)
    else:
        missing.append(jd_skill)

overall_match = int((len(matched) / len(jd_skills)) * 100)

# ---------------- DASHBOARD LAYOUT ----------------
left, right = st.columns([3, 2])

# ================= SIMILARITY MATRIX =================
with left:
    st.markdown("### Similarity Matrix (BERT-based)")

    fig = go.Figure()

    for i, jd_skill in enumerate(jd_skills):
        for j, rs_skill in enumerate(resume_skills):
            sim = similarity_matrix[i][j]
            color = "green" if sim >= 0.8 else "orange" if sim >= 0.5 else "red"

            fig.add_trace(go.Scatter(
                x=[rs_skill],
                y=[jd_skill],
                mode="markers",
                marker=dict(
                    size=sim * 40,
                    color=color
                ),
                showlegend=False
            ))

    fig.update_layout(
        height=350,
        xaxis_title="Resume Skills",
        yaxis_title="JD Skills",
        plot_bgcolor="white"
    )

    st.plotly_chart(fig, use_container_width=True)

# ================= RIGHT PANEL =================
with right:
    st.markdown("### Skill Match Overview")

    st.metric("Overall Match", f"{overall_match}%")
    st.metric("Matched Skills", len(matched))
    st.metric("Partial Matches", len(partial))
    st.metric("Missing Skills", len(missing))

    donut = go.Figure(go.Pie(
        labels=["Matched", "Partial", "Missing"],
        values=[len(matched), len(partial), len(missing)],
        hole=0.6,
        marker_colors=["green", "orange", "red"],
        textinfo="none"
    ))

    donut.update_layout(height=250)
    st.plotly_chart(donut, use_container_width=True)

# ---------------- SKILL LIST OUTPUT ----------------
st.markdown("### Skill Gap Report")

col1, col2, col3 = st.columns(3)
with col1:
    st.success("Matched Skills")
    st.write(matched)

with col2:
    st.warning("Partial Matches")
    st.write(partial)

with col3:
    st.error("Missing Skills")
    st.write(missing)
