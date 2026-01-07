import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Skill Gap Visual Insights", layout="wide")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

st.markdown("""
<div>
    <h2 style="color:white;">Milestone 4: Skill Gap Report & Visual Insights</h2>
</div>
""", unsafe_allow_html=True)

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
    st.info("Upload both Resume and Job Description to continue")
    st.stop()

SKILLS = [
    "python","java","sql","mysql","mongodb","aws",
    "react","node.js","spring boot",
    "machine learning","data analysis",
    "communication","leadership"
]

def extract_skills(text):
    return [s for s in SKILLS if s in text]

resume_skills = extract_skills(resume_text)
jd_skills = extract_skills(jd_text)

resume_emb = model.encode(resume_skills)
jd_emb = model.encode(jd_skills)

similarity = cosine_similarity(jd_emb, resume_emb)

matched = []
partial = []
missing = []
ranking = []

for i, jd_skill in enumerate(jd_skills):
    max_sim = similarity[i].max() if resume_skills else 0

    if max_sim >= 0.8:
        matched.append(jd_skill)
    elif max_sim >= 0.5:
        partial.append(jd_skill)
    else:
        missing.append(jd_skill)
        ranking.append((jd_skill, round(1 - max_sim, 2)))

overall_match = int((len(matched) / len(jd_skills)) * 100)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Overall Match", f"{overall_match}%")
k2.metric("Matched Skills", len(matched))
k3.metric("Partial Matches", len(partial))
k4.metric("Missing Skills", len(missing))

tab1, tab2, tab3 = st.tabs([
    "Similarity Heatmap",
    "Skill Gap Report",
    "Missing Skill Ranking"
])

with tab1:
    st.subheader("Skill Similarity Heatmap")

    heatmap = go.Figure(data=go.Heatmap(
        z=similarity,
        x=resume_skills,
        y=jd_skills
    ))

    heatmap.update_layout(height=500)
    st.plotly_chart(heatmap, use_container_width=True)

with tab2:
    c1, c2, c3 = st.columns(3)

    with c1:
        st.success("Matched Skills")
        st.write(matched)

    with c2:
        st.warning("Partial Matches")
        st.write(partial)

    with c3:
        st.error("Missing Skills")
        st.write(missing)

with tab3:
    st.subheader("Ranked Missing Skills (Critical First)")

    if ranking:
        df_rank = pd.DataFrame(
            ranking, columns=["Skill", "Gap Score"]
        ).sort_values("Gap Score", ascending=False)

        st.dataframe(df_rank, use_container_width=True)

        st.download_button(
            "Download Skill Gap Report",
            df_rank.to_csv(index=False),
            file_name="skill_gap_report.csv",
            mime="text/csv"
        )
    else:
        st.success("No critical skill gaps found ")
