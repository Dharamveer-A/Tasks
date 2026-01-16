import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="Unified Skill Gap Analysis Dashboard",
    page_icon="🌸",
    layout="wide"
)

st.markdown("""
<style>
.pastel-card {
    background-color: #f3efff;
    padding: 1.5rem;
    border-radius: 14px;
    margin-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1>🌸 Unified Skill Gap Analysis Dashboard</h1>
<p style="color:#6b7280;">
Single-page, scrollable skill gap analysis with basic + semantic insights.
</p>
""", unsafe_allow_html=True)

# ===================================================
# Upload Section
# ===================================================
st.markdown('<div class="pastel-card">', unsafe_allow_html=True)
st.subheader("📂 Upload Resume & Job Description")

resume_file = st.file_uploader("Upload Resume (.txt)", type=["txt"])
jd_file = st.file_uploader("Upload Job Description (.txt)", type=["txt"])

def read_file(file):
    if file:
        return file.read().decode("utf-8").lower()
    return None

resume_text = read_file(resume_file)
jd_text = read_file(jd_file)

if not resume_text or not jd_text:
    st.info("Please upload both files to continue.")
    st.stop()

st.success("Files uploaded successfully")
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<div class="pastel-card">', unsafe_allow_html=True)
st.subheader("🔍 Document Preview")

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Resume Preview**")
    st.text(resume_text[:400])
with c2:
    st.markdown("**Job Description Preview**")
    st.text(jd_text[:400])

st.markdown('</div>', unsafe_allow_html=True)
BASIC_SKILLS = [
    "python", "sql", "machine learning", "deep learning",
    "statistics", "data analysis", "aws", "communication"
]

matched_basic = [s for s in BASIC_SKILLS if s in resume_text and s in jd_text]
missing_basic = [s for s in BASIC_SKILLS if s not in resume_text and s in jd_text]

match_percent_basic = int((len(matched_basic) / len(BASIC_SKILLS)) * 100)

st.markdown('<div class="pastel-card">', unsafe_allow_html=True)
st.subheader("📈 Basic Skill Match Summary")

st.metric("Match Percentage", f"{match_percent_basic}%")

c1, c2 = st.columns(2)
with c1:
    st.success("Matched Skills")
    st.write(matched_basic)
with c2:
    st.error("Missing Skills")
    st.write(missing_basic)

fig, ax = plt.subplots(figsize=(5, 3))
ax.bar(
    ["Matched", "Missing"],
    [len(matched_basic), len(missing_basic)],
    color=["#c7d2fe", "#fde68a"]
)
ax.set_title("Basic Skill Distribution")
st.pyplot(fig)

st.markdown('</div>', unsafe_allow_html=True)
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

SKILLS = [
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
    "software development lifecycle", "agile", "scrum",
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
]

def extract_skills(text):
    return [s for s in SKILLS if s in text]

resume_skills = extract_skills(resume_text)
jd_skills = extract_skills(jd_text)

resume_emb = model.encode(resume_skills) if resume_skills else np.array([])
jd_emb = model.encode(jd_skills) if jd_skills else np.array([])

similarity = (
    cosine_similarity(jd_emb, resume_emb)
    if len(resume_emb)
    else np.zeros((len(jd_skills), 1))
)

matched, partial, missing, ranking = [], [], [], []

for i, jd_skill in enumerate(jd_skills):
    max_sim = similarity[i].max() if resume_skills else 0
    if max_sim >= 0.8:
        matched.append(jd_skill)
    elif max_sim >= 0.5:
        partial.append(jd_skill)
    else:
        missing.append(jd_skill)
        ranking.append((jd_skill, round(1 - max_sim, 2)))

overall_match = int((len(matched) / len(jd_skills)) * 100) if jd_skills else 0

st.markdown('<div class="pastel-card">', unsafe_allow_html=True)
st.subheader("🧠 Advanced Semantic Skill Insights")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Overall Match", f"{overall_match}%")
k2.metric("Matched", len(matched))
k3.metric("Partial", len(partial))
k4.metric("Missing", len(missing))

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="pastel-card">', unsafe_allow_html=True)
st.subheader("🔥 Skill Similarity Heatmap")

heatmap = go.Figure(data=go.Heatmap(
    z=similarity,
    x=resume_skills,
    y=jd_skills,
    colorscale="Purples"
))
heatmap.update_layout(height=500)
st.plotly_chart(heatmap, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="pastel-card">', unsafe_allow_html=True)
st.subheader("📋 Skill Gap Report")

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

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="pastel-card">', unsafe_allow_html=True)
st.subheader("🚨 Ranked Missing Skills (Critical First)")

if ranking:
    df_rank = pd.DataFrame(
        ranking, columns=["Skill", "Gap Score"]
    ).sort_values("Gap Score", ascending=False)

    st.dataframe(df_rank, use_container_width=True)

    st.download_button(
        "📥 Download Skill Gap Report",
        df_rank.to_csv(index=False),
        file_name="skill_gap_report.csv",
        mime="text/csv"
    )
else:
    st.success("No critical skill gaps found 🎉")

st.markdown('</div>', unsafe_allow_html=True)
