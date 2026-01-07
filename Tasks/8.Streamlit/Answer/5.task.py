# 5. Show the first 300 characters of uploaded text inside the Streamlit app.

import streamlit as st
import pdfplumber
from docx import Document

def extract_text(uploaded_file):
    text = ""

    # TXT
    if uploaded_file.name.endswith(".txt"):
        text = uploaded_file.read().decode("utf-8")

    # PDF
    elif uploaded_file.name.endswith(".pdf"):
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""

    # DOCX
    elif uploaded_file.name.endswith(".docx"):
        doc = Document(uploaded_file)
        for para in doc.paragraphs:
            text += para.text + "\n"

    return text.strip()


st.title("File Name Example")

file = st.file_uploader(
    "Upload file (PDF / DOCX / TXT)",
    type=["pdf", "docx", "txt"]
)

if file is not None:
    content = extract_text(file)

    file_name, file_extension = file.name.rsplit(".", 1)

    st.write(f"**Your file name:** {file_name}")
    st.write(f"**Your file extension:** {file_extension}")

    st.text_area(
        label="File content",
        value=content,
        height=400
    )
