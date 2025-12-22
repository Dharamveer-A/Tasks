import streamlit as st
from docx import Document
import os
from re import sub

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


def preview(content, length=200):
    """
    This will return a preview of default length of 200 characters
    args : content -> string
           length -> integer
    return-type : string
    """
    return content[:length]


def cleanText(content):
    """
    This function will clean the content
    args : content -> string
    return-type : string
    """
    clean = content.lower()
    clean = sub(r"[^a-zA-Z0-9/:\s]", "", clean)
    clean = sub(r"\s+", " ", clean)
    return clean.strip()


def saveFile(content="", fileName="output"):
    """
    This will save the content in a text file
    args : fileName -> string
           content -> string
    return-type : void
    """
    os.makedirs("files", exist_ok=True)
    path = os.path.join("files", fileName + ".txt")
    with open(path, "w", encoding="utf-8") as file:
        file.write(content)
    return path


# Streamlit Application

st.title("Job Description Cleaner")

st.write(
    "Upload a JD `.docx` file, clean the content, "
    "and preview the processed output."
)

uploaded_file = st.file_uploader(
    "Drag and drop JD file here",
    type=["docx"]
)

if uploaded_file:

    st.success("File uploaded successfully!")

    rawContent = loadDocx(uploaded_file)

    st.subheader("---------- Preview for Raw Content ----------")
    st.write(preview(rawContent))

    if st.button("Clean & Process"):

        cleanContent = cleanText(rawContent)

        st.subheader("---------- Preview for Cleaned Content ----------")
        st.write(preview(cleanContent))

        outputPath = saveFile(cleanContent, fileName="output1")

        st.success("Output saved successfully")

        st.download_button(
            label="Download Cleaned Output",
            data=cleanContent,
            file_name="cleaned_jd.txt",
            mime="text/plain"
        )
