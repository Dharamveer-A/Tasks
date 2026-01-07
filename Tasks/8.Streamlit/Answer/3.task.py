# 3. Create a file uploader that accepts only PDF, DOCX, and TXT files.

import streamlit as st

st.title("File uploader example")

file = st.file_uploader("Upload", type=['pdf', 'txt', 'docx'])