# 4. Display the uploaded file name on the Streamlit UI.

import streamlit as st
import os

st.title("File name example")

file = st.file_uploader("Upload", type=['pdf', 'docx', 'txt'])

file_name_with_extenstion = file.name.split('.')
st.write(f"You file name is {file_name_with_extenstion[0]}\n")
st.write(f"Your file extenstion is {file_name_with_extenstion[1]}")