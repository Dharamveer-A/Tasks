# 2. Add a Streamlit sidebar and display navigation text inside it.

import streamlit as st

st.title("My Awesome App")

with st.sidebar:
    selected_option = st.selectbox("Choose an Option", ["Resume", "Job Description"])

if selected_option == 'Resume' :
    st.write("Now you are in Resume option")
    st.write(f"Selected Option: {selected_option}")
elif selected_option == 'Job Description' :
    st.write("Now you are in JD option")
    st.write(f"Selected Option: {selected_option}")