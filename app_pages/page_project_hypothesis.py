import streamlit as st


def page_project_hypothesis_body():
    st.header("Project Hypothesis")

    st.write("The project had one main hypothesis that was worked toward")

    st.info("Powdery mildew infected leaves have visual signs that differentiate "
            "them from healthy leaves. An ML model can be created to visually "
            "differentiate healthy from infected leaves at a rate of greater than 97% accuracy.")

    st.write("Exploratory data analysis was performed on the image dataset and "
            "an ML model was created to differentiates between healthy "
            "and infected leaves.")