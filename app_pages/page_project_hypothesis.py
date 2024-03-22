import streamlit as st


def page_project_hypothesis_body():
    st.header("Project Hypothesis")

    st.subheader("The project had one main hypothesis that was worked toward")

    st.write("Powdery mildew infected leaves have visual signs that differentiate "
            "them from healthy leaves. An ML model can be created to visually "
            "differentiate healthy from infected leaves at a rate of greater than 97% accuracy.")

    st.info("Exploratory data analysis was performed on the image dataset. "
            "An ML model was created that successfully differentiates healthy "
            "and infected leaves at an accuracy rate of __%.")