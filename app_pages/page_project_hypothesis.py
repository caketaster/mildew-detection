import streamlit as st


def page_project_hypothesis_body():
    st.header("Project Hypothesis")

    st.write("The project had one main hypothesis that was worked toward")

    st.info("Powdery mildew infected leaves have visual signs that differentiate "
            "them from healthy leaves. An ML model can be created to visually "
            "differentiate healthy from infected leaves at a rate of greater than 97% accuracy, as per the project requirements.")
            # need more info here

    st.write("Exploratory data analysis was performed on the image dataset and "
            "two ML models were created to differentiate between healthy "
            "and infected leaves. The required accuracy was 97%, "
            "but both models - created with different mathematical logic - "
            "significantly outperformed the project requirement. The slightly "
            "more accurate model was then used in the final project to be "
            "submitted to the client.")
            # or maybe just here