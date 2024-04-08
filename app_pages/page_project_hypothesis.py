import streamlit as st


def page_project_hypothesis_body():
    st.header("Project Hypothesis")

    st.write("The project had one main hypothesis that was worked toward:")

    st.info("Powdery mildew infected leaves have visual signs that "
            "differentiate them from healthy leaves. An ML model can be "
            "created to visually differentiate healthy from infected "
            "leaves at a rate of greater than 97% accuracy, "
            "as per the project requirements.")

    st.write("Exploratory data analysis was performed on the image "
             "dataset and two ML models were created to differentiate "
             "between healthy and infected leaves. \n\n"
             "The required accuracy was 97%, but both models - "
             "created with different mathematical logic - significantly "
             "outperformed the project requirement. The slightly "
             "more accurate model was then used in the final project to "
             "be submitted to the client. \n\nA final accuracy rate of "
             "99.76% accuracy was achieved, proving the hypothesis.\n\n "
             "This accuracy rate was reached despite significantly "
             "resizing the images down due to deployment restrictions. "
             "We can further hypothesise that near-perfect prediction "
             "using an ML model could be achieved with the extra "
             "information available in larger images. ")
