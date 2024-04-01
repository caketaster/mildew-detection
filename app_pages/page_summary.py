import streamlit as st


def page_summary_body():
    st.header("Project Summary\n\n")

    st.write("Powdery mildew refers to a fungal infection that can impact various plants, "
              "including crops and trees. Powdery mildew infected plants can be diagnosed by "
              "noting the white or grayish powdery growth that appears on their leaves or stems, "
              "made up of fungal spores and mycelium. \n\n"
              "Powdery mildew infected crops cannot be sold on the market due to the infection. \n\n"
              "Our client (Farmy & Foods) has a powdery mildew problem with their "
              "cherry trees. Current manual methods of diagnosis take far too long "
              "to be scalable and effective, and so a data analyst was employed "
              "to help expedite the diagnosis process.\n\n "
              "A dataset of 4200 images is provided to the data analyst to use in creating and testing "
              "a machine learning [ML] model for use in powdery mildew diagnosis.\n\n"
              "The client requires a 97% accuracy rate in correct diagnosis for their cherry trees "
              "(i.e. healthy or mildew infected) through analysis of leaf images. \n\n"
              "The client further requests a dashboard so that the results can be easily visualised. \n\n"
              "The client also asks for data privacy to be observed, in that the data from the project "
              "not be shared with anyone not involved with the project.\n\n")

    st.subheader("Business Case")

    st.info("The project business requirements are:\n\n"
            "1. Create an ML model that correctly diagnoses powdery mildew or lack thereof in "
            "cherry leaves with a greater than 97% accuracy\n\n"
            "2. Create a dashboard to display the results in a simple and clear way\n\n"
            "3. Data is provided under an NDA, and only those involved in the project can "
            "have access to it")

    st.write("Successful completion of the project will speed up the client's testing process "
            "and allow them to reduce the amount of mildew-infected product they produce.\n\n")

    st.subheader("Dataset")

    st.write("The client provided a dataset of 4200 images of cherry leaves, pre-labelled "
                "*healthy* or *powdery_mildew* for the developer to use to create the model. \n\n"
                "The dataset can be found [here](https://www.kaggle.com/codeinstitute/cherry-leaves)")

