import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import imread
from src.machine_learning.evaluate_clf import load_evaluate_pkl

def page_ml_performance_body():
    version = 'v1'
    
    # n.b. page titles (check other pages)
    st.header("ML Performance")

    st.subheader("Label Distribution Between the Sets")

    st.info(
            "The provided dataset was split into train, validation and test sets with a standard 70/10/20 split.\n\n"
            "The labels are 'healthy' and 'powdery_mildew'.\n\n" 
            "Image split:\n\n"
            "* Train - healthy: 1472 images, powdery_mildew: 1472 images\n\n"
            "* Validation - healthy: 210 images, powdery_mildew: 210 images\n\n"
            "* Test - healthy: 422 images, powdery_mildew: 422 images\n\n"
            )
    
    if st.checkbox("Bar Chart"):
        labels_distribution = plt.imread(f"outputs/{version}/label_distrib_pre_augmentation.png")
        st.image(labels_distribution, caption="Bar chart for label distribution: train, validation "
                                              "and test sets.")
    if st.checkbox("Pie Chart"):
        labels_distribution = plt.imread(f"outputs/{version}/image_distribution_pie.png")
        st.image(labels_distribution, caption="Pie chart for label distribution: train, validation "
                                              "and test sets.")
    
    st.header("Model History")

    st.info(
        f"The graphs show the learning cycle for the ML model. "
        f"The first graph shows the accuracy plot, the second shows the loss plot."
        )

    acc_plot = plt.imread(f"outputs/{version}/model_training_acc.png")
    st.image(acc_plot, caption="Plot showing how accuracy performed during model training")
    
    loss_plot = plt.imread(f"outputs/{version}/model_training_losses.png")
    st.image(loss_plot, caption="Plot showing how loss performed during model training")

    st.success("The plots show good, 'normal' performance. "
               "Both plots show an increase towards accuracy with no evidence of "
               "over or underfitting.")

    st.header("Generalised Performance on Test Set")

    st.dataframe(pd.DataFrame(load_evaluate_pkl(version), index=["Loss", "Accuracy"]))

    st.write(
        "The test set was fresh, unseen data for the model and thus "
        "could not be affected by the model 'remembering' the correct label. "
    )

    st.success("The scores on the test set closely align with the scores reached on the validation set. "
              "Thus there is no evidence of under or overfitting.\n\n "
              "The accuracy score is __, far better than the required score of 97% accuracy.")