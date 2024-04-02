import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import imread
from src.machine_learning.evaluate_clf import load_evaluate_pkl

def page_ml_performance_body():
    version = 'v1'
    
    st.header("ML Performance")

    st.subheader("Label Distribution Between the Sets\n\n")

    st.info("The provided dataset was split into train, validation and test sets with a standard 70:10:20 split.\n\n"
            "The leaf labels are *healthy* and *powdery_mildew*.\n\n" 
            "Image split:\n\n"
            "* Train - healthy: 1472 images, powdery_mildew: 1472 images\n\n"
            "* Validation - healthy: 210 images, powdery_mildew: 210 images\n\n"
            "* Test - healthy: 422 images, powdery_mildew: 422 images\n\n")
    
    labels_distribution = plt.imread(f"outputs/{version}/label_distrib_pre_augmentation.png")
    st.image(labels_distribution, caption="Bar chart for label distribution: train, validation "
                                              "and test sets.")

    labels_distribution = plt.imread(f"outputs/{version}/image_distribution_pie.png")
    st.image(labels_distribution, caption="Pie chart for label distribution: train, validation "
                                            "and test sets.")
    
    st.header("Model History")

    st.info("The graphs show the learning cycle for the ML model. "
            "The first graph shows the accuracy plot, the second shows the loss plot.")

    acc_plot = plt.imread(f"outputs/{version}/model_training_acc_sigmoid.png")
    st.image(acc_plot, caption="Plot showing how accuracy performed during model training")
    
    loss_plot = plt.imread(f"outputs/{version}/model_training_losses_sigmoid.png")
    st.image(loss_plot, caption="Plot showing how loss performed during model training")

    st.info("*Accuracy* here refers to the amount of correct deductions by the ML model "
            "(with 1.0 being 'perfect).\n\n "
            "*Loss* represents the discrepancy between the predicted " "values and the actual values in the training data. A lower " 
            "loss value indicates that the model's predictions are closer "
            "to the actual values, implying better performance\n\n"
            "The blue lines refer to the Training set data, the orange lines to the "
            "Validation set data \n\n"
            "The model improved accuracy and loss metrics during the fitting process, "
            "reaching an excellent level of accuracy with high confidence")


    st.success("The plots show good, 'normal' performance. "
               "Both plots show an increase towards accuracy with no evidence of "
               "over or underfitting")

    st.header("Generalised Performance on Test Set")

    st.dataframe(pd.DataFrame(load_evaluate_pkl(version), index=["Loss", "Accuracy"]))

    st.write(
        "The test set was fresh, unseen data for the model and thus "
        "could not be affected by the model 'remembering' the correct label. "
    )

    st.success("The scores on the test set closely align with the scores reached on the validation set. "
              "Thus there is no evidence of under or overfitting.\n\n "
              "The accuracy score is 99.76%, far better than the required score of 97% accuracy, "
              "and the loss metric of under 0.5% shows high confidence in predictions")