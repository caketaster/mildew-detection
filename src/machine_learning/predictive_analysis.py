import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from tensorflow.keras.models import load_model
from PIL import Image
from src.data_management import load_pkl_file


def plot_predictions_probabilities(pred_probability, pred_class):
    # Initialize DataFrame for probability results
    prob_per_class = pd.DataFrame(
        {'Probability': [1 - pred_probability, pred_probability]},
        index=['Healthy', 'Mildew infected']
    )

    # Plot using Plotly
    fig = px.bar(
        prob_per_class,
        x=prob_per_class.index,
        y='Probability',
        range_y=[0, 1],
        width=600, height=400, template='plotly_white'
    )

    st.plotly_chart(fig)


def resize_input_image(img, version):

    # Resize and reshape image to av image size
    image_shape = load_pkl_file(file_path=f"outputs/{version}/image_shape.pkl")  # noqa
    img_resized = img.resize((image_shape[1], image_shape[0]), Image.LANCZOS)  # noqa
    my_image = np.expand_dims(img_resized, axis=0)/255

    return my_image


def load_model_and_predict(my_image, version):

    # Prediction on dragged and dropped images
    model = load_model(f"outputs/{version}/mildew_detection_model_2.h5")
    pred_probability = model.predict(my_image)[0, 0]
    target_map = {v: k for k, v in {'Healthy': 0, 'Mildew infected': 1}.items()}  # noqa

    pred_class = target_map[pred_probability > 0.5]
    if pred_class == target_map[0]:
        pred_probability = 1 - pred_probability

    st.write("The ML model predicts the sample is labelled "
             f"**{pred_class}** at a probability of "
             f"**{round(pred_probability, 2)}**")

    return pred_probability, pred_class
