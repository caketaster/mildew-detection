import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd

from src.data_management import download_dataframe_as_csv
from src.machine_learning.predictive_analysis import (
                                                    load_model_and_predict,
                                                    resize_input_image,
                                                    plot_predictions_probabilities  # noqa
                                                    )


def page_mildew_detector_body():
    st.write("Using the app below, cherry leaf images can be uploaded "
             "to check the ML model. ")
    st.write("The label probability between zero and one will be "
             "given to 2 decimal points, with a higher value "
             "meaning higher confidence in the prediction. "
             "You can hover over the bar chart to find the "
             "more precise value.")
    st.write("**Please upload only .jpg or .jpeg images**" )
    st.info("An image set of healthy and mildew infected leaves can be "
            "downloaded for live prediction. "
            "Images can be downloaded from "
            "[here](https://www.kaggle.com/codeinstitute/cherry-leaves)."
            )

    st.write("---")

    upload_img = st.file_uploader("Upload leaf image(s). You may "
                                  "upload more than one at a time.",
                        type=['jpg', 'jpeg'], accept_multiple_files=True)  # noqa

    if upload_img is not None:
        analysis_report = pd.DataFrame([])
        for image in upload_img:

            img_pil = Image.open(image)
            st.info(f"Leaf image: **{image.name}**")
            img_array = np.array(img_pil)
            st.image(img_pil, caption=f"Image Size: {img_array.shape[1]}px width x {img_array.shape[0]}px height")  # noqa

            version = 'v1'
            resized_img = resize_input_image(img=img_pil, version=version)
            pred_probability, pred_class = load_model_and_predict(resized_img, version=version)  # noqa
            plot_predictions_probabilities(pred_probability, pred_class)

            analysis_report = analysis_report.append({'Image': image.name, 'Result': pred_class, 'Probability': pred_probability},  # noqa
                                ignore_index=True)

        if not analysis_report.empty:
            st.success("Analysis Report")
            st.table(analysis_report)
            st.markdown(download_dataframe_as_csv(analysis_report), unsafe_allow_html=True)  # noqa
            