import streamlit as st
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread

import itertools
import random


def page_leaf_visualiser_body():
    st.write("### Cherry Leaf Visualiser")
    st.info(
        "The client (Farmy & Foods) requires a model that can visually "
        "differentiate between healthy and powdery mildew "
        "infected cherry leaves. A montage for either label allows us to "
        "visually check both categories. The difference and variability "
        "images may show patterns within each category"
        )

    version = 'v1'

    # Creates a montage of sample images from either label
    if st.checkbox("Image Montage"):
        st.write("* To refresh the montage, click on the "
                 "'Create Montage' button")
        my_data_dir = 'inputs/cherry_leaves_raw_dataset/cherry-leaves'
        labels = os.listdir(my_data_dir + '/train')
        label_to_display = st.selectbox(label="Select label", options=labels, index=0)  # noqa
        if st.button("Create Montage"):
            image_montage(dir_path=my_data_dir + '/train',
                          label_to_display=label_to_display,
                          nrows=5, ncols=3, figsize=(10, 25))

        st.warning("Intuitively, fairly clear differences can be seen at "
                   "a glance between healthy and mildew infected leaves, "
                   "which gives confidence that an ML model will be able "
                   "to tell them apart too")

        st.write("---")

    if st.checkbox("Difference between average and variability image"):

        avg_parasitized = plt.imread(f"outputs/{version}/avg_diff_healthy.png")
        avg_uninfected = plt.imread(f"outputs/{version}/avg_diff_powdery_mildew.png")  # noqa

        st.image(avg_parasitized, caption="Healthy leaf - Average "
                                          "and Variability")
        st.image(avg_uninfected, caption="Mildew infected leaf - "
                                         "Average and Variability")

        st.warning("The variability images show that there are clear "
                   "features that differentiate healthy and "
                   "mildew infected leaves. Healthy leaves appear to "
                   "be more regular in shape")

        st.write("---")
        st.info("**Explanation of terms**\n\n"
                "The average here refers to the *Mean* - the sum of "
                "all images divided by the number of images.\n\n")


# Function to create the montage
def image_montage(dir_path, label_to_display, nrows, ncols, figsize=(10, 8)):
    sns.set_style("white")
    labels = os.listdir(dir_path)

    # Subset the class to display
    if label_to_display in labels:

      # Checks if the montage space is greater than subset size
      images_list = os.listdir(dir_path + '/' + label_to_display)
      if nrows * ncols < len(images_list):
          img_idx = random.sample(images_list, nrows * ncols)
      else:
          print(
              f"Decrease nrows or ncols to create your montage. \n"
              f"There are {len(images_list)} in your subset. "
              f"You requested a montage with {nrows * ncols} spaces")
          return

      # Create list of axes indices based on nrows and ncols
      list_rows = range(0, nrows)
      list_cols = range(0, ncols)
      plot_idx = list(itertools.product(list_rows, list_cols))

      # Create figure and display images
      fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
      for x in range(0, nrows*ncols):
          img = imread(dir_path + '/' + label_to_display + '/' + img_idx[x])
          img_shape = img.shape
          axes[plot_idx[x][0], plot_idx[x][1]].imshow(img)
          axes[plot_idx[x][0], plot_idx[x][1]].set_title(f"Width {img_shape[1]}px x Height {img_shape[0]}px")  # noqa
          axes[plot_idx[x][0], plot_idx[x][1]].set_xticks([])
          axes[plot_idx[x][0], plot_idx[x][1]].set_yticks([])
      plt.tight_layout()

      st.pyplot(fig=fig)
      plt.show()

    else:
        print("The label you selected doesn't exist.")
        print(f"The options are: {labels}")
