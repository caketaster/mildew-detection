import streamlit as st
from app_pages.multipage import MultiPage

# Page scripts
from app_pages.page_summary import page_summary_body
from app_pages.page_leaf_visualiser import page_leaf_visualiser_body
from app_pages.page_mildew_detector import page_mildew_detector_body
from app_pages.page_project_hypothesis import page_project_hypothesis_body
from app_pages.page_ml_performance import page_ml_performance_body

# Create an instance of the app
app = MultiPage(app_name="Cherry Leaf Mildew Detection")

# App pages
app.add_page("Project Summary", page_summary_body)
app.add_page("Project Hypothesis", page_project_hypothesis_body)
app.add_page("Leaf Visualiser", page_leaf_visualiser_body)
app.add_page("Mildew Detection", page_mildew_detector_body)
app.add_page("ML Performance Metrics", page_ml_performance_body)

app.run()