import streamlit as st
from src.data_management import load_pkl_file


def load_evaluate_pkl(version):
    """ 
    Calls load_pkl_file to load the saved evaluate pickle file
    """
    return load_pkl_file(file_path=f"outputs/{version}/evaluation_sigmoid.pkl")