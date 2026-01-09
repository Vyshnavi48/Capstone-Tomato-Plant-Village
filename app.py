import os
import json
import requests
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(
    page_title="Tomato Disease Detection",
    page_icon="🍅",
    layout="centered"
)

# ---------------------------------
# FILE CONFIG (UPDATE FILE IDs)
# ---------------------------------
MODEL_PATH = "tomato_model_resnet18_best.pth"
CLASSES_PATH = "tomato_class_names.json"

# 🔴 YOU MUST REPLACE THESE WITH YOUR OWN GOOGLE DRIVE FILE IDs
MODEL_FILE_ID = "PASTE_MODEL_FILE_ID_HERE"
CLASSES_FILE_ID = "PASTE_CLASSES_FILE_ID_HERE"

# ---------------------------------
# DOWNLOAD FROM GOOGLE DRIVE
# ---------------------------------
def download_from_drive(file_id, output_path):
    """
    Robust Google Drive downloader for Streamlit Cloud.
    Works even when Drive shows confirmation/virus warning pages.
    """
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    # First request
    response = session.get(URL, params={"id": file_id}, stream=True)
    token = None

    # Check if Google asks for confirmation token
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break

    # Second request with token (if needed)
    if token:
        response = session.get(URL, params={"id": file_id, "confirm": token}, stream=True)

    # Save file
    response.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)

MODEL_FILE_ID = "1s3VFqAusNEwMUUg51samL_olHg_-XgC2"
CLASSES_FILE_ID = "1T4ypwiViYf5Mo7GNTeT9BWxE71J9Kcq9"



