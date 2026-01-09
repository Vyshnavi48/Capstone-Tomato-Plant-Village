import os
import json
import gdown
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
def download_from_gdrive(file_id, output):
    # Streamlit Cloud friendly Google Drive download
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    gdown.download(url, output, quiet=False, fuzzy=True)


if not os.path.exists(MODEL_PATH):
    st.info("Downloading trained model (first run only)…")
    download_from_gdrive(MODEL_FILE_ID, MODEL_PATH)

if not os.path.exists(CLASSES_PATH):
    st.info("Downloading class labels (first run only)…")
    download_from_gdrive(CLASSES_FILE_ID, CLASSES_PATH)

# ---------------------------------
# LOAD CLASS NAMES
# ---------------------------------
with open(CLASSES_PATH, "r") as f:
    class_names = json.load(f)

num_classes = len(class_names)

# ---------------------------------
# LOAD MODEL
# ---------------------------------
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    model.load_state_dict(state)
    model.eval()
    return model

model = load_model()

# ---------------------------------
# IMAGE TRANSFORM (SAME AS TRAINING)
# ---------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------------------------
# UI
# ---------------------------------
st.title("🍅 Tomato Leaf Disease Detection")
st.write("Upload a tomato leaf image and the AI model will predict the disease.")

uploaded_file = st.file_uploader(
    "Upload an image (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    input_tensor = transform(image).unsqueeze(0)

    # Prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_index = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_index].item()

    st.subheader("Prediction Result")
    st.write(f"**Disease:** {class_names[predicted_index]}")
    st.write(f"**Confidence:** {confidence:.2%}")

    st.subheader("Class Probabilities")
    for i, class_name in enumerate(class_names):
        st.write(f"{class_name}: {probabilities[i].item():.2%}")
    MODEL_FILE_ID = "1s3VFqAusNEwMUUg51samL_olHg_-XgC2"
CLASSES_FILE_ID = "1T4ypwiViYf5Mo7GNTeT9BWxE71J9Kcq9"

