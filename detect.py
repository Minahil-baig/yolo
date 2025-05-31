import streamlit as st
import cv2
from PIL import Image
from ultralytics import YOLO
import tempfile
import os

st.set_page_config(page_title="YOLOv8 Live Webcam Detection", layout="centered")
st.title("📸 YOLOv8 Live Webcam Detection App")

# Load model
model_path = "best.pt"  # Replace with your path if different
model = YOLO(model_path)

# Use webcam input
picture = st.camera_input("Take a picture")

if picture:
    # Save image from webcam input
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(picture.getbuffer())
        image_path = temp_file.name

    # Run prediction
    with st.spinner("Running detection..."):
        results = model.predict(source=image_path, save=True, conf=0.25)
        result_path = results[0].save_dir / os.path.basename(image_path)
        st.image(str(result_path), caption="Detected Image", use_column_width=True)
