import streamlit as st
import cv2
from PIL import Image
from ultralytics import YOLO
import tempfile
import time

st.set_page_config(page_title="YOLOv8 Live Detection", layout="centered")
st.title("🎥 YOLOv8 Live Webcam Detection")

# Load YOLO model
model_path = "best.pt"  # Update if your model is in a different location
model = YOLO(model_path)

# Start webcam
run = st.checkbox('Start Webcam Detection')

FRAME_WINDOW = st.image([])

if run:
    cap = cv2.VideoCapture(0)  # 0 is the default webcam

    # Check if webcam is opened
    if not cap.isOpened():
        st.error("Webcam not accessible")
    else:
        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to grab frame")
                break

            # Save frame temporarily
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
                cv2.imwrite(temp.name, frame)

                # Run YOLOv8 detection
                results = model.predict(source=temp.name, conf=0.25)
                annotated_frame = results[0].plot()

                # Convert BGR to RGB for Streamlit display
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(annotated_frame_rgb)

            time.sleep(0.05)  # small delay for responsiveness

        cap.release()
else:
    st.warning("Enable the checkbox to start webcam detection.")
