import streamlit as st
from PIL import Image
import torch
import numpy as np
import os

def load_model(model_path):
    """Load the YOLOv5 model from the specified path."""
    return torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

def detect_objects(model, image):
    """Run object detection on the image using the YOLOv5 model."""
    results = model(image)
    return results

def count_objects(results, class_names):
    """Count detected objects by class."""
    counts = {cls: 0 for cls in class_names}
    for pred in results.xyxy[0]:
        class_id = int(pred[-1])  # Class ID is in the last column
        if class_id < len(class_names):
            counts[class_names[class_id]] += 1
    return counts

# Streamlit app
st.title("YOLOv5 Object Detection App")
st.write("Upload an image and detect objects using a custom-trained YOLOv5 model.")

# Model path
model_path = 'best.pt'

default_confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)

if not os.path.exists(model_path):
    st.sidebar.error("Model path does not exist. Please provide a valid path.")
else:
    model = load_model(model_path)
    model.conf = default_confidence  # Set confidence threshold

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

class_names = ["cardboard", "glass", "plastic", "metal", "paper", "trash"]

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")

    # Convert the image to a NumPy array for YOLO
    image_np = np.array(image)

    # Run object detection
    results = detect_objects(model, image_np)

    # Display detection results
    st.write("### Detection Results")
    st.image(results.imgs[0], caption="Detection Output", use_column_width=True)

    # Count detected objects
    counts = count_objects(results, class_names)
    st.write("### Detected Objects Count")
    for cls, count in counts.items():
        st.write(f"{cls.capitalize()}: {count}")

    # Display raw predictions
    st.write("### Raw Predictions")
    st.write(results.pandas().xyxy[0])  # Predictions as a pandas DataFrame

    # Handle case where no objects are detected
    if len(results.xyxy[0]) == 0:
        st.warning("No objects detected. Check your model or input image.")

# Footer
st.write("Developed with YOLOv5 and Streamlit.")
