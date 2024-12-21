import streamlit as st
from PIL import Image
import torch
import os

def load_model(model_path):
    """Load the YOLO model from the specified path."""
    return torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

def detect_objects(model, image):
    """Run object detection on the image using the model."""
    results = model(image)
    return results

def count_objects(results, class_names):
    """Count detected objects by class."""
    counts = {cls: 0 for cls in class_names}
    for pred in results.xyxy[0]:
        class_id = int(pred[-1])
        if class_id < len(class_names):
            counts[class_names[class_id]] += 1
    return counts

# Streamlit app
st.title("YOLO Object Detection App")
st.write("Upload an image and detect objects using a custom-trained YOLO model.")

# Sidebar for model path
# model_path = st.sidebar.text_input("Model Path", "yolov5/runs/train/yolo_trash_quick4/weights/best.pt")
model_path = 'best.pt'

default_confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.001)

if not os.path.exists(model_path):
    st.sidebar.error("Model path does not exist. Please provide a valid path.")
else:
    model = load_model(model_path)
    model.conf = default_confidence

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

class_names = ["cardboard", "glass", "plastic", "metal", "paper", "trash"]

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")

    # Object detection
    results = detect_objects(model, image)
    results.print()  # Print results to console for debugging

    # Display results
    st.write("### Detection Results")
    st.image(results.imgs[0], caption="Detection Output", use_column_width=True)

    # Count detected objects
    counts = count_objects(results, class_names)
    st.write("### Detected Objects Count")
    for cls, count in counts.items():
        st.write(f"{cls.capitalize()}: {count}")

    # Display raw predictions
    st.write("### Raw Predictions")
    st.write(results.xyxy[0].numpy())

    # Handle no detection case
    if len(results.xyxy[0]) == 0:
        st.warning("No objects detected. Check your model or input image.")

# Section for displaying metrics screenshot
st.write("### Model Metrics")
st.write("Upload a screenshot of your model's metrics (accuracy, precision, recall, F1-score):")
metrics_file = st.file_uploader("Upload Metrics Screenshot", type=["jpg", "jpeg", "png"])
if metrics_file is not None:
    metrics_image = Image.open(metrics_file)
    st.image(metrics_image, caption="Model Metrics", use_column_width=True)

# Footer
st.write("Developed with YOLOv5 and Streamlit.")
