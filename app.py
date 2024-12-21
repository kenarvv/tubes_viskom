import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os

def load_model(model_path):
    """Load the YOLO model from the specified path with CPU mode."""
    model = YOLO(model_path).to('cpu')  # Memaksa penggunaan CPU
    return model

def detect_objects(model, image):
    """Run object detection on the image using the model."""
    results = model(image)
    return results

def count_objects(results, class_names):
    """Count detected objects by class."""
    counts = {cls: 0 for cls in class_names}
    for pred in results[0].boxes:
        class_id = int(pred.cls[0])
        if class_id < len(class_names):
            counts[class_names[class_id]] += 1
    return counts

# Streamlit app
st.title("YOLO Object Detection App")
st.write("Upload an image and detect objects using a custom-trained YOLO model.")

# Model path is now in the same directory as the app
model_path = 'best.pt'

default_confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.001)

if not os.path.exists(model_path):
    st.sidebar.error("Model path does not exist. Please provide a valid path.")
else:
    model = load_model(model_path)
    model.overrides["conf"] = default_confidence
    st.sidebar.write(f"Using device: {model.device}")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

class_names = ["cardboard", "glass", "plastic", "metal", "paper", "trash"]

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")

    # Object detection
    results = detect_objects(model, image)

    # Display results
    st.write("### Detection Results")
    detected_image = results[0].plot
    # Tampilkan gambar hasil deteksi
    st.image(detected_image, caption="Detection Output", use_column_width=True)

    # Hitung objek yang terdeteksi
    counts = count_objects(results, class_names)
    st.write("### Detected Objects Count")
    for cls, count in counts.items():
        st.write(f"{cls.capitalize()}: {count}")

    # Tampilkan prediksi mentah
    st.write("### Raw Predictions")
    st.write(results[0].boxes.xyxy.cpu().numpy())  # Prediksi dalam format [x1, y1, x2, y2, conf, class_id]

    # Tangani kasus tanpa deteksi objek
    if len(results[0].boxes) == 0:
        st.warning("No objects detected. Check your model or input image.")

# Footer
st.write("Developed with YOLOv5 and Streamlit.")
