import streamlit as st
from PIL import Image
import torch
import os

# Fungsi untuk memuat model
@st.cache_resource
def load_model(model_path):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    model.conf = 0.1  # Set confidence threshold
    return model

# Fungsi untuk memprediksi gambar
def predict_image(model, image):
    results = model(image)
    return results

# Header aplikasi
st.title("Trash Classification App")
st.markdown(
    """
    Upload an image to classify the type of trash in it using the YOLOv5 model.
    Supported classes: **cardboard, glass, metal, paper, plastic, trash**.
    """
)

# Pastikan model berada di direktori yang sama
model_path = "best.pt"
if not os.path.exists(model_path):
    st.error("Model file not found! Please place 'best.pt' in the same directory as this app.")
else:
    # Muat model
    model = load_model(model_path)

    # Unggah gambar
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Menampilkan gambar yang diunggah
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Classifying...")

        # Prediksi menggunakan model
        results = predict_image(model, image)

        # Tampilkan hasil
        st.markdown("### Results")
        if len(results.xyxy[0]) == 0:
            st.write("No objects detected. Try another image.")
        else:
            for det in results.xyxy[0]:
                label = model.names[int(det[5])]
                confidence = det[4] * 100
                st.write(f"- **{label}** with confidence {confidence:.2f}%")

        # Menampilkan hasil visualisasi
        st.image(results.render()[0], caption="Detection Results", use_column_width=True)
