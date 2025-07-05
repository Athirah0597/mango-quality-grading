import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# ------------------------
# Load model from Google Drive
# ------------------------
MODEL_PATH = "mango_cnn_model.h5"
FILE_ID = "1nwGqYn5v26ij5tbg4LbtKHdZSVQ2bxdN"  # Replace with your own if different
DOWNLOAD_URL = f"https://drive.google.com/uc?id={FILE_ID}"

if not os.path.exists(MODEL_PATH):
    st.info("Downloading model...")
    gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False)

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)
class_names = ['Grade A', 'Grade B', 'Grade C']

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="Mango Grading", page_icon="🥭", layout="centered")
st.title("🥭 Mango Quality Grading App")
st.markdown("Upload an image of a mango to predict its quality grade.")

# Upload image
uploaded_file = st.file_uploader("Choose a mango image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess
    img = image.resize((180, 180))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_label = class_names[predicted_index]
    confidence = prediction[0][predicted_index] * 100

    # Output
    st.success(f"**Predicted Grade: {predicted_label}**")
    st.info(f"Confidence: {confidence:.2f}%")
