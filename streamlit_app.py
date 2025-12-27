import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

st.title("Cat vs Dog Image Recognition")
st.write("Upload an image and the model will predict if it's a Cat or Dog.")

# -------------------------------
# Load model safely
# -------------------------------
@st.cache_data
def load_my_model():
    model_path = "cat_dog_model_fixed.keras"  # Make sure this file is uploaded
    if not os.path.exists(model_path):
        st.warning(f"Model file '{model_path}' not found!")
        return None
    try:
        model = load_model(model_path, compile=False)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_my_model()

# -------------------------------
# Image upload and prediction
# -------------------------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file and model is not None:
    img = Image.open(uploaded_file).resize((224, 224))
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0

    # Predict
    pred = model.predict(x)
    if pred[0][0] > 0.5:
        st.success("Prediction: Dog 🐶")
    else:
        st.success("Prediction: Cat 🐱")
elif uploaded_file and model is None:
    st.warning("Cannot make predictions because the model failed to load.")
