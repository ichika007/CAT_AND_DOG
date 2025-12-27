import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

st.set_page_config(page_title="Cat vs Dog Classifier", layout="centered")
st.title("🐱🐶 Cat vs Dog Image Recognition")
st.write("Upload an image, and the model will predict whether it's a Cat or Dog.")

# Function to safely load model
@st.cache_resource
def load_model_safe(model_path):
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except ValueError as e:
        st.error(f"Failed to load model: {e}")
        return None
    except OSError as e:
        st.error(f"Model file not found or corrupted: {e}")
        return None

# Load your model
model = load_model_safe("cat_dog_model_fixed.keras")  # Make sure this file exists

if model is None:
    st.stop()  # Stop the app if model cannot be loaded

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display image
    img = Image.open(uploaded_file).convert('RGB').resize((224, 224))
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0

    # Predict
    try:
        pred = model.predict(x)
        if pred[0][0] > 0.5:
            st.success("Prediction: Dog 🐶")
        else:
            st.success("Prediction: Cat 🐱")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
