import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

st.title("Cat vs Dog Image Recognition")
st.write("Upload an image and the model will predict if it's a Cat or Dog.")

# Function to load model safely
@st.cache_resource
def load_my_model():
    model_path = "cat_dog_model_fixed.keras"
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None
    try:
        model = load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load model
model = load_my_model()

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    if model is not None:
        # Preprocess image
        img_resized = img.resize((224, 224))  # Match your model input size
        x = image.img_to_array(img_resized)
        x = np.expand_dims(x, axis=0) / 255.0
        
        # Predict
        pred = model.predict(x)
        if pred[0][0] > 0.5:
            st.success("Prediction: Dog 🐶")
        else:
            st.success("Prediction: Cat 🐱")
    else:
        st.warning("Cannot predict because the model failed to load.")
