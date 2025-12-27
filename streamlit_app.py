import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Cat vs Dog Image Recognition")

# Safe model loading
@st.cache_data
def load_my_model():
    try:
        model = tf.keras.models.load_model("cat_dog_model.keras", compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_my_model()

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file and model:
    img = Image.open(uploaded_file).convert("RGB")
    
    # Get model input shape
    input_shape = model.input_shape[1:3]  # (height, width)
    img = img.resize(input_shape)
    
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    x = np.array(img)/255.0
    x = np.expand_dims(x, axis=0)

    # Predict
    pred = model.predict(x)
    if pred[0][0] > 0.5:
        st.success("Prediction: Dog 🐶")
    else:
        st.success("Prediction: Cat 🐱")
elif uploaded_file:
    st.warning("Cannot make predictions because the model failed to load.")

