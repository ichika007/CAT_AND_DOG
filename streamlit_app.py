import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

@st.cache_resource
def load_my_model():
 return tf.keras.models.load_model(
    "cat_dog_model_fixed.keras",
    compile=False
)

model = load_my_model()

st.title("Cat vs Dog Image Recognition")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))
    st.image(img, caption="Uploaded Image")

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        st.success("Dog 🐶")
    else:
        st.success("Cat 🐱")

