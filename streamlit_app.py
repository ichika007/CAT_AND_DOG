import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page config (must be at the top)
st.set_page_config(page_title="Cat vs Dog Classifier")

# Cache the model so it loads only once
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model("cat_dog_model.h5", compile=False)


model = load_my_model()

st.title("Cat vs Dog Image Recognition")
st.write("Upload an image and the model will predict if it's a Cat or Dog.")

uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB").resize((224, 224))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        st.success("Prediction: Dog 🐶")
    else:
        st.success("Prediction: Cat 🐱")
