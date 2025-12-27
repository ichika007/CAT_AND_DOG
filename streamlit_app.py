import streamlit as st
import tensorflow as tf

@st.cache_data
def load_my_model():
    try:
        model = tf.keras.models.load_model("cat_dog_model.keras", compile=False)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_my_model()
if model:
    st.success("Model loaded successfully!")
