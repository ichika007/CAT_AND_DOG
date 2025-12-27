import streamlit as st
import tensorflow as tf

st.title("Cat vs Dog App")

try:
    model = tf.keras.models.load_model("cat_dog_model_fixed.keras", compile=False)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {e}")
