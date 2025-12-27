import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------
# Load Model with Cache
# -----------------------
@st.cache_resource
def load_my_model():
    model = tf.keras.models.load_model("cat_dog_model_fixed (1).keras", compile=False)
    return model

model = load_my_model()

# -----------------------
# Streamlit UI
# -----------------------
st.title("🐱 vs 🐶 Cat vs Dog Image Recognition")
st.write("Upload an image and the model will predict if it's a Cat or Dog.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    img = Image.open(uploaded_file).convert("RGB").resize((224, 224))
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    x = np.array(img)/255.0               # Normalize pixel values
    x = np.expand_dims(x, axis=0)         # Add batch dimension

    # Predict
    pred = model.predict(x)
    confidence = float(pred[0][0])

    if confidence > 0.5:
        st.success(f"Prediction: Dog 🐶 ({confidence*100:.2f}% confidence)")
    else:
        st.success(f"Prediction: Cat 🐱 ({(1-confidence)*100:.2f}% confidence)")

# Optional: Footer
st.markdown("---")
st.write("Model trained on Cats vs Dogs dataset. Resize images to 224x224 for best results.")


