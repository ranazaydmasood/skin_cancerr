import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Configure page style and title
st.set_page_config(
    page_title="Skin Lesion Classifier",
    page_icon="🔬",
    layout="centered"
)

st.title("🔬 Skin Lesion Classifier")
st.write(
    "Upload a clinical image of a skin lesion to determine if it is classified as **Benign** or **Malignant**."
)

# Cache model loading to avoid reloading on every user interaction
@st.cache_resource
def load_keras_model():
    return tf.keras.models.load_model('best_skin_cancer_model.keras')

try:
    model = load_keras_model()
    st.success("Model weights loaded successfully!")
except Exception as e:
    st.error(f"Error loading the model. Ensure 'best_skin_cancer_model.keras' is present in the directory. Error: {e}")

# Upload image interface
uploaded_file = st.file_uploader("Choose a skin image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    st.write("Processing and analyzing image...")
    
    # Preprocess image to match training specification
    # Target size: (224, 224), scaled pixels: [0, 1]
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalized scale
    
    # Run model prediction
    prediction_prob = model.predict(img_array)[0][0]
    
    # Determine classes and formatting based on 0.5 threshold
    if prediction_prob >= 0.5:
        label = "Malignant"
        confidence = prediction_prob * 100
        color = "red"
    else:
        label = "Benign"
        confidence = (1 - prediction_prob) * 100
        color = "green"
        
    # Render the results
    st.markdown(f"### Prediction: :{color}[{label}]")
    st.markdown(f"### Confidence: **{confidence:.2f}%**")
    
    # Warning/Disclaimers
    st.warning("Disclaimer: This model is for educational purposes only and does not replace medical professional diagnostics.")
