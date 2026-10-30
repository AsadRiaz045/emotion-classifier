import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Configure Streamlit page
st.set_page_config(
    page_title="Facial Emotion Classifier",
    page_icon="🎭",
    layout="centered"
)

# Title and Description
st.title("🎭 Facial Emotion Classifier")
st.write("Upload an image, and our fine-tuned MobileNetV2 model will predict the underlying emotion.")

# Set up class names mapping from training
class_names = ['angry', 'happy', 'sad', 'surprised']

# Load the trained model
@st.cache_resource
def load_emotion_model():
    model_path = 'emotion_model.keras'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(
            model_path, 
            custom_objects={'preprocess_input': preprocess_input}
        )
    else:
        st.warning("Model file 'emotion_model.keras' not found. Please save your model first using `transfer_model.save('emotion_model.keras')`.")
        return None

model = load_emotion_model()

# Image Uploader (Camera input removed)
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Target Image', use_container_width=True)
    
    if model is not None:
        with st.spinner('Analyzing emotion...'):
            # 1. Resize to 64x64 (Model ki input shape ke mutabiq)
            resized_image = image.resize((64, 64))
            img_array = tf.keras.utils.img_to_array(resized_image)
            
            # 2. Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            # 3. Perform inference (Preprocessing ab model ke andar khud hogi)
            predictions = model.predict(img_array)
            raw_scores = predictions[0]
            
            best_class_idx = np.argmax(raw_scores)
            best_class = class_names[best_class_idx]
            confidence = raw_scores[best_class_idx] * 100
            
            # Display Result
            st.success(f"Predicted Emotion: **{best_class.upper()}** ({confidence:.2f}% Confidence)")
            
            # Detailed confidence breakdown
            st.write("### Probability Breakdown")
            for emotion, score in zip(class_names, raw_scores):
                st.write(f"{emotion.capitalize()}")
                st.progress(float(score))
                st.caption(f"Confidence: {score*100:.2f}%")