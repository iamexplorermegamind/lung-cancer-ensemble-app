
import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Load models
model1 = load_model("ensemble_cnn_1.h5")
model2 = load_model("ensemble_cnn_2.h5")
model3 = load_model("ensemble_cnn_3.h5")

def preprocess_image(img):
    img = img.convert("L")
    img = img.resize((128, 128))
    img = np.array(img) / 255.0
    img = img.reshape(1, 128, 128, 1)
    return img

def ensemble_predict(image):
    preds = [m.predict(image)[0] for m in [model1, model2, model3]]
    avg_pred = np.mean(preds, axis=0)
    return np.argmax(avg_pred), avg_pred

st.title("🫁 Lung Cancer Prediction - Ensemble CNNs")
st.write("Upload a lung scan image to predict the presence of lung cancer.")

uploaded_file = st.file_uploader("Choose a lung scan image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Predicting..."):
        processed_image = preprocess_image(image)
        pred_class, prob = ensemble_predict(processed_image)

        st.subheader("Prediction Result")
        st.write("Class:", "Cancer" if pred_class == 1 else "Normal")
        st.write("Confidence:", f"{prob[pred_class]*100:.2f}%")
