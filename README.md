# Lung Cancer Prediction App (Ensemble CNN + GAN)

This Streamlit web app allows users to upload lung scan images and predicts the presence of lung cancer using an ensemble of Convolutional Neural Networks (CNNs). A GAN is also included for synthetic image generation (not used in inference).

## Features
- Upload X-ray or CT lung scan images
- Ensemble prediction using 3 CNN models
- Simple UI powered by Streamlit

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run lung_cancer_streamlit_app.py
