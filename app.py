import streamlit as st
import numpy as np
import nibabel as nib
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.transform import resize
from tensorflow.keras.models import load_model
import tempfile
import os

def load_nifti_image(file):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp_file:
        tmp_file.write(file.read())  # Write uploaded file content to temp file
        tmp_file_path = tmp_file.name  # Get file path

    # Load the NIfTI image using nibabel
    img = nib.load(tmp_file_path)
    img_data = img.get_fdata()
    
    # Remove the temporary file
    os.remove(tmp_file_path)

    return img_data

def preprocess_image(img_data):
    slices = []
    for i in range(img_data.shape[2]):
        slices.append(img_data[:, :, i])
    slices = np.array(slices)
    slices = (slices - np.min(slices)) / (np.max(slices) - np.min(slices))
    slices = slices[:, :, :, np.newaxis]  # Add channel dimension
    return slices

@st.cache_resource
def load_trained_model():
    return load_model("liver_tumor_model.h5", compile=False)

def predict_mask(model, image):
    prediction = model.predict(image)
    return prediction

st.title("Liver Tumor Prediction from NIfTI Images")

uploaded_file = st.file_uploader("Upload a NIfTI (.nii.gz) file", type=["nii", "nii.gz"])

if uploaded_file is not None:
    with st.spinner("Processing the file..."):
        img_data = load_nifti_image(uploaded_file)
        processed_image = preprocess_image(img_data)
        model = load_trained_model()
        predicted_mask = predict_mask(model, processed_image)

    st.write("## Predicted Tumor Segmentation Mask")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    slice_idx = processed_image.shape[0] // 2  # Middle slice
    
    axes[0].imshow(processed_image[slice_idx, :, :, 0], cmap='gray')
    axes[0].set_title("Original Image")
    
    axes[1].imshow(predicted_mask[slice_idx, :, :, 0], cmap='viridis')
    axes[1].set_title("Predicted Mask")
    
    axes[2].imshow(processed_image[slice_idx, :, :, 0], cmap='gray')
    axes[2].imshow(predicted_mask[slice_idx, :, :, 0], cmap='viridis', alpha=0.5)
    axes[2].set_title("Overlay")
    
    st.pyplot(fig)
