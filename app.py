import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.models import load_model

# Load your trained model
newmodel = load_model('BTD_model')  # Replace with your model path

# Preprocessing function
def getpred(img):
    # Resize the image
    img = cv2.resize(img, (224, 224))
    # Add a batch dimension
    img = np.expand_dims(img, axis=0)
    # Predict the result using the model
    predictions = newmodel.predict(img)
    probability = predictions[0][0]
    predicted_class = int(probability > 0.5)

    # Return the message with prediction and confidence
    message = 'Prediction: {} \nConfidence: {:.2f}%'.format(
        'Tumor Detected' if predicted_class == 1 else 'No Tumor Detected',
        probability * 100
    )
    return message

# Streamlit UI
st.title("Brain Tumor Detection")

# File uploader for the image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Convert PIL image to OpenCV format (BGR)
    image = np.array(image)  
    image = image[:, :, ::-1]  # Convert RGB to BGR for OpenCV

    # Get prediction
    result = getpred(image)

    # Display the prediction
    st.write(result)
