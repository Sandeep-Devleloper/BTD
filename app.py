import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import imutils

# Load your trained model
newmodel = load_model('BTD_model')  # Replace with your model path

# Preprocessing functions (crop_imgs and preprocess_imgs) remain the same
def getpred(img):
    img = cv2.resize(img, (224, 224))
    X = []
    X.append(img)
    Xc = crop_imgs(X)
    Xcp = preprocess_imgs(Xc, (224, 224))
    predictions = newmodel.predict(Xcp)
    probability = predictions[0][0]
    predicted_class = int(probability > 0.5)

    message = 'Prediction: {} \nConfidence: {:.2f}%'.format(
        'Tumor Detected' if predicted_class == 1 else 'No Tumor Detected',
        probability * 100
    )

    return message

# Streamlit UI
st.title("Brain Tumor Detection")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Convert to OpenCV format and make prediction
    image = np.array(image)  
    image = image[:, :, ::-1].copy()  # Convert RGB to BGR 
    result = getpred(image)

    # Display the prediction
    st.write(result)
