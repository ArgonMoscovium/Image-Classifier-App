import streamlit as st
import tensorflow as tf
from PIL import Image # For processing user ip images
import numpy as np
import os

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/trained_fashion_mnist_model.h5"
# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Define class labels for Fashion MNIST dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# Function to preprocess the uploaded image
def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((28, 28))
    img = img.convert('L')  # Convert any image (even rgb) to grayscale
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape((1, 28, 28, 1)) # prediction for just one image (one is batch no.)
    return img_array

# Streamlit App
st.title('Fashion Item Classifier')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2) # Showing two columns Layout in the app

    with col1:
        resized_img = image.resize((100, 100)) # Display enlarged user ip image
        st.image(resized_img) # ... and displays it

    with col2:
        if st.button('Classify'): # Classification starts on pressing of button
            # Preprocess the uploaded image
            img_array = preprocess_image(uploaded_image)

            # Make a prediction using the pre-trained model
            result = model.predict(img_array)
            # st.write(str(result))
            predicted_class = np.argmax(result)
            prediction = class_names[predicted_class]

            st.success(f'Prediction: {prediction}')