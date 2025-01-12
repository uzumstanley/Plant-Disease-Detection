import streamlit as st
import tensorflow as tf
import numpy as np
import gdown  # To download files from Google Drive

# Function to download the model from Google Drive
def download_model_from_drive():
    # Replace this with your actual file ID from Google Drive
    file_id = "1-2lmdge9ZkfYTpojRzgmlpzxuEL6_urO"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "trained_plant_disease_model.keras"
    gdown.download(url, output, quiet=False)
    return output

# TensorFlow Model Prediction
def model_prediction(test_image):
    model_file = download_model_from_drive()  # Download model from Drive
    model = tf.keras.models.load_model(model_file)  # Load the model
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍

    ...

    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!
    """)

# About Project
elif app_mode == "About":
    st.header("About")
    st.markdown("""
                #### About Dataset
                ...

                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)
                """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if st.button("Show Image"):
        st.image(test_image, width=4, use_column_width=True)
    # Predict button
    if st.button("Predict"):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        # Reading Labels
        class_name = [
            'Apple___Apple_scab', 'Apple___Black_rot', ...
        ]
        st.success(f"Model is Predicting it's a {class_name[result_index]}")
