import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json
from PIL import Image
import os

# Load the trained model
model_path = 'oral_disease_model.keras'
if not os.path.isfile(model_path):
    st.error(f"Model file {model_path} not found.")
else:
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading the model: {e}")

    # Load class labels and descriptions
    class_labels_path = 'class_labels.json'
    if not os.path.isfile(class_labels_path):
        st.error(f"Class labels file {class_labels_path} not found.")
    else:
        try:
            with open(class_labels_path, 'r') as f:
                class_labels = json.load(f)
        except json.JSONDecodeError:
            st.error("Error decoding the class labels JSON file.")
        except Exception as e:
            st.error(f"An error occurred while loading the class labels: {e}")
        
        # Function to preprocess the uploaded image
        def preprocess_image(img):
            img = img.resize((150, 150))  # Ensure the size matches model input
            img_array = np.array(img)
            img_array = img_array / 255.0  # Normalize
            img_array = np.expand_dims(img_array, axis=0)
            return img_array

        # Streamlit app
        col1, col2 = st.columns([1, 4])
    
        with col1:
            st.image("dental.jpg", width=150)  # Adjust the width if needed

        with col2:
            st.title("Oral Disease Diagnosis")

        st.write('Upload an image of an oral cavity to diagnose the disease.')

        # Updated to accept both 'jpg' and 'jpeg' files
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg"])

        if uploaded_file is not None:
            try:
                # Load and display image
                img = Image.open(uploaded_file)
                st.image(img, caption='Uploaded Image.', use_column_width=True)
                
                # Preprocess image
                img_array = preprocess_image(img)
                
                # Predict
                predictions = model.predict(img_array)
                predicted_class_index = np.argmax(predictions[0])
                prediction_confidence = predictions[0][predicted_class_index]
                
                # Set a confidence threshold
                confidence_threshold = 0.5
                
                if prediction_confidence < confidence_threshold:
                    st.write("The model is not confident about this prediction. The image might not match any known class or the model may require further training.")
                    st.write("If the issue persists, please ensure that the image is clear and relevant to the dataset.")
                else:
                    # Ensure class_labels is a dictionary
                    if isinstance(class_labels, dict):
                        class_info = class_labels.get(str(predicted_class_index), {})
                        
                        # Ensure class_info is a dictionary
                        if isinstance(class_info, dict):
                            predicted_class_name = class_info.get("name", "Unknown")
                            description = class_info.get("description", "No description available.")
                            suggested_treatment = class_info.get("suggested_treatment", "No treatment information available.")
                            
                            # Display the result with bold text and increased font size, and formatted output
                            st.markdown(f'<p style="font-size:20px; font-weight:bold;">Predicted diagnosis:</p> <p style="font-size:20px; margin-top: -10px;">{predicted_class_name}</p>', unsafe_allow_html=True)
                            st.markdown(f'<p style="font-size:18px; font-weight:bold;">Explanation:</p> <p style="font-size:18px; margin-top: -10px;">{description}</p>', unsafe_allow_html=True)
                            st.markdown(f'<p style="font-size:18px; font-weight:bold;">Suggested treatment:</p> <p style="font-size:18px; margin-top: -10px;">{suggested_treatment}</p>', unsafe_allow_html=True)
                        else:
                            st.error(f"Expected class_info to be a dictionary, but got {type(class_info)}")
                            st.write("No description or treatment information available.")
                    else:
                        st.error("Class labels data is not in the expected format.")
                        st.write("No description or treatment information available.")
            except Exception as e:
                st.error(f"An error occurred while processing the image: {e}")
        else:
            st.write("Please upload an image to proceed.")

