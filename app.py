import streamlit as st
from PIL import Image
import importlib.util
import sys
import os

# Import the backend script
# We do this because the filename has a space in it, which makes standard import hard
spec = importlib.util.spec_from_file_location("ai_project", "Ai project.py")
ai_project = importlib.util.module_from_spec(spec)
sys.modules["ai_project"] = ai_project
spec.loader.exec_module(ai_project)

st.set_page_config(page_title="AI Image Classifier", page_icon="📸")

st.title("📸 AI Image Classifier")
st.write("Upload an image and the AI will tell you what it is!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    st.write("")
    st.write("🤖 **Classifying...**")
    
    # Call the backend function
    # We pass the uploaded file object directly
    try:
        with st.spinner('Analyzing image...'):
            results, _ = ai_project.classify_image(uploaded_file)
        
        st.success("Done!")
        
        st.subheader("Top Predictions:")
        
        # Display results nicely
        for i, (imagenet_id, label, score) in enumerate(results):
            # Create a progress bar for the confidence score
            st.write(f"**{i+1}. {label.replace('_', ' ').title()}** ({score*100:.2f}%)")
            st.progress(float(score))
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
