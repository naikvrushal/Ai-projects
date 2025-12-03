import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import requests
from io import BytesIO
from PIL import Image

def classify_image(img_input):
    # Handle different input types
    if isinstance(img_input, str):
        print(f"Loading image from: {img_input}...")
        if img_input.startswith('http'):
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
            response = requests.get(img_input, headers=headers)
            img = Image.open(BytesIO(response.content))
        else:
            clean_path = img_input.strip('"').strip("'")
            img = Image.open(clean_path)
    else:
        # Assume it's a PIL Image or file-like object from Streamlit
        img = img_input if isinstance(img_input, Image.Image) else Image.open(img_input)

    # Resize for the model
    img_resized = img.resize((224, 224))
    
    # Convert to array and preprocess
    x = image.img_to_array(img_resized)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    print("Loading model...")
    model = MobileNetV2(weights='imagenet')

    print("Classifying...")
    preds = model.predict(x)
    
    # Decode results
    results = decode_predictions(preds, top=3)[0]
    return results, img

if __name__ == "__main__":
    # Example image of a Golden Retriever
    IMAGE_URL = r"E:\volume f\download.jpeg"
    # You can change this URL to any image you want to test!
    
    try:
        results, _ = classify_image(IMAGE_URL)
        print("\n--- Results ---")
        for i, (imagenet_id, label, score) in enumerate(results):
            print(f"{i+1}: {label} ({score*100:.2f}%)")
    except Exception as e:
        print(f"An error occurred: {e}")
