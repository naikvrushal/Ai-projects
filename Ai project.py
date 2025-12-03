from transformers import pipeline
from PIL import Image
import requests
from io import BytesIO

# Initialize the image classification pipeline (cached after first load)
classifier = None

def get_classifier():
    global classifier
    if classifier is None:
        print("Loading AI model...")
        # Using a lightweight model that works well on Streamlit Cloud
        classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
    return classifier

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

    print("Classifying...")
    clf = get_classifier()
    predictions = clf(img)
    
    # Convert to format similar to TensorFlow output: (id, label, score)
    # For compatibility with the Streamlit app
    results = [(f"class_{i}", pred['label'], pred['score']) for i, pred in enumerate(predictions[:3])]
    
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
