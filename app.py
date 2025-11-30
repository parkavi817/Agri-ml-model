import gradio as gr
import os
import gdown
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import json

# ----- STEP 1: DOWNLOAD MODEL -----
FILE_ID = "1DWY7nBMUiVttXNz0kdL83QyQKhGn1DTv"
url = f"https://drive.google.com/uc?id={FILE_ID}&confirm=t"
MODEL_FILE = "plant_disease_model.h5"

if not os.path.exists(MODEL_FILE):
    print("Downloading model...")
    gdown.download(url, MODEL_FILE, quiet=False)
    print("Model downloaded.")
else:
    print("Model already exists.")

# ----- STEP 2: LOAD MODEL -----
model = load_model(MODEL_FILE)
print("Model loaded successfully.")

# ----- STEP 3: LABEL MAP -----
label_map = {
    "0": "Pepper__bell___Bacterial_spot",
    "1": "Pepper__bell___healthy",
    "2": "Potato___Early_blight",
    "3": "Potato___Late_blight",
    "4": "Potato___healthy",
    "5": "Tomato_Bacterial_spot",
    "6": "Tomato_Early_blight",
    "7": "Tomato_Late_blight",
    "8": "Tomato_Leaf_Mold",
    "9": "Tomato_Septoria_leaf_spot",
    "10": "Tomato_Spider_mites_Two_spotted_spider_mite",
    "11": "Tomato__Target_Spot",
    "12": "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "13": "Tomato__Tomato_mosaic_virus",
    "14": "Tomato_healthy"
}

class_names = [label_map[str(i)] for i in range(len(label_map))]

# ----- STEP 4: FRIENDLY NAMES -----
pretty_names = {
    "Pepper__bell___Bacterial_spot": "Pepper Bell – Bacterial Spot",
    "Pepper__bell___healthy": "Pepper Bell – Healthy",
    "Potato___Early_blight": "Potato – Early Blight",
    "Potato___Late_blight": "Potato – Late Blight",
    "Potato___healthy": "Potato – Healthy",
    "Tomato_Bacterial_spot": "Tomato – Bacterial Spot",
    "Tomato_Early_blight": "Tomato – Early Blight",
    "Tomato_Late_blight": "Tomato – Late Blight",
    "Tomato_Leaf_Mold": "Tomato – Leaf Mold",
    "Tomato_Septoria_leaf_spot": "Tomato – Septoria Leaf Spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Tomato – Two-Spotted Spider Mite",
    "Tomato__Target_Spot": "Tomato – Target Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Tomato – Yellow Leaf Curl Virus",
    "Tomato__Tomato_mosaic_virus": "Tomato – Mosaic Virus",
    "Tomato_healthy": "Tomato – Healthy"
}

# ----- STEP 5: SOLUTIONS -----
disease_solutions = {
    "Pepper__bell___Bacterial_spot": 
        "Remove infected leaves, use copper spray weekly, avoid overhead watering.",
    "Pepper__bell___healthy": 
        "Plant appears healthy. Maintain good watering and sunlight.",
    
    "Potato___Early_blight": 
        "Use Mancozeb/Chlorothalonil, remove infected leaves, avoid moisture.",
    "Potato___Late_blight": 
        "Destroy infected plants. Spray with copper-based fungicides.",
    "Potato___healthy": 
        "Plant is healthy. Keep soil well-drained and fertilized.",
    
    "Tomato_Bacterial_spot": 
        "Remove infected leaves. Apply copper fungicide every 7 days.",
    "Tomato_Early_blight": 
        "Prune lower leaves, apply Chlorothalonil/Mancozeb weekly.",
    "Tomato_Late_blight": 
        "Remove infected plants immediately. Use preventive copper sprays.",
    "Tomato_Leaf_Mold": 
        "Improve ventilation, avoid overhead watering, apply fungicide.",
    "Tomato_Septoria_leaf_spot": 
        "Remove lower leaves, use Chlorothalonil spray weekly.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": 
        "Spray neem oil every 3 days. Increase humidity.",
    "Tomato__Target_Spot": 
        "Use preventative fungicides and remove affected leaves.",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": 
        "Caused by whiteflies. Use insecticidal soap, remove affected leaves.",
    "Tomato__Tomato_mosaic_virus": 
        "No cure. Remove infected plants. Wash tools thoroughly.",
    "Tomato_healthy": 
        "Tomato plant looks healthy. Continue regular care."
}

# ----- STEP 6: IMAGE PREPROCESS -----
def preprocess_image(image):
    img = image.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# ----- STEP 7: PREDICT -----
def predict(image):
    img = preprocess_image(image)
    preds = model.predict(img)
    index = np.argmax(preds[0])
    confidence = float(np.max(preds[0]))

    label = class_names[index]
    pretty = pretty_names.get(label, label)
    solution = disease_solutions.get(label, "No solution available.")

    return f"""
🔍 **Prediction:** {pretty}
📊 **Confidence:** {confidence*100:.2f}%
🌱 **Solution:** {solution}
"""

# ----- STEP 8: GRADIO APP -----
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Markdown(),
    title="🌿 Plant Disease Detector",
    description="Upload a leaf image to detect plant diseases and get treatment solutions."
)

iface.launch(server_name="0.0.0.0", server_port=7860)
