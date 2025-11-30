import gradio as gr
import os
import gdown
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import json

# ----- Step 1: Download model if not exists -----
FILE_ID = "1DWY7nBMUiVttXNz0kdL83QyQKhGn1DTv"
url = f"https://drive.google.com/uc?id={FILE_ID}&confirm=t"
MODEL_FILE = "plant_disease_model.h5"

if not os.path.exists(MODEL_FILE):
    print("Downloading model...")
    gdown.download(url, MODEL_FILE, quiet=False)
    print("Download complete.")
else:
    print("Model already exists.")

# ----- Step 2: Load the model -----
model = load_model(MODEL_FILE)
print("Model loaded successfully.")

# ----- Step 3: Load label map -----
if os.path.exists("label_map.json"):
    with open("label_map.json", "r") as f:
        label_map = json.load(f)
    class_names = [label_map[str(i)] for i in range(len(label_map))]
else:
    # fallback if you don't have label_map.json
    class_names = [
        "Tomato Early Blight",
        "Potato Late Blight",
        "Pepper Bacterial Spot",
        "Healthy"
        # add all your classes here
    ]

# ----- Step 4: Disease solutions mapping -----
disease_solutions = {
    "Tomato Early Blight": "Remove infected leaves. Apply a copper-based fungicide weekly.",
    "Potato Late Blight": "Destroy infected plants. Use Mancozeb or Chlorothalonil spray.",
    "Pepper Bacterial Spot": "Use certified seeds. Apply streptomycin spray weekly.",
    "Healthy": "Your plant appears healthy!"
    # add other diseases and solutions as needed
}

# ----- Step 5: Preprocess function -----
def preprocess_image(image):
    img = image.convert("RGB")
    img = img.resize((224, 224))  # match your model input
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# ----- Step 6: Prediction function -----
def predict(image):
    img = preprocess_image(image)
    preds = model.predict(img)
    index = np.argmax(preds[0])
    confidence = float(np.max(preds[0]))
    label = class_names[index]
    solution = disease_solutions.get(label, "No solution available. Please consult an expert.")
    
    return f"{label} ({confidence*100:.2f}% confidence)\nSolution: {solution}"

# ----- Step 7: Gradio Interface -----
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="Prediction & Solution"),
    title="Plant Disease Detector",
    description="Upload an image of a leaf and get the disease prediction and solution."
)

iface.launch(server_name="0.0.0.0", server_port=7860)
