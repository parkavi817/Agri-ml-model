from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import json

app = Flask(__name__)
CORS(app)

# Load model and label map
model = load_model('plant_disease_model.h5')

with open("label_map.json", "r") as f:
    label_map = json.load(f)
class_names = [label_map[str(i)] for i in range(len(label_map))]

# Disease to solution mapping
disease_solutions = {
    "Tomato___Early_blight": "Remove infected leaves. Apply a copper-based fungicide weekly.",
    "Potato___Late_blight": "Destroy infected plants. Use Mancozeb or Chlorothalonil spray.",
    "Pepper__bell___Bacterial_spot": "Use certified seeds. Apply streptomycin spray weekly.",
    "Healthy": "Your plant appears healthy!",
    # Add other diseases and solutions as needed
}

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    img_bytes = file.read()
    img = preprocess_image(img_bytes)
    
    preds = model.predict(img)
    index = np.argmax(preds[0])
    confidence = float(np.max(preds[0]))
    raw_label = class_names[index]
    pretty_label = raw_label.replace("_", " ").replace("  ", " ").strip()

    solution = disease_solutions.get(raw_label, "No solution available. Please consult an expert.")
    
    return jsonify({
        'result': pretty_label,
        'confidence': confidence,
        'solution': solution
    })

if __name__ == '__main__':
    app.run(port=5001)
