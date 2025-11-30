import gradio as gr
import os
import gdown
import tensorflow as tf  # assuming your model is Keras .h5
from tensorflow.keras.models import load_model

# ----- Step 1: Download the model -----
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

# ----- Step 3: Define prediction function -----
def predict(image):  # assuming your model takes an image
    # Preprocess the image as you did in your training script
    image = image.resize((224, 224))  # example, adjust as needed
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)  # batch dimension

    predictions = model.predict(img_array)
    class_idx = predictions.argmax(axis=1)[0]

    # Map class_idx to your class names
    class_names = ["Apple Scab", "Apple Black Rot", "Healthy", ...]  # replace with your classes
    return class_names[class_idx]

# ----- Step 4: Setup Gradio interface -----
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),   # take PIL image
    outputs=gr.Textbox(label="Predicted Disease"),
    live=False
)

iface.launch(server_name="0.0.0.0", server_port=7860)
