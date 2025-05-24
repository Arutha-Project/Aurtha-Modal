import os
import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO
from Constant.QuickDraw.quick_draw_constant import quick_draw_class_names

# Model Path
MODEL_PATH = "BackendTesting/QuickDraw/quickDraw.keras"

# Load Model
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
else:
    print(f"Model file not found: {MODEL_PATH}")
    model = None

# Class Names

def process_image(image: Image.Image):
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to model input size
    image = np.array(image)

    # Ensure white strokes on black background
    if np.mean(image) > 127:  # If background is white, invert
        image = 255 - image

    image = image / 255.0  # Normalize
    image = image.reshape((1, 28, 28, 1))  # Reshape for model input
    return image


def predict_drawing(image):
    """Predict the drawing using the model."""
    pred = model.predict(image)[0]
    top_index = np.argmax(pred)
    return quick_draw_class_names[top_index], pred[top_index]

def validate_prediction(file: BytesIO, selected_object: str):
    """Processes an image and checks if the prediction matches the selected object."""
    if model is None:
        return {"error": "Model not loaded."}

    # Read and process the image
    image = Image.open(file)
    image = process_image(image)

    # Make prediction
    predicted_class, confidence = predict_drawing(image)

    # Check correctness
    is_correct = predicted_class.lower() == selected_object.lower()

    return {
        "predicted_class": predicted_class,
        # "confidence": round(float(confidence), 4),
        "selected_object": selected_object,
        "is_correct": is_correct
    }
