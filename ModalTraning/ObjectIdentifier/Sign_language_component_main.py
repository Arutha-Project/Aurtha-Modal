from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from inference import SignLanguageInference
import tempfile
import shutil

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Allow your React app
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers (needed for file uploads)
)

shapes = ["circle", "rectangle", "square", "triangle"]
animals = ["elephant", "cat1", "cat2", "dog1", "dog2", "parrot", "butterfly"]
fruits = ["apple", "banana1", "banana2", "mango", "pineapple"]

model_paths = {
    "shapes": Path(r"C:\Users\WW\Desktop\SF-25\Final Reseach Data\Aurtha-Modal\ModalTraning\ObjectIdentifier\shapes_sign_language_model.joblib"),
    "animals": Path(r"C:\Users\WW\Desktop\SF-25\Final Reseach Data\Aurtha-Modal\ModalTraning\ObjectIdentifier\animals_sign_language_model.joblib"),
    "fruits": Path(r"C:\Users\WW\Desktop\SF-25\Final Reseach Data\Aurtha-Modal\ModalTraning\ObjectIdentifier\fruits_sign_language_model.joblib"),
}

def get_category(name: str):
    if name in shapes:
        return "shapes"
    elif name in animals:
        return "animals"
    elif name in fruits:
        return "fruits"
    else:
        return None

# Initialize the inference class

@app.post("/predict/")
async def predict(file: UploadFile = File(...),random_name: str = Form(...)):
    """Handle the uploaded MP4 file and return a prediction."""

    category = get_category(random_name.lower())
    if not category:
        return {"error": f"Unknown category for {random_name}"}
    
    # Load the model for this category
    model_path = model_paths[category]
    inference = SignLanguageInference(model_path)

    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp_path = temp.name
        shutil.copyfileobj(file.file, temp)

    try:
        # Perform prediction
        prediction = inference.predict(temp_path)
        print(f"[{category.upper()}] Predicted: {prediction}")
        return {"predicted_sign": prediction}
    finally:
        # Clean up the temporary file
        Path(temp_path).unlink(missing_ok=True)
        
        