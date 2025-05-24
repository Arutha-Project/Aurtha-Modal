from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from ModalTraning.ObjectIdentifier.inference import SignLanguageInference
import tempfile
import shutil

app = FastAPI()

# locate script, project root, and models folder
script_dir   = Path(__file__).resolve().parent
project_root = script_dir.parent
models_dir   = project_root / "ObjectIdentifier"

shapes = ["circle", "rectangle", "square", "triangle"]
animals = ["elephant", "cat1", "cat2", "dog1", "dog2", "parrot", "butterfly"]
fruits = ["apple", "banana1", "banana2", "mango", "pineapple"]

model_paths = {
    "shapes": Path("shapes_sign_language_model.joblib"),
    "animals": Path("animals_sign_language_model.joblib"),
    "fruits": Path("fruits_sign_language_model.joblib"),
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
inference_models = {}
for key, fname in model_paths.items():
    model_file = models_dir / fname
    if not model_file.is_file():
        raise FileNotFoundError(f"[{key}] model not found: {model_file}")
    inference_models[key] = SignLanguageInference(model_file)

@app.post("/predict/")
async def predict(file: UploadFile = File(...),random_name: str = Form(...)):
    """Handle the uploaded MP4 file and return a prediction."""

    category = get_category(random_name.lower())
    if not category:
        return {"error": f"Unknown category for {random_name}"}

    # Correct inference model selection
    inference = inference_models.get(category)

    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
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
        
        