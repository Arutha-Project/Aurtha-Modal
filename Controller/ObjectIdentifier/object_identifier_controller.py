from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from Services.ObjectIdentifier.inference import SignLanguageInference
import tempfile
import shutil

object_identifier_app = FastAPI()

# Initialize the inference class
model_path = r"ModalTraning\ObjectIdentifier\rf_sign_language_model.joblib"
inference = SignLanguageInference(model_path)


@object_identifier_app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Handle the uploaded MP4 file and return a prediction."""
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp_path = temp.name
        shutil.copyfileobj(file.file, temp)

    try:
        # Perform prediction
        prediction = inference.predict(temp_path)
        return {"predicted_sign": prediction}
    finally:
        # Clean up the temporary file
        Path(temp_path).unlink(missing_ok=True)

