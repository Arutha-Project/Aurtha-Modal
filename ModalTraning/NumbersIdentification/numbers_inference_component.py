from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from number_inference import SignNumberInference
import numpy as np
import tempfile
import shutil

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Adjust for your frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load multiple models into a dictionary
model_paths = {
    "0-10": Path(r"C:\Users\jmovi\PycharmProjects\Aurtha-Modal\ModalTraning\NumbersIdentification\0-10_numbers_model.joblib"),
    "11-20": Path(r"C:\Users\jmovi\PycharmProjects\Aurtha-Modal\ModalTraning\NumbersIdentification\11-20_numbers_model.joblib"),
    "21-30": Path(r"C:\Users\jmovi\PycharmProjects\Aurtha-Modal\ModalTraning\NumbersIdentification\21-30_numbers_model.joblib"),
    "31-40": Path(r"C:\Users\jmovi\PycharmProjects\Aurtha-Modal\ModalTraning\NumbersIdentification\31-40_numbers_model.joblib"),
    "41-50": Path(r"C:\Users\jmovi\PycharmProjects\Aurtha-Modal\ModalTraning\NumbersIdentification\41-50_numbers_model.joblib"),
}

inference_models = {key: SignNumberInference(path) for key, path in model_paths.items()}


@app.get("/", response_class=HTMLResponse)
async def main():
    return """
    <!DOCTYPE html>
    <html>
    <body>
        <h2>Upload MP4 Video for Number Prediction</h2>
        <form action="/predict_numbers/" method="post" enctype="multipart/form-data">
            <label>Select Model Range:</label>
            <select name="model_key">
                <option value="0-10">0-10</option>
                <option value="11-20">11-20</option>
                <option value="21-30">21-30</option>
                <option value="31-40">31-40</option>
                <option value="41-50">41-50</option>
            </select><br><br>
            <input type="file" name="file" accept=".mp4" required>
            <button type="submit">Upload and Predict</button>
        </form>
    </body>
    </html>
    """

@app.post("/validate_number/")
async def validate_number(file: UploadFile = File(...), expected_number: int = Form(...), model_key: str = Form(...)):
    """Validate predicted number using selected model."""
    model = inference_models.get(model_key)

    if model is None:
        return {"error": "Invalid model range selected."}

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
        temp_path = temp.name
        shutil.copyfileobj(file.file, temp)

    try:
        predicted_number = model.predict(temp_path)
        predicted_number = int(predicted_number) if isinstance(predicted_number, np.integer) else predicted_number
        correct = predicted_number == expected_number
        print(f"Predicted: {predicted_number}, Correct: {correct}")
        return {"predicted_number": predicted_number, "correct": correct}
    finally:
        Path(temp_path).unlink(missing_ok=True)
