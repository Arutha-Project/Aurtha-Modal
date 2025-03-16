from pathlib import Path
from fastapi import FastAPI, UploadFile, File , Form
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

# Ensure the model path exists
model_path = Path(r"C:\Users\jmovi\PycharmProjects\Aurtha-Modal\ModalTraning\NumbersIdentification\rf_numbers_model.joblib")
inference = SignNumberInference(model_path)


@app.get("/", response_class=HTMLResponse)
async def main():
    """Serve a simple HTML form for uploading files."""
    return """
    <!DOCTYPE html>
    <html>
    <body>
        <h2>Upload MP4 Video for Number Prediction</h2>
        <form action="/predict/" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept=".mp4" required>
            <button type="submit">Upload and Predict</button>
        </form>
    </body>
    </html>
    """


@app.post("/predict_numbers/")
async def predict(file: UploadFile = File(...)):
    """Handle the uploaded MP4 file and return a number prediction."""
    import numpy as np  # Ensure NumPy is imported

    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
        temp_path = temp.name
        shutil.copyfileobj(file.file, temp)

    try:
        # Perform prediction using the uploaded video
        prediction = inference.predict(temp_path)

        # Convert numpy.int64 to Python int
        predicted_number = int(prediction) if isinstance(prediction, np.integer) else prediction

        print("Predicted Number:", predicted_number)  # Log the prediction
        return {"predicted_number": predicted_number}
    finally:
        # Clean up the temporary file
        Path(temp_path).unlink(missing_ok=True)


# Calculation
@app.post("/predict_answer/")
async def predict_answer(file: UploadFile = File(...), correct_answer: int = Form(...)):
    """Handle uploaded video, predict the number, and compare with the correct answer."""

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
        temp_path = temp.name
        shutil.copyfileobj(file.file, temp)

    try:
        # Perform prediction
        predicted_number = inference.predict(temp_path)

        # Convert numpy int64 to Python int
        predicted_number = int(predicted_number) if isinstance(predicted_number, np.integer) else predicted_number

        # Compare predicted answer with correct answer
        is_correct = predicted_number == correct_answer

        print(f"Predicted: {predicted_number}, Correct: {correct_answer}, Result: {is_correct}")

        return {"predicted_number": predicted_number, "is_correct": is_correct}

    finally:
        # Clean up temporary file
        Path(temp_path).unlink(missing_ok=True)


@app.post("/validate_number/")
async def validate_number(file: UploadFile = File(...), expected_number: int = Form(...)):
    """Handles number validation via sign language video."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
        temp_path = temp.name
        shutil.copyfileobj(file.file, temp)

    try:
        predicted_number = inference.predict(temp_path)
        predicted_number = int(predicted_number) if isinstance(predicted_number, np.integer) else predicted_number

        correct = predicted_number == expected_number
        print(f"Predicted: {predicted_number}, Correct: {correct}")
        return {"predicted_number": predicted_number, "correct": correct}
    finally:
        Path(temp_path).unlink(missing_ok=True)

