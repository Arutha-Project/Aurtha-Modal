from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from inference import SignLanguageInference
import tempfile
import shutil

app = FastAPI()

# Initialize the inference class
model_path = r"C:\Users\WW\Desktop\Final Reseach Data\rf_sign_language_model.joblib"
inference = SignLanguageInference(model_path)

@app.get("/", response_class=HTMLResponse)
async def main():
    """Serve a simple HTML form for uploading files."""
    return """
    <!DOCTYPE html>
    <html>
    <body>
        <h2>Upload MP4 Video for Sign Language Prediction</h2>
        <form action="/predict/" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept=".mp4" required>
            <button type="submit">Upload and Predict</button>
        </form>
    </body>
    </html>
    """

@app.post("/predict/")
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