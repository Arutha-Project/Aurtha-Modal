from fastapi import FastAPI, File, UploadFile, Form
from io import BytesIO
from Services.QuickDraw.quickdraw_service import validate_prediction

quick_draw_app = FastAPI()

@quick_draw_app.post("/predict/")
async def predict(file: UploadFile = File(...), selected_object: str = Form(...)):
    """API Endpoint to predict the drawing from an uploaded image."""
    response = validate_prediction(BytesIO(await file.read()), selected_object)
    return response
