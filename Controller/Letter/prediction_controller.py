from urllib.request import Request

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from Services.Letter.prediction_service import predict_letter_sinhala,predict_letter_sinhala_activity

router = APIRouter()

@router.post("/predict-letter-sinhala")
async def predict_sinhala_letter(file: UploadFile = File(...)):
    result = await predict_letter_sinhala(file)
    return JSONResponse(content={"letter": result})

@router.post("/predict-activity/sinhala-letter")
async def predict_sinhala_letter_base64(payload: dict):
    frame_base64 = payload.get("frame")
    if not frame_base64:
        return JSONResponse(content={"error": "Missing frame data"}, status_code=400)

    result = await predict_letter_sinhala_activity(frame_base64)
    if isinstance(result, dict) and "error" in result:
        return JSONResponse(content=result, status_code=400)
    return JSONResponse(content={"predicted_letter": result})
