from urllib.request import Request

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from Services.Letter.prediction_service import predict_letter

router = APIRouter()

@router.post("/")
async def predict_sinhala_letter(file: UploadFile = File(...)):
    result = await predict_letter(file)
    return JSONResponse(content={"letter": result})
