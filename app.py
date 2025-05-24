import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from Controller.QuickDraw.quick_draw_controller import quick_draw_app
from Controller.Number.numbers_inference_component import app as number_app
from Controller.ObjectIdentifier.Sign_language_component_main import app as object_identifier_app
# from ModalTraning.LettersIdentifier.Sinhala.sinhalaApp import app as sinhala_app
# from ModalTraning.LettersIdentifier.English.englishApp import app as english_app

main_app = FastAPI()

origins = [
    "http://localhost:5173",
]

main_app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

main_app.mount("/quickdraw", quick_draw_app)
main_app.mount("/numbers", number_app)
main_app.mount("/objectIdentifier", object_identifier_app)
# main_app.mount("/sinhala", sinhala_app)
# main_app.mount("/english", english_app)

if __name__ == "__main__":
    uvicorn.run("app:main_app", host="0.0.0.0", port=9090, reload=True)
