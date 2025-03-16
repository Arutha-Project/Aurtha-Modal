import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from Controller.QuickDraw.quick_draw_controller import quick_draw_app
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

if __name__ == "__main__":
    uvicorn.run("app:main_app", host="0.0.0.0", port=9090, reload=True)
