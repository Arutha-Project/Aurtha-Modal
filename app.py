import uvicorn
from fastapi import FastAPI
from Controller.QuickDraw.quick_draw_controller import quick_draw_app
from Controller.ObjectIdentifier.object_identifier_controller import object_identifier_app
main_app = FastAPI()

main_app.mount("/quickdraw", quick_draw_app)
main_app.mount("/objectIdentifier", object_identifier_app)
main_app.mount("/numberIdentifier", number_identifier_app)

if __name__ == "__main__":
    uvicorn.run("app:main_app", host="0.0.0.0", port=9090, reload=True)
