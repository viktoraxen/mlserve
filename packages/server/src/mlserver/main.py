from fastapi import FastAPI

from mlserver.routes import router

app = FastAPI()
app.include_router(router)
