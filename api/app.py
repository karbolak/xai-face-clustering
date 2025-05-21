from fastapi import FastAPI
from .routers.predict import router as predict_router

app = FastAPI(
    title="XAI Face Clustering API",
    description="Upload face embeddings → cluster + explain",
    version="0.1.0"
)
app.include_router(predict_router, prefix="/predict")
