from fastapi import FastAPI
from schemas import InsuranceInput, PredictionResponse
from src.pipeline.prediction import PredictionPipeline

app = FastAPI(
    title="Insurance Prediction API",
    version="1.0"
)

@app.post("/predict", response_model=PredictionResponse)
def predict(data: InsuranceInput):
    predictor = PredictionPipeline()
    result = predictor.predict(data.dict())
    return result
