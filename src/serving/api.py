"""
Standard FastAPI serving app for the SoC LSTM model.

Run locally:
    uvicorn src.serving.api:app --reload

Then visit http://127.0.0.1:8000/docs for interactive Swagger UI.
"""
import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.serving.predictor import SoCPredictor
from src.utils.logging_config import get_logger

logger = get_logger("api")

MODEL_PATH = os.getenv("MODEL_PATH", "models/lstm_model.pth")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.joblib")

app = FastAPI(
    title="Battery SoC Estimator API",
    description="Predicts State of Charge (SoC) for an LG 18650HG2 li-ion cell "
                "from a short window of voltage/current/temperature readings, "
                "using a trained LSTM model.",
    version="1.0.0",
)

# Open CORS -- this API has no auth (deliberate, portfolio demo) and the demo/
# page needs to call it directly from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor: Optional[SoCPredictor] = None


@app.on_event("startup")
def load_model():
    global predictor
    predictor = SoCPredictor(MODEL_PATH, SCALER_PATH)
    logger.info(f"Loaded model '{predictor.model_name}' (sequence_length={predictor.sequence_length}) "
                f"from {MODEL_PATH}")


class Reading(BaseModel):
    time_s: float = Field(..., description="Timestamp in seconds, monotonically increasing.")
    voltage_v: float = Field(..., description="Terminal voltage in volts.")
    current_a: float = Field(..., description="Current in amps (negative = discharging).")
    temperature_c: float = Field(..., description="Cell temperature in degrees Celsius.")


class PredictRequest(BaseModel):
    readings: list[Reading] = Field(
        ..., description="Chronological list of sensor readings. Must contain at "
                          "least `sequence_length` readings (see /health for the "
                          "required length). Only the most recent window is used."
    )


class PredictResponse(BaseModel):
    predicted_soc: float = Field(..., description="Predicted State of Charge, in [0, 1].")
    predicted_soc_pct: float = Field(..., description="Predicted SoC as a percentage.")
    model_name: str
    sequence_length_used: int


class HealthResponse(BaseModel):
    status: str
    model_name: str
    sequence_length_required: int
    feature_columns: list


@app.get("/health", response_model=HealthResponse)
def health():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return HealthResponse(
        status="ok",
        model_name=predictor.model_name,
        sequence_length_required=predictor.sequence_length,
        feature_columns=predictor.feature_columns,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    readings = [r.model_dump() for r in request.readings]
    try:
        soc = predictor.predict_from_readings(readings)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return PredictResponse(
        predicted_soc=soc,
        predicted_soc_pct=round(soc * 100, 2),
        model_name=predictor.model_name,
        sequence_length_used=predictor.sequence_length,
    )


@app.get("/")
def root():
    return {"message": "Battery SoC Estimator API. See /docs for usage."}
