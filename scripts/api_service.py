from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# Make sure the “models” directory exists
os.makedirs("../models", exist_ok=True)
os.makedirs("../data", exist_ok=True)

# Config (adjust paths or use env vars)
MODEL_PATH      = "../models/spy_drop_predictor.joblib"
DATA_CSV_PATH   = "../data/processed_spy_15min.csv"
THRESHOLD       = 0.10

# Load model
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError("Failed to load model")

app = FastAPI()

class Prediction(BaseModel):
    probability: float
    prediction: int

@app.get("/predict", response_model=Prediction)
def predict():
    try:
        df = pd.read_csv(DATA_CSV_PATH, parse_dates=['datetime'])
    except Exception as e:
        raise HTTPException(500, f"Failed to load data: {e}")
    
    feature_cols = [
        "open","high","low","close","volume",
        "close_v_fst_close","above_200_ema","move_percentage",
        "0.25_growth","0.25_decrement","big_move_counter","RSI_above_60"
    ]

    if df.empty or any(c not in df.columns for c in feature_cols):
        raise HTTPException(500, "Data missing required features")
    row = df.iloc[[-1]][feature_cols]

    try:
        prob = float(model.predict_proba(row)[0,1])   
        pred = int(prob >= THRESHOLD)
    except Exception as e:
        raise HTTPException(500, f"Prediction error: {e}")
    
    return {"probability": prob, "prediction": pred}

@app.get("/")
def health():
    return {"status": "ok"}


