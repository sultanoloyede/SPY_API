from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib, os, sys, subprocess

# ── ensure TensorFlow is present ────────────────────────
try:
    import tensorflow as tf
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow>=2.16"])
    import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import backend as K

# ── paths (adjust if needed) ────────────────────────────
MODEL_PATH   = "../models/spy_drop_predictor.keras"
SCALER_PATH  = "../models/scaler.joblib"
THR_PATH     = "../models/pred_threshold.txt"
DATA_CSV     = "../data/processed_spy_15min.csv"

WINDOW       = 24                     # same as training
FEATURE_COLS = [
    "open","high","low","close","volume",
    "close_v_fst_close","above_200_ema","move_percentage",
    "0.25_growth","0.25_decrement","big_move_counter","RSI_above_60"
]

os.makedirs("../models", exist_ok=True)
os.makedirs("../data",   exist_ok=True)

# ── custom loss stub so Keras can load the model ───────
def focal_loss(gamma=2.0, alpha=0.25):
    def _fl(y_true, y_pred):
        eps = 1e-9
        y_true = K.cast(y_true, tf.float32)
        pt = tf.where(K.equal(y_true, 1), y_pred + eps, 1 - y_pred + eps)
        return -alpha * K.pow(1 - pt, gamma) * K.log(pt)
    return _fl

# ── load artefacts ─────────────────────────────────────
try:
    model  = keras.models.load_model(MODEL_PATH,
                                     custom_objects={"_fl": focal_loss()})
    scaler = joblib.load(SCALER_PATH)
    THRESHOLD = float(open(THR_PATH).read().strip())
except Exception as e:
    raise RuntimeError(f"Failed to load model/scaler/threshold: {e}")

# ── FastAPI boilerplate ────────────────────────────────
app = FastAPI()

class Prediction(BaseModel):
    probability: float
    prediction: int    # 1 = drop, 0 = no-drop

@app.get("/predict", response_model=Prediction)
def predict():
    try:
        df = pd.read_csv(DATA_CSV, parse_dates=["datetime"])
    except Exception as e:
        raise HTTPException(500, f"Failed to load data: {e}")

    # basic validation
    if len(df) < WINDOW:
        raise HTTPException(500, f"Need ≥ {WINDOW} rows; got {len(df)}")
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise HTTPException(500, f"Missing features: {missing}")

    window_df = df.iloc[-WINDOW:][FEATURE_COLS].astype("float32")
    # scale & reshape
    X = scaler.transform(window_df).reshape((1, WINDOW, len(FEATURE_COLS)))

    try:
        prob = float(model.predict(X, verbose=0)[0][0])
        pred = int(prob >= THRESHOLD)
    except Exception as e:
        raise HTTPException(500, f"Prediction error: {e}")

    return {"probability": prob, "prediction": pred}

@app.get("/")
def health():
    return {"status": "ok"}
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib, os, sys, subprocess

# ── ensure TensorFlow is present ────────────────────────
try:
    import tensorflow as tf
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow>=2.16"])
    import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import backend as K

# ── paths (adjust if needed) ────────────────────────────
MODEL_PATH   = "../models/spy_drop_predictor.keras"
SCALER_PATH  = "../models/scaler.joblib"
THR_PATH     = "../models/pred_threshold.txt"
DATA_CSV     = "../data/processed_spy_15min.csv"

WINDOW       = 24                     # same as training
FEATURE_COLS = [
    "open","high","low","close","volume",
    "close_v_fst_close","above_200_ema","move_percentage",
    "0.25_growth","0.25_decrement","big_move_counter","RSI_above_60"
]

os.makedirs("../models", exist_ok=True)
os.makedirs("../data",   exist_ok=True)

# ── custom loss stub so Keras can load the model ───────
def focal_loss(gamma=2.0, alpha=0.25):
    def _fl(y_true, y_pred):
        eps = 1e-9
        y_true = K.cast(y_true, tf.float32)
        pt = tf.where(K.equal(y_true, 1), y_pred + eps, 1 - y_pred + eps)
        return -alpha * K.pow(1 - pt, gamma) * K.log(pt)
    return _fl

# ── load artefacts ─────────────────────────────────────
try:
    model  = keras.models.load_model(MODEL_PATH,
                                     custom_objects={"_fl": focal_loss()})
    scaler = joblib.load(SCALER_PATH)
    THRESHOLD = float(open(THR_PATH).read().strip())
except Exception as e:
    raise RuntimeError(f"Failed to load model/scaler/threshold: {e}")

# ── FastAPI boilerplate ────────────────────────────────
app = FastAPI()

class Prediction(BaseModel):
    probability: float
    prediction: int    # 1 = drop, 0 = no-drop

@app.get("/predict", response_model=Prediction)
def predict():
    try:
        df = pd.read_csv(DATA_CSV, parse_dates=["datetime"])
    except Exception as e:
        raise HTTPException(500, f"Failed to load data: {e}")

    # basic validation
    if len(df) < WINDOW:
        raise HTTPException(500, f"Need ≥ {WINDOW} rows; got {len(df)}")
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise HTTPException(500, f"Missing features: {missing}")

    window_df = df.iloc[-WINDOW:][FEATURE_COLS].astype("float32")
    # scale & reshape
    X = scaler.transform(window_df).reshape((1, WINDOW, len(FEATURE_COLS)))

    try:
        prob = float(model.predict(X, verbose=0)[0][0])
        pred = int(prob >= THRESHOLD)
    except Exception as e:
        raise HTTPException(500, f"Prediction error: {e}")

    return {"probability": prob, "prediction": pred}

@app.get("/")
def health():
    return {"status": "ok"}
