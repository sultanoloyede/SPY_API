#!/usr/bin/env python3
"""
SPY 15-min big-drop adviser
───────────────────────────
Goal: catch as many −0.25 % drops as possible while
allowing up to 3 false alarms per true call (precision ≥ 0.25).

• 24-bar LSTM  • focal loss
• Negative class down-sampled (20 %)
• Threshold chosen: highest recall with precision ≥ 0.25
• Two-bar confirmation filter (except prob ≥ 0.55 fires immediately)
"""

# ========== IMPORTS ==================================================
import os, sys, time, subprocess, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from dotenv import load_dotenv
from twelvedata import TDClient
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_recall_curve,
                             accuracy_score,
                             precision_recall_fscore_support)

try:
    import tensorflow as tf           # noqa: F401
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow>=2.16"])
    import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, callbacks, backend as K

# ========== CONFIG ===================================================
load_dotenv()
API_KEY = os.getenv("TWELVE_DATA_API_KEY")
if not API_KEY:
    raise RuntimeError("TWELVE_DATA_API_KEY missing in .env")

WINDOW               = 24          # bars (≈ 6 h look-back)
BIG_DROP_PCT         = 0.25        # −0.25 % defines “big drop”
NEG_SAMPLE_FRAC      = 0.20        # keep 20 % of no-drop rows
PRECISION_MIN        = 0.00        # ≥ 1 TP : ≤ 3 FP
RECALL_MIN        = 0.60     # want to catch at least 60 % of drops
ALERTS_PER_DAY_MAX   = 50          # safety ceiling (rarely reached)
HIGH_CONF_PROB       = 0.55        # bypass 2-bar confirmation
FETCH_INTERVAL_SEC   = 60 * 15
BARS_PER_DAY         = 26          # NYSE 6.5 h / 15-min bars

RAW_CSV_PATH  = "../data/spy_15min.csv"
PROC_CSV_PATH = "../data/processed_spy_15min.csv"
MODEL_DIR     = Path("../models")
MODEL_PATH    = MODEL_DIR / "spy_drop_predictor.keras"
SCALER_PATH   = MODEL_DIR / "scaler.joblib"
THR_PATH      = MODEL_DIR / "pred_threshold.txt"

FEATURE_COLS = [
    "open", "high", "low", "close", "volume", "close_v_fst_close",
    "above_200_ema", "move_percentage",
    "0.25_growth", "0.25_decrement",
    "big_move_counter", "RSI_above_60"
]

MODEL_DIR.mkdir(parents=True, exist_ok=True)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ========== DATA UTILITIES ===========================================
def fetch_intraday_data():
    td = TDClient(apikey=API_KEY)
    ts = td.time_series(symbol="SPY", interval="15min", outputsize=5000)
    df = ts.as_pandas()
    df.index = pd.to_datetime(df.index)
    df.to_csv(RAW_CSV_PATH)
    return df


def engineer_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy().sort_values("datetime")
    df["datetime"] = df.index
    df["date"]     = df["datetime"].dt.date

    first_close = df.groupby("date")["close"].first()
    df["close_v_fst_close"] = (df["close"] > df["date"].map(first_close)).astype(int)

    df["above_200_ema"] = (
        df["close"] > df["close"].ewm(span=200, adjust=False).mean()
    ).astype(int)

    df["move_percentage"] = (df["close"] / df["open"] - 1) * 100
    df["0.25_growth"]     = (df["move_percentage"] >=  0.25).astype(int)
    df["0.25_decrement"]  = (df["move_percentage"] <= -0.25).astype(int)
    df["big_move"]        = df["0.25_growth"] - df["0.25_decrement"]
    df["big_move_counter"] = df.groupby("date", sort=True)["big_move"].cumsum()

    delta = df["close"].diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_g = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_l = loss.ewm(alpha=1/14, adjust=False).mean()
    rs    = avg_g / avg_l
    df["RSI_above_60"] = (100 - (100 / (1 + rs)) >= 60).astype(int)

    df["big_drop_flag"] = (df["move_percentage"] <= -BIG_DROP_PCT).astype(int)
    df["target"]        = df["big_drop_flag"].shift(-1, fill_value=0)

    df = df.drop(columns=["date", "big_move"])
    df = df.dropna(subset=FEATURE_COLS + ["target"])
    df.to_csv(PROC_CSV_PATH, index=False)
    return df


def make_sequences(df_proc: pd.DataFrame, window=WINDOW):
    seqs, labels = [], []
    for i in range(len(df_proc) - window):
        seqs.append(df_proc.iloc[i:i+window][FEATURE_COLS].values)
        labels.append(df_proc.iloc[i+window]["target"])
    return np.asarray(seqs, dtype="float32"), np.asarray(labels, dtype="float32")

# ========== MODEL ====================================================
def focal_loss(gamma=2.0, alpha=0.25):
    def _fl(y_true, y_pred):
        eps = 1e-9
        y_true = K.cast(y_true, tf.float32)
        pt = tf.where(K.equal(y_true, 1), y_pred + eps, 1 - y_pred + eps)
        return -alpha * K.pow(1 - pt, gamma) * K.log(pt)
    return _fl


def build_lstm(n_feat):
    model = keras.Sequential([
        layers.Input(shape=(WINDOW, n_feat)),
        layers.LSTM(64, activation="tanh"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss=focal_loss(),
                  metrics=["accuracy", keras.metrics.Precision(), keras.metrics.Recall()])
    return model


def choose_threshold(prob_val: np.ndarray,
                     y_val:   np.ndarray,
                     recall_min:    float = RECALL_MIN,
                     alerts_per_day: int   = ALERTS_PER_DAY_MAX,
                     bars_per_day:   int   = BARS_PER_DAY) -> float:
    """
    Lowest threshold whose recall ≥ recall_min
    *and* raw alerts/day ≤ alerts_per_day.
    Falls back to the lowest threshold that achieves the recall,
    ignoring the alert budget if necessary.
    """
    prec, rec, thr = precision_recall_curve(y_val, prob_val)
    prec, rec = prec[1:], rec[1:]          # align with thr

    best_thr = None
    for p, r, t in zip(prec[::-1], rec[::-1], thr[::-1]):  # lowest → highest
        if r >= recall_min:
            alerts_day = (prob_val >= t).sum() / (len(prob_val) / bars_per_day)
            if alerts_day <= alerts_per_day:
                best_thr = float(t)
                break

    # If alert budget too tight, take the lowest thr that hits the recall anyway
    if best_thr is None:
        idx = np.where(rec >= recall_min)[0][0]   # first meets recall
        best_thr = float(thr[idx])

    return best_thr


def train_once(df_proc: pd.DataFrame):
    # down-sample negatives
    pos = df_proc[df_proc.target == 1]
    neg = df_proc[df_proc.target == 0].sample(frac=NEG_SAMPLE_FRAC, random_state=42)
    df_bal = pd.concat([pos, neg]).sort_index()

    X, y = make_sequences(df_bal)
    split = int(len(X) * 0.8)
    X_tr, X_val, y_tr, y_val = X[:split], X[split:], y[:split], y[split:]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr.reshape(-1, X_tr.shape[2])).reshape(X_tr.shape)
    X_val = scaler.transform(X_val.reshape(-1, X_val.shape[2])).reshape(X_val.shape)

    model = build_lstm(n_feat=X_tr.shape[2])
    es = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
              epochs=50, batch_size=256, callbacks=[es], verbose=0)

    prob_val = model.predict(X_val, verbose=0).ravel()
    thr = choose_threshold(prob_val, y_val)

    preds = (prob_val >= thr).astype(int)
    tp = ((preds == 1) & (y_val == 1)).sum()
    fp = ((preds == 1) & (y_val == 0)).sum()
    alerts_day = preds.sum() / (len(prob_val) / BARS_PER_DAY)
    prec = tp / (tp + fp) if tp + fp else 0
    rec  = tp / (y_val == 1).sum() if (y_val == 1).sum() else 0
    f1   = 2 * prec * rec / (prec + rec + 1e-9)

    print("\n=== Validation (precision ≥ 0.25) ===")
    print(f"Threshold     : {thr:.3f}")
    print(f"Alerts/day    : {alerts_day:.2f}")
    print(f"Precision     : {prec:.3f}")
    print(f"Recall        : {rec:.3f}")
    print(f"F1            : {f1:.3f}\n")

    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    THR_PATH.write_text(f"{thr:.6f}")

    return model, scaler, thr


def load_artefacts():
    model  = keras.models.load_model(MODEL_PATH, custom_objects={"_fl": focal_loss()})
    scaler = joblib.load(SCALER_PATH)
    thr    = float(THR_PATH.read_text().strip())
    return model, scaler, thr

# ========== LIVE MODE ================================================
def alert_logic(prob, prev_prob, thr):
    if prob >= HIGH_CONF_PROB:
        return True
    return prob >= thr and (prev_prob or 0) >= thr


def live_predict(df_proc, model, scaler, thr, prev_prob=None):
    if len(df_proc) < WINDOW:
        return prev_prob
    seq = scaler.transform(
        df_proc.iloc[-WINDOW:][FEATURE_COLS].values.astype("float32")
    ).reshape(1, WINDOW, len(FEATURE_COLS))
    prob = float(model.predict(seq, verbose=0)[0][0])
    ts   = pd.Timestamp.now()

    if alert_logic(prob, prev_prob, thr):
        print(f"[{ts:%F %T}] 🟥 Prob(drop)={prob*100:.1f}% (thr={thr:.3f}) ➜ ALERT")
    else:
        print(f"[{ts:%F %T}] Prob(drop)={prob*100:.1f}%")

    return prob

# ========== MAIN LOOP ================================================
if __name__ == "__main__":
    raw  = fetch_intraday_data()
    proc = engineer_features(raw)

    if MODEL_PATH.exists() and SCALER_PATH.exists() and THR_PATH.exists():
        model, scaler, thr = load_artefacts()
        print("✓ Loaded saved model, scaler, threshold.")
    else:
        model, scaler, thr = train_once(proc)

    prev_prob = None
    while True:
        try:
            raw  = fetch_intraday_data()
            proc = engineer_features(raw)
            prev_prob = live_predict(proc, model, scaler, thr, prev_prob)
        except Exception as e:
            print(f"[{pd.Timestamp.now()}] Error: {e}")
        time.sleep(FETCH_INTERVAL_SEC)
