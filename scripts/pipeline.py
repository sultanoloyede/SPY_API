import time
import pandas as pd
import joblib
from twelvedata import TDClient
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from dotenv import load_dotenv
import os

load_dotenv()

# === Configuration ===
API_KEY = os.getenv("TWELVE_DATA_API_KEY")
if not API_KEY:
    raise RuntimeError("TWELVE_DATA_API_KEY not set. Please ensure you have a .env file with that variable.")
RAW_CSV_PATH = "../data/spy_15min.csv"
PROCESSED_CSV_PATH = "../data/processed_spy_15min.csv"
MODEL_PATH = "../models/spy_drop_predictor.joblib"
FETCH_INTERVAL_SEC = 60 * 10  # fetch & run every 15 minutes

# === Helper Functions ===

def fetch_intraday_data():
    """Fetch latest 15-min SPY bars from Twelve Data and save raw CSV."""
    td = TDClient(apikey=API_KEY)
    ts = td.time_series(symbol="SPY", interval="15min", outputsize=5000)
    df = ts.as_pandas()
    df.index = pd.to_datetime(df.index)
    df.to_csv(RAW_CSV_PATH)
    return df

def process_data(df):
    """Compute target and features, save processed CSV."""
    df = df.copy()
    
    # 1) Ensure chronological order & extract datetime
    df = df.sort_values('datetime')
    df['datetime'] = df.index

    # Compute above‐first‐bar‐today feature:
    df['date'] = df['datetime'].dt.date
    first_close = df.groupby('date')['close'].first()
    df['today_first_close'] = df['date'].map(first_close)
    df['close_v_fst_close'] = (df['close'] > df['today_first_close']).astype(int)
    # 200‐EMA feature:
    df['above_200_ema'] = (df['close'] > df['close'].ewm(span=200, adjust=False).mean()).astype(int)
    # Percent change bar:
    df['move_percentage'] = (df['close'] / df['open'] - 1) * 100
    # 0.25% flags:
    df['0.25_growth'] = (df['move_percentage'] >= 0.25).astype(int)
    df['0.25_decrement'] = (df['move_percentage'] <= -0.25).astype(int)
    # 6) big_move_counter exactly like script
    df['big_move'] = df['0.25_growth'] - df['0.25_decrement']
    df['big_move_counter'] = (
        df.groupby('date', sort=True)['big_move']
          .cumsum()
    )
    df = df.drop(columns=['date', 'big_move', 'today_first_close'])  # clean up helpers
    # RSI 14 & flag:
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI_above_60'] = (100 - (100 / (1 + rs)) >= 60).astype(int)
    # Compute next‐bar drop target:
    df['target'] = df['0.25_decrement'].shift(-1, fill_value=0)
    # Cleanup:
    features = [
        "open","high","low","close","volume","close_v_fst_close",
        "above_200_ema","move_percentage","0.25_growth",
        "0.25_decrement","big_move_counter","RSI_above_60","target"
    ]
    df = df.dropna(subset=features)
    df.to_csv(PROCESSED_CSV_PATH, index=False)
    print(df.tail())
    return df[features]

def train_and_save_model(df_processed):
    """Train XGBoost on the processed data and save the model."""
    split_idx = int(len(df_processed) * 0.8)
    train = df_processed.iloc[:split_idx]
    # Compute scale_pos_weight
    n_neg = (train['target'] == 0).sum()
    n_pos = (train['target'] == 1).sum()
    scale_pos_weight = n_neg / n_pos
    # Features & target
    feature_cols = [
        "open","high","low","close","volume","close_v_fst_close",
        "above_200_ema","move_percentage","0.25_growth","0.25_decrement",
        "big_move_counter","RSI_above_60"
    ]
    X_train = train[feature_cols]
    y_train = train['target'].astype(int)
    # Train
    model = XGBClassifier(use_label_encoder=False,
                          eval_metric="logloss",
                          scale_pos_weight=scale_pos_weight,
                          random_state=42)
    model.fit(X_train, y_train)
    # Save
    joblib.dump(model, MODEL_PATH)
    return model

def predict_latest(df_processed, model, threshold=0.10):
    """Predict drop probability on the newest bar and print result."""
    feature_cols = [
        "open","high","low","close","volume","close_v_fst_close",
        "above_200_ema","move_percentage","0.25_growth","0.25_decrement",
        "big_move_counter","RSI_above_60"
    ]
    latest = df_processed.iloc[-1:][feature_cols]
    prob = model.predict_proba(latest)[0, 1]
    pred = int(prob >= threshold)
    print(f"[{pd.Timestamp.now()}] Prob(drop)={prob * 100:.3f}%, Pred={pred} (thr={threshold})")

# === Main Loop ===
if __name__ == "__main__":
    while True:
        try:
            raw_df = fetch_intraday_data()
            proc_df = process_data(raw_df)
            model = train_and_save_model(proc_df)
            predict_latest(proc_df, model)
        except Exception as e:
            print(f"[{pd.Timestamp.now()}] Error: {e}")
        time.sleep(FETCH_INTERVAL_SEC)
