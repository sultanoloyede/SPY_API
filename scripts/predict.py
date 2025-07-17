import joblib
import pandas as pd
import os

# 1) Make sure the “models” directory exists
os.makedirs("../models", exist_ok=True)
os.makedirs("../data", exist_ok=True)

# 1) Load the saved model
model = joblib.load("../models/spy_drop_predictor.joblib")

# 2) Load or receive new intraday data
df = pd.read_csv("../data/processed_spy_15min.csv", parse_dates=["datetime"])

# 3) Prepare your feature set for a given row index `idx`
features = [
    "open","high","low","close","volume",
    "close_v_fst_close","above_200_ema","move_percentage",
    "0.25_growth","0.25_decrement","big_move_counter","RSI_above_60"
]
row = df.loc[5000-361, features].values.reshape(1, -1)
print(row)

# 4) Make a probability prediction and apply your threshold
prob = model.predict_proba(row)[0, 1]
threshold = 0.10
pred = int(prob >= threshold)

print(f"Drop probability: {prob * 100:.4f}%, Prediction: {pred}")
