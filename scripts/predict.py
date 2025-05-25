import joblib
import pandas as pd

# 1) Load the saved model
model = joblib.load("/Users/bolajioloyede/Documents/FQ_Predictor/nba_betting_env/intraday/models/spy_drop_predictor.joblib")

# 2) Load or receive new intraday data
df = pd.read_csv("/Users/bolajioloyede/Documents/FQ_Predictor/nba_betting_env/intraday/data/processed_spy_15min.csv", parse_dates=["datetime"])

# 3) Prepare your feature set for a given row index `idx`
features = [
    "open","high","low","close","volume",
    "close_v_fst_close","above_200_ema","move_percentage",
    "0.25_growth","big_move_counter","RSI_above_60"
]
row = df.loc[5000-1, features].values.reshape(1, -1)
print(row)

# 4) Make a probability prediction and apply your threshold
prob = model.predict_proba(row)[0, 1]
threshold = 0.10
pred = int(prob >= threshold)

print(f"Drop probability: {prob * 100:.4f}%, Prediction: {pred}")
