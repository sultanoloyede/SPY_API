import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support
import joblib
import os

# 1) Make sure the “models” directory exists
os.makedirs("../models", exist_ok=True)

# 1) Load & prepare data
df = pd.read_csv('../data/processed_spy_15min.csv', parse_dates=['datetime'])
df['target'] = df['0.25_decrement'].shift(-1)
df = df.dropna(subset=['target'])

# 2) Train/test split
split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx].copy()
test  = df.iloc[split_idx:].copy()

# 3) Compute class weight
n_neg = (train['target'] == 0).sum()
n_pos = (train['target'] == 1).sum()
scale_pos_weight = n_neg / n_pos
print(f"scale_pos_weight = {scale_pos_weight:.2f}  (neg={n_neg}, pos={n_pos})\n")

# 4) Feature selection
features = [
    "open","high","low","close","volume",
    "close_v_fst_close","above_200_ema","move_percentage",
    "0.25_growth","big_move_counter","RSI_above_60"
]
X_train = train[features];  y_train = train['target'].astype(int)
X_test  = test[features];   y_test  = test['target'].astype(int)

# 5) Train with XGBoost
model = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42
)
model.fit(X_train, y_train)

# 6) Predict with custom threshold
probs = model.predict_proba(X_test)[:, 1]
threshold = 0.10
y_pred = (probs >= threshold).astype(int)

# 7) Report
print(f"Classification report @ threshold = {threshold:.2f}\n")
print(classification_report(y_test, y_pred, digits=4))

# Optional: detailed PRF for drop class only
p1, r1, f11, _ = precision_recall_fscore_support(
    y_test, y_pred, labels=[1], average='binary'
)
print(f"Drop-class metrics → Precision: {p1:.4f}, Recall: {r1:.4f}, F1: {f11:.4f}")

model_path = "../models/spy_drop_predictor.joblib"
joblib.dump(model, model_path)
print(f"Model saved to: {model_path}")