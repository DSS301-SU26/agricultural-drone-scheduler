import pandas as pd
import joblib

model_payload = joblib.load('models/drone_decision_model.joblib')
feature_cols = model_payload['feature_columns']
df = pd.read_csv('data/clean/weather_clean_20260702_1526.csv')

print("Feature cols:", feature_cols)
print("DF columns:", df.columns.tolist())

missing = [c for c in feature_cols if c not in df.columns]
print("Missing:", missing)
