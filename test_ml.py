import pandas as pd
import joblib

model_payload = joblib.load('models/drone_decision_model.joblib')
champion = model_payload['champion']
challenger = model_payload.get('challenger')
feature_cols = model_payload['feature_columns']
df = pd.read_csv('data/clean/weather_clean_20260702_1526.csv')

for col in feature_cols:
    if col not in df.columns:
        df[col] = 0

X = df[feature_cols].copy()
probs = champion.predict_proba(X)

if challenger:
    chall_probs = challenger.predict_proba(X)
else:
    chall_probs = probs

print("First 5 champ probs:", [p[0] for p in probs[:5]])
print("First 5 chall probs:", [p[0] for p in chall_probs[:5]])
