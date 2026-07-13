from fastapi.testclient import TestClient
from src.api import app
import json

client = TestClient(app)
res = client.get("/api/dashboard/slots?location=Dong Thap")
data = res.json()

for i, slot in enumerate(data["slots"]):
    champ = slot.get("decision_engine", {}).get("champion_score") or slot.get("champion_score")
    fly = slot.get("flyability_score", 0) * 100
    print(f"Slot {i}: champ={champ}, flyability={fly:.1f}%")
    if i == 3:
        pass
