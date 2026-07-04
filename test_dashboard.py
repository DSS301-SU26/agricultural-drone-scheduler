from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)
res = client.get("/api/dashboard/slots")
data = res.json()

for i, slot in enumerate(data["slots"]):
    champ = slot["decision_engine"]["champion_score"]
    chall = slot["decision_engine"]["challenger_score"]
    print(f"Slot {i}: champ={champ}, chall={chall}")
