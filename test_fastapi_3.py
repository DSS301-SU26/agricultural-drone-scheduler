from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)

response = client.put("/api/plots/Vuon%20Lua%20Tien%20Giang", json={})
print("Test Empty Dict:", response.status_code, response.text)

response = client.put("/api/plots/Vuon%20Lua%20Tien%20Giang", json={"latitude": 10.0, "longitude": 105.0})
print("Test No Name:", response.status_code, response.text)

