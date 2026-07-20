from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)

response = client.put("/api/plots/Vuon%20Lua%20Tien%20Giang", json={"plot_name": "Test", "latitude": 10, "longitude": 105})
print(response.status_code)
print(response.json())
