from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)

response = client.put("/api/plots/Vuon%20Lua%20Tien%20Giang", json={"id":"Vuon Lua Tien Giang","name":"Vuon Lua Tien Giang","latitude":10.3,"longitude":105.1})
print("Test 1:", response.status_code, response.text)

response = client.put("/api/plots/Vuon%20Lua%20Tien%20Giang", data='{"id": "Vuon Lua Tien Giang"}', headers={"Content-Type": "application/json"})
print("Test 2:", response.status_code, response.text)

response = client.put("/api/plots/Vuon%20Lua%20Tien%20Giang", json=[1, 2, 3])
print("Test 3:", response.status_code, response.text)

response = client.put("/api/plots/Vuon%20Lua%20Tien%20Giang", json="just a string")
print("Test 4:", response.status_code, response.text)
