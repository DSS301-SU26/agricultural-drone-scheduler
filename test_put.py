import httpx

r = httpx.put('http://127.0.0.1:8000/api/plots/Vuon%20Lua%20Tien%20Giang', json={"plot_name": "Test"})
print(r.status_code)
print(r.text)
