import requests

session = requests.Session()
session.trust_env = False

resp = session.post(
    "http://127.0.0.1:11434/v1/embeddings",
    json={"model": "bge-m3:latest", "input": ["hello"]},
    headers={"Content-Type": "application/json"},
    timeout=30
)
print(resp.status_code)
print(resp.text)
print(resp.headers)