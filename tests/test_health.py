
from fastapi.testclient import TestClient
from backend.app.main import app

def test_root():
    c=TestClient(app)
    r=c.get("/")
    assert r.status_code==200
    assert r.json()["project"]=="SIH26227"
