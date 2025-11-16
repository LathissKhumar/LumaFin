from fastapi.testclient import TestClient
from src.api.main import app
from datetime import date


client = TestClient(app)


def test_categorize_with_label_and_time():
    payload = {
        "merchant": "Starbucks Coffee",
        "amount": 4.5,
        "date": str(date(2025, 11, 15)),
        "description": "Morning latte",
        "user_id": 1,
        "label": "coffee"
    }
    response = client.post("/categorize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 'category' in data
    assert 'confidence' in data['category']
    # Ensure prediction includes category name
    assert isinstance(data['category']['name'], str)
