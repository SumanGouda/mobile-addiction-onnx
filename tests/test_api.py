"""Integration test for the /predict endpoint — verifies the full
request → preprocessing → prediction → response flow is wired correctly."""
import pytest
from fastapi.testclient import TestClient

from app.api.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_predict_endpoint_returns_valid_response(client):
    payload = {
        "age": 21,
        "daily_screen_time_hours": 5.0,
        "social_media_hours": 2.0,
        "gaming_hours": 1.0,
        "work_study_hours": 3.0,
        "sleep_hours": 7.0,
        "notifications_per_day": 50,
        "app_opens_per_day": 40,
        "weekend_screen_time": 6.0,
        "gender": "Male",
        "stress_level": "Medium",
        "academic_work_impact": "No",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert "addicted_probability" in body
    assert "addicted_label" in body
    assert 0.0 <= body["addicted_probability"] <= 1.0
    assert body["addicted_label"] in (0, 1)


def test_predict_endpoint_rejects_missing_field(client):
    payload = {"age": 21}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    