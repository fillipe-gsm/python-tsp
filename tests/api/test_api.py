from fastapi.testclient import TestClient
import json
from api.tspapi import app

client = TestClient(app)


def test_api():
    with open("tests/api/request.json") as f:
        request = json.load(f)
        response = client.post(
            "/tsp/",
            json=request,
        )

    assert response.status_code == 200
    res = response.json()
    assert res["tour"] == [
        "New York",
        "Boston",
        "Chicago",
        "Minneapolis",
        "Denver",
        "Salt Lake City",
        "Seattle",
        "San Francisco",
        "Los Angeles",
        "Phoenix",
        "Houston",
        "Dallas",
        "St. Louis",
    ]
    assert round(res["distance"]) == 11609


def test_invalid_request():
    with open("tests/api/invalid_request.json") as f:
        request = json.load(f)
        response = client.post(
            "/tsp/",
            json=request,
        )

    assert response.status_code == 422
