"""
Tests de la API FastAPI.

Los datos de la app (features, zone_stats, modelo) se inyectan mockeando
`_load_app_data`; no se leen CSVs reales ni se entrena ningún modelo. El
TestClient se usa como context manager para que corra el lifespan y se
carguen los globals antes de cada request.
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def app_data():
    features = pd.DataFrame({
        "player": ["TenZ#NA1", "s1mple#EU1"],
        "avg_acs": [280.0, 250.0],
        "avg_kd": [1.4, 1.2],
        "avg_hs": [0.30, 0.28],
        "winrate": [0.60, 0.55],
        "games": [50, 40],
    })
    zone_stats = pd.DataFrame({
        "player": ["TenZ#NA1", "TenZ#NA1", "s1mple#EU1"],
        "map": ["ascent", "ascent", "ascent"],
        "zone_row": [2, 1, 2],
        "zone_col": [3, 4, 3],
        "side": ["ATK", "DEF", "ATK"],
        "kills": [20, 12, 15],
        "deaths": [5, 8, 10],
        "total_events": [25, 20, 25],
        "kill_rate": [0.80, 0.60, 0.60],
        "insufficient_sample": [False, False, False],
    })
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.4, 0.6]])
    return features, zone_stats, model


@pytest.fixture
def client(app_data):
    with patch("api.main._load_app_data", return_value=app_data):
        from api.main import app
        with TestClient(app) as c:
            yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_match_returns_probabilities(client):
    response = client.post("/predict-match", json={
        "team_a": ["TenZ#NA1"],
        "team_b": ["s1mple#EU1"],
        "map": "ascent",
    })
    assert response.status_code == 200
    data = response.json()
    assert "team_a_win_prob" in data
    assert "team_b_win_prob" in data
    assert abs(data["team_a_win_prob"] + data["team_b_win_prob"] - 1.0) < 0.01


def test_player_card_returns_stats(client):
    response = client.get("/player/TenZ/NA1/card")
    assert response.status_code == 200
    assert "avg_acs" in response.json()


def test_player_card_404_for_unknown_player(client):
    response = client.get("/player/Nobody/XXX1/card")
    assert response.status_code == 404


def test_player_zones_returns_png(client):
    response = client.get("/player/TenZ/NA1/zones/ascent")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


def test_compare_returns_png(client):
    response = client.get("/compare/TenZ/NA1/s1mple/EU1/ascent")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


def test_simulate_returns_kill_rates(client):
    response = client.post("/simulate/TenZ/NA1/ascent", json={
        "side": "ATK",
        "new_distribution": {"2,3": 1.0},
    })
    assert response.status_code == 200
    data = response.json()
    assert "simulated_kill_rate" in data
    assert data["simulated_kill_rate"] == 0.80


def test_simulate_rejects_invalid_distribution(client):
    response = client.post("/simulate/TenZ/NA1/ascent", json={
        "side": "ATK",
        "new_distribution": {"2,3": 0.5, "1,4": 0.3},  # suma 0.8
    })
    assert response.status_code == 400


def test_team_zones_returns_png(client):
    # El '#' del player_id se envía percent-encoded (%23); crudo lo tomaría como fragment.
    response = client.get("/team-zones/TenZ%23NA1/s1mple%23EU1/ascent")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
