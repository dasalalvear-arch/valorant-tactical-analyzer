import pytest
import pandas as pd

@pytest.fixture
def sample_kills_df():
    return pd.DataFrame({
        "player": ["TenZ#NA1", "TenZ#NA1", "TenZ#NA1", "s1mple#EU1"],
        "map": ["ascent", "ascent", "ascent", "ascent"],
        "x": [3200.0, 5800.0, 4500.0, 3100.0],
        "y": [2000.0, 3800.0, 1500.0, 3200.0],
        "side": ["ATK", "DEF", "ATK", "DEF"],
        "result": ["kill", "kill", "death", "kill"],
    })

@pytest.fixture
def sample_zone_stats_df():
    return pd.DataFrame({
        "player": ["TenZ#NA1", "TenZ#NA1"],
        "map": ["ascent", "ascent"],
        "zone_row": [2, 3],
        "zone_col": [1, 3],
        "side": ["ATK", "DEF"],
        "kills": [15, 8],
        "deaths": [3, 5],
        "total_events": [18, 13],
        "kill_rate": [0.83, 0.62],
        "insufficient_sample": [False, False],
    })

@pytest.fixture
def mock_model(mocker):
    model = mocker.MagicMock()
    model.predict_proba.return_value = [[0.3, 0.7]]
    return model
