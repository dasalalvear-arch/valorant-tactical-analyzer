import pytest
import pandas as pd

@pytest.fixture
def sample_kills_df():
    return pd.DataFrame({
        "player": ["TenZ#NA1", "TenZ#NA1", "TenZ#NA1", "s1mple#EU1"],
        "map": ["ascent", "ascent", "ascent", "ascent"],
        "x": [0.3, 0.7, 0.5, 0.2],
        "y": [0.4, 0.8, 0.2, 0.6],
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
