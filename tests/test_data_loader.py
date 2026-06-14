import pandas as pd
from src.data_loader import fetch_matches, extract_kills, save_kills

MOCK_MATCH = {
    "data": [{
        "metadata": {"map": "Ascent", "matchid": "abc123"},
        "players": {"all_players": [
            {"name": "TenZ", "tag": "NA1", "team": "Red"},
            {"name": "s1mple", "tag": "EU1", "team": "Blue"},
        ]},
        "teams": {"red": {"has_won": True}},
        "rounds": [{
            "winning_team": "Red",
            "player_stats": [{
                "player_display_name": "TenZ#NA1",
                "kills": [{
                    "killer_display_name": "TenZ#NA1",
                    "victim_display_name": "s1mple#EU1",
                    "killer_team": "Red",
                    "attacker_team": "Red",
                    "player_locations_on_kill": [
                        {"player_display_name": "TenZ#NA1", "location": {"x": 1200, "y": 3400}},
                        {"player_display_name": "s1mple#EU1", "location": {"x": 2000, "y": 2800}}
                    ]
                }]
            }]
        }]
    }]
}

def test_fetch_matches_returns_list(mocker):
    mock_get = mocker.patch("src.data_loader.requests.get")
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = MOCK_MATCH

    result = fetch_matches("TenZ", "NA1", region="na")
    assert isinstance(result, list)
    assert len(result) == 1

def test_extract_kills_returns_dataframe(mocker):
    kills = extract_kills(MOCK_MATCH["data"])
    assert isinstance(kills, pd.DataFrame)
    assert "x" in kills.columns
    assert "y" in kills.columns
    assert "side" in kills.columns
    assert "result" in kills.columns

def test_extract_kills_includes_both_killer_and_victim():
    kills = extract_kills(MOCK_MATCH["data"])
    assert len(kills) >= 1
    results = kills["result"].unique()
    assert "kill" in results
    # Verify ATK/DEF logic: TenZ is Red team, attacking_team is Red → ATK
    tenz_kill = kills[(kills["player"] == "TenZ#NA1") & (kills["result"] == "kill")]
    assert len(tenz_kill) == 1
    assert tenz_kill.iloc[0]["side"] == "ATK"
    # s1mple is on the defending side
    s1mple_death = kills[(kills["player"] == "s1mple#EU1") & (kills["result"] == "death")]
    assert len(s1mple_death) == 1
    assert s1mple_death.iloc[0]["side"] == "DEF"

def test_extract_kills_empty_matches_returns_empty_df():
    kills = extract_kills([])
    assert isinstance(kills, pd.DataFrame)
    assert len(kills) == 0
    assert "x" in kills.columns


def test_save_kills_creates_csv(tmp_path, monkeypatch):
    import src.data_loader as dl
    monkeypatch.setattr(dl, "PROCESSED_DIR", tmp_path)
    df = extract_kills(MOCK_MATCH["data"])
    path = save_kills(df, "TenZ#NA1")
    assert path.exists()
    assert path.name == "kills_TenZ_NA1.csv"
