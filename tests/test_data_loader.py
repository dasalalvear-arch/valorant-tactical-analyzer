import pandas as pd
import pytest
import requests
from src.data_loader import fetch_matches, extract_kills, save_kills

# Estructura REAL de la API HenrikDev v3 (verificada contra datos en vivo):
# - Los eventos de kill viven en match["kills"] (lista plana), NO en player_stats[].kills (que es un int).
# - No hay "attacker_team": el lado ATK/DEF se deriva de quién planta la spike por ronda.
# - La posición del killer está en player_locations_on_kill; la de la víctima en victim_death_location.
MOCK_MATCH = {
    "data": [{
        "metadata": {"map": "Ascent", "matchid": "abc123", "mode": "Swiftplay"},
        "kills": [
            {
                "round": 0,
                "killer_display_name": "TenZ#NA1",
                "killer_team": "Red",
                "victim_display_name": "s1mple#EU1",
                "victim_team": "Blue",
                "victim_death_location": {"x": 5000, "y": 3000},
                "player_locations_on_kill": [
                    {"player_display_name": "TenZ#NA1", "player_team": "Red", "location": {"x": 3200, "y": 2000}},
                    {"player_display_name": "s1mple#EU1", "player_team": "Blue", "location": {"x": 5000, "y": 3000}},
                ],
            },
            {
                "round": 1,
                "killer_display_name": "s1mple#EU1",
                "killer_team": "Blue",
                "victim_display_name": "TenZ#NA1",
                "victim_team": "Red",
                "victim_death_location": {"x": 4000, "y": 2500},
                "player_locations_on_kill": [
                    {"player_display_name": "s1mple#EU1", "player_team": "Blue", "location": {"x": 4200, "y": 2600}},
                    {"player_display_name": "TenZ#NA1", "player_team": "Red", "location": {"x": 4000, "y": 2500}},
                ],
            },
        ],
        # Round 0: planta Red → Red ataca. Round 1: planta Blue → swap, Blue ataca.
        "rounds": [
            {"bomb_planted": True, "plant_events": {"planted_by": {"team": "Red"}}},
            {"bomb_planted": True, "plant_events": {"planted_by": {"team": "Blue"}}},
        ],
    }]
}


def test_fetch_matches_returns_list(mocker):
    mock_get = mocker.patch("src.data_loader.requests.get")
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = MOCK_MATCH

    result = fetch_matches("TenZ", "NA1", region="na")
    assert isinstance(result, list)
    assert len(result) == 1


def test_fetch_matches_redacts_api_key_in_errors(mocker, monkeypatch):
    # Como la key viaja en la URL (?api_key=), un error HTTP de requests la incluye
    # en su mensaje. fetch_matches debe ocultarla antes de propagar el error.
    monkeypatch.setenv("HENRIK_API_KEY", "HDEV-supersecret123")
    resp = mocker.patch("src.data_loader.requests.get").return_value
    resp.status_code = 404
    resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "404 Not Found for url: https://api.henrikdev.xyz/v3/matches/na/X/1?size=20&api_key=HDEV-supersecret123"
    )
    with pytest.raises(RuntimeError) as exc:
        fetch_matches("X", "1", region="na")
    assert "HDEV-supersecret123" not in str(exc.value)
    assert "api_key=***" in str(exc.value)


def test_extract_kills_returns_dataframe():
    kills = extract_kills(MOCK_MATCH["data"])
    assert isinstance(kills, pd.DataFrame)
    for col in ("match_id", "map", "player", "side", "result", "x", "y"):
        assert col in kills.columns


def test_extract_kills_includes_both_killer_and_victim():
    kills = extract_kills(MOCK_MATCH["data"])
    # 2 kills → cada uno genera fila de kill (killer) y death (víctima)
    assert len(kills) == 4
    assert set(kills["result"].unique()) == {"kill", "death"}


def test_extract_kills_derives_side_from_plant():
    kills = extract_kills(MOCK_MATCH["data"])
    # Round 0: Red plantó → Red ataca. TenZ (Red) mató → ATK; s1mple (Blue) murió → DEF.
    tenz_kill = kills[(kills["player"] == "TenZ#NA1") & (kills["result"] == "kill")]
    assert tenz_kill.iloc[0]["side"] == "ATK"
    s1mple_death = kills[(kills["player"] == "s1mple#EU1") & (kills["result"] == "death")]
    assert s1mple_death.iloc[0]["side"] == "DEF"
    # Round 1: swap, Blue ataca. s1mple (Blue) mató → ATK; TenZ (Red) murió → DEF.
    s1mple_kill = kills[(kills["player"] == "s1mple#EU1") & (kills["result"] == "kill")]
    assert s1mple_kill.iloc[0]["side"] == "ATK"
    tenz_death = kills[(kills["player"] == "TenZ#NA1") & (kills["result"] == "death")]
    assert tenz_death.iloc[0]["side"] == "DEF"


def test_extract_kills_uses_correct_locations():
    kills = extract_kills(MOCK_MATCH["data"])
    # Killer: posición tomada de player_locations_on_kill.
    tenz_kill = kills[(kills["player"] == "TenZ#NA1") & (kills["result"] == "kill")].iloc[0]
    assert (tenz_kill["x"], tenz_kill["y"]) == (3200, 2000)
    # Víctima: posición tomada de victim_death_location.
    s1mple_death = kills[(kills["player"] == "s1mple#EU1") & (kills["result"] == "death")].iloc[0]
    assert (s1mple_death["x"], s1mple_death["y"]) == (5000, 3000)


def test_extract_kills_ignores_non_tactical_modes():
    # Modos arcade (Team Deathmatch, Escalation, etc.) no traen player_locations_on_kill
    # ni spike/plant; deben descartarse para no ensuciar el análisis espacial.
    tdm = {"data": [{
        "metadata": {"map": "Drift", "matchid": "tdm1", "mode": "Team Deathmatch"},
        "kills": [{
            "round": 0,
            "killer_display_name": "X#1", "killer_team": "Red",
            "victim_display_name": "Y#2", "victim_team": "Blue",
            "victim_death_location": {"x": 100, "y": 100},
            "player_locations_on_kill": [],
        }],
        "rounds": [],
    }]}
    kills = extract_kills(tdm["data"])
    assert len(kills) == 0


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
