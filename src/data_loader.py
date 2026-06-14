import os
import requests
import pandas as pd
from pathlib import Path

HENRIK_BASE = "https://api.henrikdev.xyz/valorant"
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def fetch_matches(name: str, tag: str, region: str = "na", count: int = 20) -> list:
    url = f"{HENRIK_BASE}/v3/matches/{region}/{name}/{tag}?size={count}"
    headers = {}
    api_key = os.getenv("HENRIK_API_KEY", "")
    if api_key:
        headers["Authorization"] = api_key

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    data = response.json()
    return data.get("data", [])


def extract_kills(matches: list) -> pd.DataFrame:
    rows = []
    for match in matches:
        map_name = match["metadata"]["map"].lower()
        match_id = match["metadata"]["matchid"]

        for round_data in match.get("rounds", []):
            attacking_team = round_data.get("attacker_team", "")
            for ps in round_data.get("player_stats", []):
                player = ps["player_display_name"]
                player_team = _get_player_team(match, player)
                side = "ATK" if player_team == attacking_team else "DEF"

                for kill in ps.get("kills", []):
                    killer = kill.get("killer_display_name", "")
                    victim = kill.get("victim_display_name", "")

                    killer_loc = _find_location(kill["player_locations_on_kill"], killer)
                    if not killer_loc:
                        continue

                    rows.append({
                        "match_id": match_id,
                        "map": map_name,
                        "player": killer,
                        "side": side,
                        "result": "kill",
                        "x": killer_loc["x"],
                        "y": killer_loc["y"],
                    })

                    victim_loc = _find_location(kill["player_locations_on_kill"], victim)
                    if victim_loc:
                        victim_side = "DEF" if side == "ATK" else "ATK"
                        rows.append({
                            "match_id": match_id,
                            "map": map_name,
                            "player": victim,
                            "side": victim_side,
                            "result": "death",
                            "x": victim_loc["x"],
                            "y": victim_loc["y"],
                        })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["match_id", "map", "player", "side", "result", "x", "y"]
    )


def save_kills(df: pd.DataFrame, player_tag: str) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / f"kills_{player_tag.replace('#', '_')}.csv"
    df.to_csv(path, index=False)
    return path


def _get_player_team(match: dict, player: str) -> str:
    for p in match.get("players", {}).get("all_players", []):
        if p.get("name", "") + "#" + p.get("tag", "") == player:
            return p.get("team", "")
    return ""


def _find_location(locations: list, player: str) -> dict | None:
    for loc in locations:
        if loc.get("player_display_name") == player:
            return loc.get("location")
    return None
