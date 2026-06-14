import os
import requests
import pandas as pd
from pathlib import Path

HENRIK_BASE = "https://api.henrikdev.xyz/valorant"
ROOT = Path(__file__).parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"


def fetch_matches(name: str, tag: str, region: str = "na", count: int = 20) -> list:
    url = f"{HENRIK_BASE}/v3/matches/{region}/{name}/{tag}?size={count}"
    headers = {}
    api_key = os.getenv("HENRIK_API_KEY", "")
    if api_key:
        headers["Authorization"] = api_key

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        raise RuntimeError(f"Timeout al contactar HenrikDev API para {name}#{tag}")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"Error HTTP {response.status_code} de HenrikDev API: {e}")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error de red al contactar HenrikDev API: {e}")

    data = response.json()
    return data.get("data", [])


def extract_kills(matches: list) -> pd.DataFrame:
    rows = []
    for match in matches:
        map_name = match["metadata"]["map"].lower()
        match_id = match["metadata"]["matchid"]

        for round_data in match.get("rounds", []):
            for ps in round_data.get("player_stats", []):
                for kill in ps.get("kills", []):
                    killer = kill.get("killer_display_name", "")
                    victim = kill.get("victim_display_name", "")
                    killer_team = kill.get("killer_team", "")
                    attacking_team = kill.get("attacker_team", "")
                    killer_side = "ATK" if killer_team == attacking_team else "DEF"
                    victim_side = "DEF" if killer_side == "ATK" else "ATK"

                    killer_loc = _find_location(kill.get("player_locations_on_kill", []), killer)
                    if not killer_loc:
                        continue

                    rows.append({
                        "match_id": match_id,
                        "map": map_name,
                        "player": killer,
                        "side": killer_side,
                        "result": "kill",
                        "x": killer_loc["x"],
                        "y": killer_loc["y"],
                    })
                    victim_loc = _find_location(kill.get("player_locations_on_kill", []), victim)
                    if victim_loc:
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


def _find_location(locations: list, player: str) -> dict | None:
    for loc in locations:
        if loc.get("player_display_name") == player:
            return loc.get("location")
    return None
