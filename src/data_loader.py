"""
Carga de datos desde la HenrikDev API.

Este módulo es la entrada del pipeline: baja partidas crudas de Valorant
y las aplana a un DataFrame de eventos (kills/deaths) con coordenadas.
El resto del pipeline (zones, model) consume ese DataFrame.
"""
import os
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

HENRIK_BASE = "https://api.henrikdev.xyz/valorant"
# Paths absolutos resueltos desde la ubicación de este archivo, no desde el cwd:
# así funciona igual corriendo en local, en los tests o dentro de Docker.
ROOT = Path(__file__).parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"

# Carga las variables del archivo .env al entorno del proceso (HENRIK_API_KEY, etc.).
# Sin esto, os.getenv() devolvería vacío aunque el archivo .env exista.
# Si el .env no está (ej. en Docker, donde las vars se inyectan), no hace nada.
load_dotenv(ROOT / ".env")


def fetch_matches(name: str, tag: str, region: str = "na", count: int = 20) -> list:
    """Baja las últimas `count` partidas de un jugador (name#tag) desde HenrikDev."""
    url = f"{HENRIK_BASE}/v3/matches/{region}/{name}/{tag}"
    params = {"size": count}
    # La API key es opcional: sin ella se usa el tier gratuito (rate limit más bajo).
    # HenrikDev autentica por query param (?api_key=...), no por header HTTP.
    api_key = os.getenv("HENRIK_API_KEY", "")
    if api_key:
        params["api_key"] = api_key

    # Traducimos cualquier fallo de red a un RuntimeError con mensaje claro,
    # para no propagar stacktraces crípticos al resto del pipeline / la API.
    try:
        response = requests.get(url, params=params, timeout=10)
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
    """
    Aplana las partidas crudas a un DataFrame de eventos.

    Estructura anidada de la API: match -> rounds -> player_stats -> kills.
    Cada kill produce DOS filas: una "kill" (posición del que mató) y una
    "death" (posición de la víctima), con lados ATK/DEF opuestos.
    """
    rows = []
    for match in matches:
        map_name = match["metadata"]["map"].lower()
        match_id = match["metadata"]["matchid"]

        for round_data in match.get("rounds", []):
            for ps in round_data.get("player_stats", []):
                for kill in ps.get("kills", []):
                    killer = kill.get("killer_display_name", "")
                    victim = kill.get("victim_display_name", "")
                    # El lado se determina POR KILL, no por partida: en Valorant
                    # los equipos cambian de lado en el halftime. El killer es
                    # atacante si su equipo era el atacante en esa ronda.
                    killer_team = kill.get("killer_team", "")
                    attacking_team = kill.get("attacker_team", "")
                    killer_side = "ATK" if killer_team == attacking_team else "DEF"
                    victim_side = "DEF" if killer_side == "ATK" else "ATK"

                    # Sin la posición del killer no hay dato espacial útil: salteamos.
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
                    # La víctima solo se agrega si también tenemos su ubicación.
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

    # DataFrame vacío PERO con columnas: así el resto del pipeline no rompe
    # cuando un jugador no tiene kills con datos de posición.
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["match_id", "map", "player", "side", "result", "x", "y"]
    )


def save_kills(df: pd.DataFrame, player_tag: str) -> Path:
    """Persiste los eventos a CSV en data/processed/ (crea el dir si no existe)."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / f"kills_{player_tag.replace('#', '_')}.csv"
    df.to_csv(path, index=False)
    return path


def _find_location(locations: list, player: str) -> dict | None:
    """Busca la posición {x, y} de un jugador en la lista de ubicaciones de un kill."""
    for loc in locations:
        if loc.get("player_display_name") == player:
            return loc.get("location")
    return None
