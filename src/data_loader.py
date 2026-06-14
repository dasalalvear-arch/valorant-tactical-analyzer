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

# Modos con estructura táctica (rondas + spike + ATK/DEF) aptos para análisis
# espacial. Los modos arcade (Deathmatch, Escalation, etc.) se descartan: no
# traen player_locations_on_kill ni plant, así que no aportan datos de zona útiles.
TACTICAL_MODES = {"competitive", "unrated", "swiftplay", "premier"}

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

    Usa el array top-level match["kills"] (lista plana de TODOS los kills de la
    partida). Cada kill produce DOS filas: una "kill" (posición del killer, sacada
    de player_locations_on_kill) y una "death" (posición de la víctima, sacada de
    victim_death_location), con lados ATK/DEF opuestos.

    El lado se deriva de quién planta la spike por ronda (ver _round_attackers):
    el equipo que plantó es el atacante de esa ronda.
    """
    rows = []
    for match in matches:
        metadata = match.get("metadata", {})
        # Solo modos tácticos: los arcade no traen posiciones ni lados (ver TACTICAL_MODES).
        if metadata.get("mode", "").lower() not in TACTICAL_MODES:
            continue
        map_name = metadata.get("map", "").lower()
        match_id = metadata.get("matchid", "")
        attackers = _round_attackers(match)

        for kill in match.get("kills", []):
            attacker_team = _attacker_for_round(attackers, kill.get("round", 0))

            # Killer → fila "kill" con su posición en player_locations_on_kill.
            killer = kill.get("killer_display_name", "")
            killer_loc = _find_location(kill.get("player_locations_on_kill", []), killer)
            if killer_loc:
                rows.append({
                    "match_id": match_id,
                    "map": map_name,
                    "player": killer,
                    "side": _side(kill.get("killer_team", ""), attacker_team),
                    "result": "kill",
                    "x": killer_loc["x"],
                    "y": killer_loc["y"],
                })

            # Víctima → fila "death" con su posición de muerte.
            victim_loc = kill.get("victim_death_location")
            if victim_loc:
                rows.append({
                    "match_id": match_id,
                    "map": map_name,
                    "player": kill.get("victim_display_name", ""),
                    "side": _side(kill.get("victim_team", ""), attacker_team),
                    "result": "death",
                    "x": victim_loc["x"],
                    "y": victim_loc["y"],
                })

    # DataFrame vacío PERO con columnas: así el resto del pipeline no rompe
    # cuando una partida no tiene kills con datos de posición.
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["match_id", "map", "player", "side", "result", "x", "y"]
    )


def _round_attackers(match: dict) -> dict:
    """
    Determina el equipo atacante por ronda a partir de los plants de spike.

    El que planta la spike es el atacante de esa ronda. El atacante cambia una
    sola vez (halftime swap), así que registramos el primer plantador y el round
    donde aparece el otro equipo (el swap). Devuelve
    {"first": team, "other": team, "swap": round_idx|None}, o {} si no hay plants.

    Limitación: el modelo de un único swap no cubre overtime (donde el lado
    alterna cada ronda); en esos casos el lado queda aproximado.
    """
    plants = {}  # {round_idx: equipo que plantó la spike}
    for i, r in enumerate(match.get("rounds", [])):
        pe = r.get("plant_events") if isinstance(r, dict) else None
        pb = pe.get("planted_by") if isinstance(pe, dict) else None
        team = pb.get("team") if isinstance(pb, dict) else None
        if team:
            plants[i] = team
    if not plants:
        return {}
    ordered = sorted(plants.items())
    first = ordered[0][1]
    other = "Blue" if first == "Red" else "Red"
    swap = next((rnd for rnd, team in ordered if team != first), None)
    return {"first": first, "other": other, "swap": swap}


def _attacker_for_round(attackers: dict, round_idx: int) -> str:
    """Equipo atacante de una ronda según el modelo de un único swap de halftime."""
    if not attackers:
        return ""  # sin datos de plant: lado indeterminable
    if attackers["swap"] is None or round_idx < attackers["swap"]:
        return attackers["first"]
    return attackers["other"]


def _side(team: str, attacker_team: str) -> str:
    """ATK si el equipo es el atacante de la ronda; DEF en caso contrario."""
    return "ATK" if team and team == attacker_team else "DEF"


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
