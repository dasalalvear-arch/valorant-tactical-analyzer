"""
Assets y transformación de coordenadas para dibujar kills sobre el mapa real.

Núcleo puro (sin matplotlib): baja/cachea la imagen del minimapa y las constantes
de transformación de valorant-api, y convierte coordenadas del juego a píxeles.
"""
import json
from pathlib import Path

import pandas as pd
import requests

VALORANT_API_MAPS = "https://valorant-api.com/v1/maps"
MAPS_DIR = Path(__file__).parent.parent / "data" / "maps"
TRANSFORMS_PATH = MAPS_DIR / "transforms.json"


def _get(url: str):
    """
    GET con mensaje claro si no hay conexión.

    Solo la primera descarga (constantes + imagen) necesita red; después todo sale
    del caché de data/maps/. `from None` evita que la excepción encadenada de
    requests tape el mensaje útil con un muro de urllib3.
    """
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp
    except requests.exceptions.RequestException:
        raise RuntimeError(
            "No se pudieron descargar los assets del mapa. Se necesita conexión la "
            "primera vez; después se usa el caché de data/maps/."
        ) from None


def fetch_map_transforms() -> dict:
    """
    Baja de valorant-api las constantes de transformación por mapa.

    Devuelve {nombre_en_minúsculas: {xMultiplier, yMultiplier, xScalarToAdd,
    yScalarToAdd, displayIcon}}. Descarta mapas de práctica/TDM, que traen los
    multiplicadores en 0 (no tienen transformación válida).
    """
    resp = _get(VALORANT_API_MAPS)
    out = {}
    for m in resp.json()["data"]:
        if not m["xMultiplier"] or not m["yMultiplier"]:
            continue
        out[m["displayName"].lower()] = {
            "xMultiplier": m["xMultiplier"],
            "yMultiplier": m["yMultiplier"],
            "xScalarToAdd": m["xScalarToAdd"],
            "yScalarToAdd": m["yScalarToAdd"],
            "displayIcon": m["displayIcon"],
        }
    return out


def _load_transforms() -> dict:
    """Lee transforms.json del caché; lo genera desde la API la primera vez."""
    if not TRANSFORMS_PATH.exists():
        MAPS_DIR.mkdir(parents=True, exist_ok=True)
        TRANSFORMS_PATH.write_text(json.dumps(fetch_map_transforms(), indent=2))
    return json.loads(TRANSFORMS_PATH.read_text())


def get_map_asset(map_name: str) -> tuple[Path, dict]:
    """
    Devuelve (ruta_png, transform) para un mapa, usando caché.

    Descarga la imagen del minimapa la primera vez. Mapa sin transformación válida
    (p. ej. mapas de práctica) → ValueError.
    """
    map_name = map_name.lower()
    transforms = _load_transforms()
    if map_name not in transforms:
        raise ValueError(
            f"Mapa '{map_name}' sin transformación válida en valorant-api "
            "(si es un mapa nuevo, borra data/maps/transforms.json para refrescar el caché)."
        )
    transform = transforms[map_name]

    image_path = MAPS_DIR / f"{map_name}.png"
    if not image_path.exists():
        MAPS_DIR.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(_get(transform["displayIcon"]).content)
    return image_path, transform


def game_to_pixel(df: pd.DataFrame, transform: dict, size: tuple[int, int]) -> pd.DataFrame:
    """
    Añade columnas px, py (píxeles de la imagen) a partir de las coords del juego.

    Transformación oficial de Valorant (normalizado [0,1] × tamaño), con swap de
    ejes: el X del minimapa usa la 'y' del juego, y el yMultiplier negativo hace el
    flip vertical. Función pura: devuelve una copia, no muta la entrada.
    """
    w, h = size
    nx = df["y"] * transform["xMultiplier"] + transform["xScalarToAdd"]
    ny = df["x"] * transform["yMultiplier"] + transform["yScalarToAdd"]
    out = df.copy()
    out["px"] = nx * w
    out["py"] = ny * h
    return out
