"""
Análisis espacial por zonas: el módulo central del proyecto.

Toma los eventos de kills/deaths (de data_loader) y los mapea a una grilla
de rows×cols celdas, normalizando las coordenadas del juego por mapa.
Luego agrega estadísticas (kills, deaths, kill_rate) por celda y lado.
"""
import warnings

import pandas as pd
import numpy as np

# Rangos reales de coordenadas del juego por mapa: (x_min, x_max, y_min, y_max).
# Las coordenadas de la API NO vienen normalizadas (ascent va de x:2900 a 6300).
MAP_BOUNDS: dict[str, tuple[float, float, float, float]] = {
    "ascent":   (2900, 6300, 1300, 4200),
    "bind":     (3800, 7100, 2200, 5100),
    "haven":    (2200, 6600, 1100, 4400),
    "split":    (3100, 6500, 1500, 4100),
    "fracture": (2500, 6800, 1800, 4600),
    # Mapas restantes — bounds estimados, necesitan calibración con datos reales.
    "pearl":    (2200, 6400, 1400, 4600),
    "lotus":    (2400, 6200, 1200, 4800),
    "sunset":   (2600, 6500, 1300, 4300),
    "abyss":    (2800, 6600, 1500, 4500),
    "icebox":   (2300, 6100, 1100, 4100),
}
# Fallback para mapas desconocidos: rango amplio, la zona queda imprecisa pero no rompe.
DEFAULT_BOUNDS = (0, 10_000, 0, 10_000)


def normalize_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Lleva las coordenadas del juego al rango [0, 1] usando los bounds de cada mapa."""
    df = df.copy()
    # Avisamos si hay mapas sin bounds calibrados: caerán en DEFAULT_BOUNDS.
    unknown_maps = set(df["map"].unique()) - set(MAP_BOUNDS.keys())
    if unknown_maps:
        warnings.warn(
            f"Maps {unknown_maps} not in MAP_BOUNDS. Using DEFAULT_BOUNDS — zone accuracy will be degraded.",
            stacklevel=2,
        )
    # Vectorizado: asignamos los bounds por fila según el mapa, sin loops de Python.
    bounds = df["map"].map(lambda m: MAP_BOUNDS.get(m, DEFAULT_BOUNDS))
    x_min = bounds.map(lambda b: b[0])
    x_max = bounds.map(lambda b: b[1])
    y_min = bounds.map(lambda b: b[2])
    y_max = bounds.map(lambda b: b[3])
    # clip(0, 1) garantiza que nada se salga del rango aunque la coord esté fuera de bounds.
    df["x_norm"] = ((df["x"] - x_min) / (x_max - x_min)).clip(0, 1)
    df["y_norm"] = ((df["y"] - y_min) / (y_max - y_min)).clip(0, 1)
    return df


def assign_zones(df: pd.DataFrame, rows: int = 4, cols: int = 6) -> pd.DataFrame:
    """Discretiza las coordenadas normalizadas en celdas de una grilla rows×cols."""
    df = normalize_coordinates(df)
    # y → fila, x → columna. astype(int) trunca [0,1) a la celda; clip evita el borde 1.0.
    df["zone_row"] = (df["y_norm"] * rows).astype(int).clip(0, rows - 1)
    df["zone_col"] = (df["x_norm"] * cols).astype(int).clip(0, cols - 1)
    return df


def compute_zone_stats(
    df: pd.DataFrame,
    min_events: int = 10,
) -> pd.DataFrame:
    """Agrega kills/deaths/kill_rate por jugador, mapa, celda y lado."""
    # Validamos que vengan las columnas de zona: si faltan, no se llamó assign_zones().
    required = {"player", "map", "zone_row", "zone_col", "side", "result"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}. Call assign_zones() first.")
    group_cols = ["player", "map", "zone_row", "zone_col", "side"]

    # Contamos kills y deaths por separado y luego los unimos.
    kills = df[df["result"] == "kill"].groupby(group_cols).size().reset_index(name="kills")
    deaths = df[df["result"] == "death"].groupby(group_cols).size().reset_index(name="deaths")

    # outer: conservamos celdas que tienen solo kills o solo deaths.
    stats = kills.merge(deaths, on=group_cols, how="outer").fillna(0)
    stats["kills"] = stats["kills"].astype(int)
    stats["deaths"] = stats["deaths"].astype(int)
    stats["total_events"] = stats["kills"] + stats["deaths"]
    stats["kill_rate"] = (stats["kills"] / stats["total_events"]).round(4)
    # Marcamos (no borramos) las celdas con pocos datos: sus stats serían ruido.
    stats["insufficient_sample"] = stats["total_events"] < min_events

    return stats
