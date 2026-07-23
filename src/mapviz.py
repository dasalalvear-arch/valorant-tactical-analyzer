"""
Assets y transformación de coordenadas para dibujar kills sobre el mapa real.

Núcleo puro (sin matplotlib): baja/cachea la imagen del minimapa y las constantes
de transformación de valorant-api, y convierte coordenadas del juego a píxeles.
"""
import pandas as pd


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
