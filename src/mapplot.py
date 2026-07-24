"""Presentación matplotlib: dibuja eventos de kills/muertes sobre la imagen real del mapa."""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from src.mapviz import get_map_asset, game_to_pixel

MODOS = ("scatter", "heatmap")

# Etiqueta legible por tipo de evento, para que el título no mienta.
_ETIQUETA = {"ALL": "eventos", "kill": "kills", "death": "muertes"}


def plot_kills_on_map(kills, map_name, player=None, side="ALL", result="ALL", mode="scatter"):
    """
    Dibuja los eventos de un mapa sobre su minimapa real.

    mode="scatter" → un punto por evento; mode="heatmap" → densidad hexbin.
    Filtra por player (None = todos), side ("ALL"/"ATK"/"DEF") y result
    ("ALL"/"kill"/"death"). Devuelve la Figure.

    Ojo con `result`: extract_kills emite DOS filas por cada kill — una "kill" en la
    posición del asesino y otra "death" en la de la víctima. Con result="ALL" se
    mezclan ambas; usa "kill" para ver dónde matas y "death" para dónde mueres.
    """
    if mode not in MODOS:
        raise ValueError(f"mode debe ser uno de {MODOS}, no {mode!r}")

    image_path, transform = get_map_asset(map_name)
    img = mpimg.imread(image_path)
    h, w = img.shape[0], img.shape[1]

    df = kills[kills["map"] == map_name.lower()]
    if player is not None:
        df = df[df["player"] == player]
    if side != "ALL":
        df = df[df["side"] == side.upper()]
    if result != "ALL":
        df = df[df["result"] == result.lower()]
    df = game_to_pixel(df, transform, size=(w, h))

    clave = "ALL" if result == "ALL" else result.lower()
    etiqueta = _ETIQUETA.get(clave, clave)
    lado = side.upper()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    if df.empty:
        ax.set_title(f"{map_name}: sin {etiqueta} para el filtro")
    elif mode == "heatmap":
        ax.hexbin(df["px"], df["py"], gridsize=30, cmap="inferno", mincnt=1, alpha=0.75)
        ax.set_title(f"{map_name} — densidad de {etiqueta} ({lado})")
    else:
        ax.scatter(df["px"], df["py"], s=12, c="cyan",
                   edgecolors="black", linewidths=0.3, alpha=0.8)
        ax.set_title(f"{map_name} — {etiqueta} ({lado})")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)  # origen arriba-izquierda, como la imagen
    ax.axis("off")
    return fig
