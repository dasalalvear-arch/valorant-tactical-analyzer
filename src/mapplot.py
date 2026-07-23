"""Presentación matplotlib: dibuja kills sobre la imagen real del mapa."""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from src.mapviz import get_map_asset, game_to_pixel


def plot_kills_on_map(kills, map_name, player=None, side="ALL", mode="scatter"):
    """
    Dibuja las kills de un mapa sobre su minimapa real.

    mode="scatter" → un punto por kill; mode="heatmap" → densidad hexbin.
    Filtra por player (None = todos) y side ("ALL"/"ATK"/"DEF"). Devuelve la Figure.
    """
    image_path, transform = get_map_asset(map_name)
    img = mpimg.imread(image_path)
    h, w = img.shape[0], img.shape[1]

    df = kills[kills["map"] == map_name.lower()]
    if player is not None:
        df = df[df["player"] == player]
    if side != "ALL":
        df = df[df["side"] == side.upper()]
    df = game_to_pixel(df, transform, size=(w, h))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    if df.empty:
        ax.set_title(f"{map_name}: sin kills para el filtro")
    elif mode == "heatmap":
        ax.hexbin(df["px"], df["py"], gridsize=30, cmap="inferno", mincnt=1, alpha=0.75)
        ax.set_title(f"{map_name} — densidad de kills ({side})")
    else:
        ax.scatter(df["px"], df["py"], s=12, c="cyan",
                   edgecolors="black", linewidths=0.3, alpha=0.8)
        ax.set_title(f"{map_name} — kills ({side})")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)  # origen arriba-izquierda, como la imagen
    ax.axis("off")
    return fig
