"""Presentación matplotlib: dibuja eventos de kills/muertes sobre la imagen real del mapa."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from src.mapviz import get_map_asset, game_to_pixel

MODOS = ("scatter", "heatmap")

# Etiqueta legible por tipo de evento, para que el título no mienta.
_ETIQUETA = {"ALL": "eventos", "kill": "kills", "death": "muertes"}

# Azul/naranja en vez de verde/rojo: se distinguen también con daltonismo rojo-verde.
_COLOR_EVENTO = {"kill": "blue", "death": "darkorange"}

_MARGEN_RECORTE = 12   # px de aire alrededor del contenido del minimapa


def _encuadre(img):
    """
    Devuelve (x0, x1, y0, y1) del contenido real del minimapa.

    Los PNG de valorant-api traen padding transparente alrededor (en ascent el mapa
    ocupa 949x889 de una imagen de 1024x1024). Encuadrar al contenido lo centra y lo
    agranda sin tocar las coordenadas de los puntos: solo cambia la ventana de vista.
    """
    alto, ancho = img.shape[0], img.shape[1]
    if img.ndim != 3 or img.shape[2] < 4:
        return 0, ancho, 0, alto            # sin canal alfa: imagen completa
    visible = img[:, :, 3] > 0
    filas = np.where(visible.any(axis=1))[0]
    cols = np.where(visible.any(axis=0))[0]
    if filas.size == 0 or cols.size == 0:
        return 0, ancho, 0, alto
    m = _MARGEN_RECORTE
    return (max(int(cols[0]) - m, 0), min(int(cols[-1]) + m, ancho),
            max(int(filas[0]) - m, 0), min(int(filas[-1]) + m, alto))


def plot_kills_on_map(kills, map_name, player=None, side="ALL", result="ALL", mode="scatter"):
    """
    Dibuja los eventos de un mapa sobre su minimapa real.

    mode="scatter" → un punto por evento; mode="heatmap" → densidad hexbin con
    barra de color. Filtra por player (None = todos), side ("ALL"/"ATK"/"DEF") y
    result ("ALL"/"kill"/"death"). Devuelve la Figure.

    Ojo con `result`: extract_kills emite DOS filas por cada kill — una "kill" en la
    posición del asesino y otra "death" en la de la víctima. Con result="ALL" el
    scatter las colorea por separado y añade leyenda; con "kill" o "death" filtra.
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
        # El color del hexágono = cuántos eventos cayeron en esa celda; sin la barra
        # de color el degradado no se puede interpretar.
        hb = ax.hexbin(df["px"], df["py"], gridsize=20, cmap="plasma", mincnt=1, alpha=0.75)
        fig.colorbar(hb, ax=ax, shrink=0.7, label=f"nº de {etiqueta} por celda")
        ax.set_title(f"{map_name} — densidad de {etiqueta} ({lado})")
    elif result == "ALL":
        # Kills y muertes mezcladas: pintarlas del mismo color serían puntos
        # idénticos con significados opuestos, así que van por color + leyenda.
        for tipo, color in _COLOR_EVENTO.items():
            sub = df[df["result"] == tipo]
            if sub.empty:
                continue
            ax.scatter(sub["px"], sub["py"], s=20, c=color, edgecolors="black",
                       linewidths=0.3, alpha=0.8, label=_ETIQUETA[tipo])
        if ax.collections:
            ax.legend(loc="upper right", framealpha=0.9)
        ax.set_title(f"{map_name} — {etiqueta} ({lado})")
    else:
        ax.scatter(df["px"], df["py"], s=20, c=_COLOR_EVENTO.get(clave, "blue"),
                   edgecolors="black", linewidths=0.3, alpha=0.8)
        ax.set_title(f"{map_name} — {etiqueta} ({lado})")

    x0, x1, y0, y1 = _encuadre(img)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)        # invertido: origen arriba-izquierda, como la imagen
    ax.set_aspect("equal")
    ax.axis("off")
    return fig
