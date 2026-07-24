# Kill Map Overlay — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dibujar las kills reales sobre la imagen real de cada mapa (scatter y heatmap de densidad, alternables) desde Jupyter, con el núcleo reutilizable en `src/`.

**Architecture:** Núcleo puro en `src/mapviz.py` (baja/cachea imagen + constantes de valorant-api y convierte coords del juego → píxel). Capa de presentación matplotlib en `src/mapplot.py`. El notebook `notebooks/explora.ipynb` añade una celda con dropdowns/toggle (ipywidgets) que llama a esa capa. No se toca `zones.py`/`MAP_BOUNDS`.

**Tech Stack:** Python 3.11, pandas, requests, matplotlib (ya dependencias), ipywidgets (nueva, dev). Tests con pytest + pytest-mock (sin red real).

## Global Constraints

- Comentarios y docstrings en **español** (convención del repo).
- Los tests **no** hacen red real: `requests.get` se mockea con `pytest-mock`.
- Se ejecuta pytest desde la raíz del repo; los imports son `from src.xxx import ...`.
- **No** modificar `src/zones.py` ni `MAP_BOUNDS`.
- Única dependencia nueva permitida: `ipywidgets` (a `requirements-dev.txt`).
- Transformación oficial de Valorant con **swap de ejes** (X del minimapa usa `y` del juego) y `yMultiplier` negativo (flip vertical).
- Caché en `data/maps/`: `*.png` **no** se versiona (va a `.gitignore`); `transforms.json` **sí** se versiona.
- Commits con el trailer del repo: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

## File Structure

- `src/mapviz.py` (crear) — núcleo puro: `fetch_map_transforms`, `get_map_asset`, `game_to_pixel`. Sin matplotlib.
- `src/mapplot.py` (crear) — presentación matplotlib: `plot_kills_on_map`. Importa `mapviz`.
- `tests/test_mapviz.py` (crear) — tests del núcleo.
- `tests/test_mapplot.py` (crear) — test del render (backend Agg).
- `.gitignore` (modificar) — ignorar `data/maps/*.png`.
- `requirements-dev.txt` (modificar) — añadir `ipywidgets`.
- `notebooks/explora.ipynb` (modificar) — celda de visualización con widgets.

---

## Task 1: Transformación coords → píxel (`game_to_pixel`)

**Files:**
- Create: `src/mapviz.py`
- Test: `tests/test_mapviz.py`

**Interfaces:**
- Consumes: nada.
- Produces: `game_to_pixel(df: pd.DataFrame, transform: dict, size: tuple[int, int]) -> pd.DataFrame` — devuelve una copia del df con columnas `px`, `py` (float, píxeles). `transform` tiene claves `xMultiplier, yMultiplier, xScalarToAdd, yScalarToAdd`. `size` es `(ancho, alto)` de la imagen.

- [ ] **Step 1: Write the failing test**

Crear `tests/test_mapviz.py`:

```python
import pandas as pd
import pytest

from src.mapviz import game_to_pixel


def test_game_to_pixel_aplica_transformacion_con_swap_de_ejes():
    # px se calcula desde 'y', py desde 'x'; yMultiplier negativo hace el flip.
    df = pd.DataFrame({"x": [100], "y": [200]})
    transform = {
        "xMultiplier": 0.001, "yMultiplier": -0.001,
        "xScalarToAdd": 0.5, "yScalarToAdd": 0.5,
    }
    out = game_to_pixel(df, transform, size=(1000, 1000))
    # nx = 200*0.001 + 0.5 = 0.7 -> px = 700 ; ny = 100*(-0.001) + 0.5 = 0.4 -> py = 400
    assert out["px"].iloc[0] == pytest.approx(700.0)
    assert out["py"].iloc[0] == pytest.approx(400.0)
    # No muta el df original.
    assert "px" not in df.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mapviz.py::test_game_to_pixel_aplica_transformacion_con_swap_de_ejes -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'src.mapviz'`.

- [ ] **Step 3: Write minimal implementation**

Crear `src/mapviz.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mapviz.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mapviz.py tests/test_mapviz.py
git commit -m "feat: game_to_pixel — transformación coords del juego a píxel del minimapa" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Descarga + caché de assets (`fetch_map_transforms`, `get_map_asset`)

**Files:**
- Modify: `src/mapviz.py`
- Test: `tests/test_mapviz.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: nada de tareas previas.
- Produces:
  - `fetch_map_transforms() -> dict` — `{nombre_minúsculas: {xMultiplier, yMultiplier, xScalarToAdd, yScalarToAdd, displayIcon}}`. Descarta mapas con multiplicadores en 0.
  - `get_map_asset(map_name: str) -> tuple[pathlib.Path, dict]` — `(ruta_png, transform)`. Descarga la imagen la primera vez, luego usa caché. Mapa sin transformación → `ValueError`.

- [ ] **Step 1: Write the failing test**

Añadir a `tests/test_mapviz.py` (arriba, junto a los imports):

```python
import src.mapviz as mapviz
from src.mapviz import get_map_asset
```

Y añadir el test:

```python
def test_get_map_asset_cachea_imagen_y_no_redescarga(tmp_path, mocker):
    mocker.patch.object(mapviz, "MAPS_DIR", tmp_path)
    mocker.patch.object(mapviz, "TRANSFORMS_PATH", tmp_path / "transforms.json")
    mocker.patch.object(mapviz, "fetch_map_transforms", return_value={
        "ascent": {
            "xMultiplier": 0.00007, "yMultiplier": -0.00007,
            "xScalarToAdd": 0.813895, "yScalarToAdd": 0.573242,
            "displayIcon": "http://img/ascent.png",
        }
    })
    get = mocker.patch("src.mapviz.requests.get")
    get.return_value.content = b"PNGDATA"
    get.return_value.raise_for_status = mocker.Mock()

    path1, transform = get_map_asset("Ascent")           # case-insensitive
    assert path1.read_bytes() == b"PNGDATA"
    assert transform["xMultiplier"] == 0.00007
    assert get.call_count == 1                            # una descarga de imagen

    path2, _ = get_map_asset("ascent")
    assert path2 == path1
    assert get.call_count == 1                            # segunda vez: caché, no re-descarga


def test_get_map_asset_mapa_invalido_lanza_valueerror(tmp_path, mocker):
    mocker.patch.object(mapviz, "MAPS_DIR", tmp_path)
    mocker.patch.object(mapviz, "TRANSFORMS_PATH", tmp_path / "transforms.json")
    mocker.patch.object(mapviz, "fetch_map_transforms", return_value={})
    with pytest.raises(ValueError):
        get_map_asset("the range")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mapviz.py::test_get_map_asset_cachea_imagen_y_no_redescarga -v`
Expected: FAIL con `AttributeError: module 'src.mapviz' has no attribute 'MAPS_DIR'`.

- [ ] **Step 3: Write minimal implementation**

En `src/mapviz.py`, reemplazar la cabecera de imports y añadir las constantes y funciones (dejando `game_to_pixel` como está):

```python
import json
from pathlib import Path

import pandas as pd
import requests

VALORANT_API_MAPS = "https://valorant-api.com/v1/maps"
MAPS_DIR = Path(__file__).parent.parent / "data" / "maps"
TRANSFORMS_PATH = MAPS_DIR / "transforms.json"


def fetch_map_transforms() -> dict:
    """
    Baja de valorant-api las constantes de transformación por mapa.

    Devuelve {nombre_en_minúsculas: {xMultiplier, yMultiplier, xScalarToAdd,
    yScalarToAdd, displayIcon}}. Descarta mapas de práctica/TDM, que traen los
    multiplicadores en 0 (no tienen transformación válida).
    """
    resp = requests.get(VALORANT_API_MAPS, timeout=30)
    resp.raise_for_status()
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
        raise ValueError(f"Mapa '{map_name}' sin transformación válida en valorant-api.")
    transform = transforms[map_name]

    image_path = MAPS_DIR / f"{map_name}.png"
    if not image_path.exists():
        MAPS_DIR.mkdir(parents=True, exist_ok=True)
        resp = requests.get(transform["displayIcon"], timeout=30)
        resp.raise_for_status()
        image_path.write_bytes(resp.content)
    return image_path, transform
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_mapviz.py -v`
Expected: PASS (los 3 tests).

- [ ] **Step 5: Ignorar los PNG en git**

Añadir al final de `.gitignore`:

```
data/maps/*.png
```

- [ ] **Step 6: Commit**

```bash
git add src/mapviz.py tests/test_mapviz.py .gitignore
git commit -m "feat: get_map_asset — descarga y cachea minimapa + constantes de valorant-api" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Render sobre el mapa (`plot_kills_on_map`)

**Files:**
- Create: `src/mapplot.py`
- Test: `tests/test_mapplot.py`

**Interfaces:**
- Consumes: `get_map_asset(map_name) -> (Path, dict)` y `game_to_pixel(df, transform, size)` de `src/mapviz.py`.
- Produces: `plot_kills_on_map(kills: pd.DataFrame, map_name: str, player=None, side="ALL", mode="scatter") -> matplotlib.figure.Figure`. `kills` tiene columnas `map, player, side, result, x, y`. `mode` ∈ {"scatter", "heatmap"}.

- [ ] **Step 1: Write the failing test**

Crear `tests/test_mapplot.py`:

```python
import matplotlib
matplotlib.use("Agg")  # sin display en tests

import numpy as np
import pandas as pd
import matplotlib.image as mpimg

from src import mapplot


def _kills_demo():
    return pd.DataFrame({
        "map": ["ascent", "ascent", "bind"],
        "player": ["P#EU", "P#EU", "Q#EU"],
        "side": ["ATK", "DEF", "ATK"],
        "result": ["kill", "death", "kill"],
        "x": [10, 50, 10],
        "y": [10, 50, 10],
    })


def test_plot_kills_on_map_scatter_devuelve_figura(tmp_path, mocker):
    img_path = tmp_path / "ascent.png"
    mpimg.imsave(img_path, np.zeros((100, 100, 3), dtype=np.uint8))
    transform = {"xMultiplier": 0.001, "yMultiplier": -0.001,
                 "xScalarToAdd": 0.5, "yScalarToAdd": 0.5}
    mocker.patch("src.mapplot.get_map_asset", return_value=(img_path, transform))

    fig = mapplot.plot_kills_on_map(_kills_demo(), "ascent", mode="scatter")
    assert fig is not None
    assert len(fig.axes) >= 1


def test_plot_kills_on_map_sin_datos_no_rompe(tmp_path, mocker):
    img_path = tmp_path / "ascent.png"
    mpimg.imsave(img_path, np.zeros((100, 100, 3), dtype=np.uint8))
    transform = {"xMultiplier": 0.001, "yMultiplier": -0.001,
                 "xScalarToAdd": 0.5, "yScalarToAdd": 0.5}
    mocker.patch("src.mapplot.get_map_asset", return_value=(img_path, transform))

    # Filtro que no matchea ningún jugador -> figura vacía, sin excepción.
    fig = mapplot.plot_kills_on_map(_kills_demo(), "ascent", player="NADIE", mode="heatmap")
    assert fig is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mapplot.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'src.mapplot'`.

- [ ] **Step 3: Write minimal implementation**

Crear `src/mapplot.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_mapplot.py -v`
Expected: PASS (los 2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/mapplot.py tests/test_mapplot.py
git commit -m "feat: plot_kills_on_map — scatter/heatmap de kills sobre el minimapa" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Celda interactiva en el notebook + dependencia ipywidgets

**Files:**
- Modify: `requirements-dev.txt`
- Modify: `notebooks/explora.ipynb`

**Interfaces:**
- Consumes: `plot_kills_on_map(kills, map_name, player, side, mode)` de `src/mapplot.py`. Reutiliza el DataFrame `kills` que ya produce la celda de carga del notebook.
- Produces: nada que consuman otras tareas (capa final).

- [ ] **Step 1: Añadir ipywidgets a las dependencias de desarrollo**

Añadir al final de `requirements-dev.txt`:

```
ipywidgets==8.1.5
```

- [ ] **Step 2: Instalar la dependencia**

Run: `pip install ipywidgets==8.1.5`
Expected: instala ipywidgets sin errores.

- [ ] **Step 3: Añadir una celda nueva al final de `notebooks/explora.ipynb`**

Añadir una celda de código con este contenido (usa el `kills` ya cargado en las celdas anteriores):

```python
import ipywidgets as widgets
from IPython.display import display
from src.mapplot import plot_kills_on_map

mapas = sorted(kills["map"].unique())
jugadores = ["(todos)"] + sorted(kills["player"].unique())

@widgets.interact(
    mapa=widgets.Dropdown(options=mapas, description="Mapa"),
    jugador=widgets.Dropdown(options=jugadores, description="Jugador"),
    lado=widgets.Dropdown(options=["ALL", "ATK", "DEF"], description="Lado"),
    modo=widgets.ToggleButtons(options=["scatter", "heatmap"], description="Modo"),
)
def _ver(mapa, jugador, lado, modo):
    player = None if jugador == "(todos)" else jugador
    plot_kills_on_map(kills, mapa, player=player, side=lado, mode=modo)
    import matplotlib.pyplot as plt
    plt.show()
```

- [ ] **Step 4: Verificación manual**

Abrir `notebooks/explora.ipynb` en JupyterLab, ejecutar todas las celdas (`Run All`) y comprobar:
- Aparecen los dropdowns Mapa/Jugador/Lado y el toggle scatter/heatmap.
- Al cambiar el toggle, la vista alterna entre puntos y densidad.

**Nota de calibración (importante para datos reales):** con `USA_DATOS_REALES = True` y una `HENRIK_API_KEY` en `.env`, los kills reales deberían caer sobre las zonas correctas del mapa. Los **datos demo** de la celda de carga usan rangos de `MAP_BOUNDS` (estimados), que no están en el espacio de coordenadas de valorant-api, así que con datos demo los puntos pueden caer fuera del mapa — es esperado. La validación geográfica real (¿los kills caen donde deben?) se hace con datos reales; si hay desalineación, es el ajuste de calibración que anticipaba el spec.

- [ ] **Step 5: Commit**

```bash
git add requirements-dev.txt notebooks/explora.ipynb
git commit -m "feat: celda interactiva de kills sobre el mapa (ipywidgets)" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Notas de ejecución

- Correr toda la suite al final: `pytest tests/ -v` — deben pasar los tests nuevos y los existentes.
- La primera llamada a `get_map_asset` (fuera de tests) hará **descargas** desde valorant-api (imágenes ~1 MB) y creará `data/maps/`. Confirmar con el usuario antes de la primera ejecución con red real.
