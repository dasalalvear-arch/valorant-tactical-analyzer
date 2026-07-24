import matplotlib
matplotlib.use("Agg")  # sin display en tests

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from src import mapplot


@pytest.fixture(autouse=True)
def _cierra_figuras():
    """Cierra las figuras al terminar cada test: matplotlib avisa a partir de 20 abiertas."""
    yield
    plt.close("all")


@pytest.fixture
def mapa_mockeado(tmp_path, mocker):
    """Parchea get_map_asset con una imagen 100x100 y un transform conocido."""
    img_path = tmp_path / "ascent.png"
    mpimg.imsave(img_path, np.zeros((100, 100, 3), dtype=np.uint8))
    transform = {"xMultiplier": 0.001, "yMultiplier": -0.001,
                 "xScalarToAdd": 0.5, "yScalarToAdd": 0.5}
    mocker.patch("src.mapplot.get_map_asset", return_value=(img_path, transform))
    return img_path, transform


def _kills_demo():
    # En ascent: 1 kill y 1 death del mismo jugador -> sirve para probar el filtro result.
    return pd.DataFrame({
        "map": ["ascent", "ascent", "bind"],
        "player": ["P#EU", "P#EU", "Q#EU"],
        "side": ["ATK", "DEF", "ATK"],
        "result": ["kill", "death", "kill"],
        "x": [10, 50, 10],
        "y": [10, 50, 10],
    })


def test_plot_kills_on_map_scatter_devuelve_figura(mapa_mockeado):
    fig = mapplot.plot_kills_on_map(_kills_demo(), "ascent", mode="scatter")
    assert fig is not None
    assert len(fig.axes) >= 1


def test_plot_kills_on_map_sin_datos_no_rompe(mapa_mockeado):
    # Filtro que no matchea ningún jugador -> figura vacía, sin excepción.
    fig = mapplot.plot_kills_on_map(_kills_demo(), "ascent", player="NADIE", mode="heatmap")
    assert fig is not None


def test_plot_kills_on_map_heatmap_con_datos_dibuja(mapa_mockeado):
    fig = mapplot.plot_kills_on_map(_kills_demo(), "ascent", mode="heatmap")
    ax = fig.axes[0]
    assert len(ax.collections) >= 1          # la rama heatmap (hexbin) corrió con datos
    assert "densidad" in ax.get_title()


def test_plot_kills_on_map_filtra_por_player_y_side(mapa_mockeado):
    df = pd.DataFrame({
        "map": ["ascent"] * 4,
        "player": ["A#EU", "A#EU", "B#EU", "B#EU"],
        "side": ["ATK", "DEF", "ATK", "DEF"],
        "result": ["kill"] * 4,
        "x": [10, 20, 30, 40],
        "y": [10, 20, 30, 40],
    })
    fig = mapplot.plot_kills_on_map(df, "ascent", player="A#EU", side="ATK", mode="scatter")
    ax = fig.axes[0]
    # player=A#EU y side=ATK deja exactamente 1 kill; un filtro no-op dejaría más.
    assert len(ax.collections[0].get_offsets()) == 1
    assert "ATK" in ax.get_title()


def test_plot_kills_on_map_filtra_por_result(mapa_mockeado):
    # extract_kills emite 2 filas por kill (una "kill", una "death"): sin filtro se
    # mezclan y el mapa dibujaria muertes bajo el rotulo de kills.
    fig = mapplot.plot_kills_on_map(_kills_demo(), "ascent", result="kill", mode="scatter")
    ax = fig.axes[0]
    assert len(ax.collections[0].get_offsets()) == 1   # de las 2 filas de ascent, solo la kill
    assert "kills" in ax.get_title()

    fig2 = mapplot.plot_kills_on_map(_kills_demo(), "ascent", result="death", mode="scatter")
    assert len(fig2.axes[0].collections[0].get_offsets()) == 1
    assert "muertes" in fig2.axes[0].get_title()


def test_plot_kills_on_map_sin_filtro_de_result_mezcla_kills_y_muertes(mapa_mockeado):
    # Documenta el comportamiento por defecto: result="ALL" dibuja los 2 eventos de
    # ascent y el titulo dice "eventos", no "kills".
    fig = mapplot.plot_kills_on_map(_kills_demo(), "ascent", mode="scatter")
    ax = fig.axes[0]
    assert len(ax.collections[0].get_offsets()) == 2
    assert "eventos" in ax.get_title()


def test_plot_kills_on_map_mode_invalido_lanza_valueerror():
    # La validacion precede a cualquier I/O, por eso no hace falta mockear el asset.
    with pytest.raises(ValueError):
        mapplot.plot_kills_on_map(_kills_demo(), "ascent", mode="heat_map")
