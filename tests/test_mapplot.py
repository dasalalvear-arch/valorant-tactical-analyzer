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
