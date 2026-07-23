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
