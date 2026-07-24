import pandas as pd
import pytest

import src.mapviz as mapviz
from src.mapviz import game_to_pixel
from src.mapviz import get_map_asset


def test_game_to_pixel_aplica_transformacion_con_swap_de_ejes():
    # px se calcula desde 'y', py desde 'x'; yMultiplier negativo hace el flip.
    df = pd.DataFrame({"x": [100], "y": [200]})
    transform = {
        "xMultiplier": 0.001, "yMultiplier": -0.001,
        "xScalarToAdd": 0.5, "yScalarToAdd": 0.5,
    }
    # Tamaño NO cuadrado a proposito: si la implementacion intercambiara ancho y alto,
    # saldria px=350/py=400 y el test fallaria. Con (1000,1000) el swap pasaria inadvertido.
    out = game_to_pixel(df, transform, size=(1000, 500))
    # nx = 200*0.001 + 0.5 = 0.7 -> px = 0.7*1000 = 700
    # ny = 100*(-0.001) + 0.5 = 0.4 -> py = 0.4*500  = 200
    assert out["px"].iloc[0] == pytest.approx(700.0)
    assert out["py"].iloc[0] == pytest.approx(200.0)
    # No muta el df original.
    assert "px" not in df.columns


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
    get = mocker.patch("src.mapviz.requests.get")   # red cortada por si el orden cambia
    with pytest.raises(ValueError):
        get_map_asset("the range")
    assert get.call_count == 0                      # el ValueError precede a la descarga


def test_fetch_map_transforms_descarta_mapas_sin_multiplicadores(mocker):
    fake = {"data": [
        {"displayName": "Ascent", "xMultiplier": 0.00007, "yMultiplier": -0.00007,
         "xScalarToAdd": 0.813895, "yScalarToAdd": 0.573242, "displayIcon": "http://img/ascent.png"},
        {"displayName": "The Range", "xMultiplier": 0, "yMultiplier": 0,
         "xScalarToAdd": 0, "yScalarToAdd": 0, "displayIcon": "http://img/range.png"},
    ]}
    resp = mocker.Mock()
    resp.json.return_value = fake
    resp.raise_for_status = mocker.Mock()
    mocker.patch("src.mapviz.requests.get", return_value=resp)

    out = mapviz.fetch_map_transforms()
    assert set(out.keys()) == {"ascent"}          # The Range descartado (multiplicadores en 0)
    assert out["ascent"]["xMultiplier"] == 0.00007
    assert out["ascent"]["displayIcon"] == "http://img/ascent.png"
    assert "displayName" not in out["ascent"]     # solo las 5 claves esperadas
