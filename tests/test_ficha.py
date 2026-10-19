from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any

import pytest
import yaml
from ayudas import REGISTRO, NodeArg, ficha_real

from fanalysis.ficha import (
    FichaError,
    canonical,
    check_session,
    load_ficha,
    parse_ficha,
    sha256_of,
    verify_model,
)


def bloque_real() -> dict[str, Any]:
    return yaml.safe_load(REGISTRO.read_text(encoding="utf-8"))["models"]["tdeed-snb"]


def test_la_ficha_de_tdeed_se_lee_entera():
    ficha = ficha_real()
    assert ficha.clip_frames == 100
    assert (ficha.height, ficha.width) == (448, 796)
    assert ficha.sample_fps == 12.5
    assert ficha.window_s == 8.0
    assert ficha.activation == "softmax"
    assert dict(ficha.outputs) == {"logits": "logits", "displacement": "displacement"}
    assert ficha.kinds[0] == "normal_play"
    assert ficha.kinds[-1] == "goal"
    assert {"shot", "header", "high_pass", "free_kick"} <= set(ficha.kinds)
    assert ficha.sha256.startswith("dde3c1d8")
    assert "SOLO PARA MEDIR" in ficha.license


def test_la_ficha_es_la_misma_que_declara_football_ai():
    """El bloque es copia literal del registro de football-ai, si está al lado."""
    origen = REGISTRO.parents[2] / "football-ai" / "models" / "registry.yaml"
    if not origen.is_file():
        pytest.skip("football-ai no está junto a este repo")
    alli = yaml.safe_load(origen.read_text(encoding="utf-8"))["models"]["tdeed-snb"]
    assert alli == bloque_real()


@pytest.mark.parametrize(
    ("nombre", "esperado"),
    [("GOAL", "goal"), ("HIGH PASS", "high_pass"), (" Ball  Player Block ", "ball_player_block")],
)
def test_canonical(nombre, esperado):
    assert canonical(nombre) == esperado


def _roto(**cambios: Any) -> dict[str, Any]:
    bloque = bloque_real()
    for ruta, valor in cambios.items():
        destino = bloque
        *camino, ultima = ruta.split("__")
        for paso in camino:
            destino = destino[paso]
        if valor is None:
            destino.pop(ultima)
        else:
            destino[ultima] = valor
    return bloque


@pytest.mark.parametrize(
    ("cambios", "motivo"),
    [
        ({"classes": ["normal_play", "SHOT"]}, "gol"),
        ({"classes": ["GOAL", "goal"]}, "repite"),
        ({"input__layout": "NCHW", "input__shape": [1, 3, 448, 796]}, "temporal"),
        ({"input__layout": "TNCHW"}, "batch"),
        ({"input__sample_fps": None}, "sample_fps"),
        ({"input__sample_fps": 0}, "sample_fps"),
        ({"input__color": "YUV"}, "color"),
        ({"input__std": [1.0, 0.0, 1.0]}, "std"),
        ({"input__shape": [2, 100, 3, 448, 796]}, "batch 1"),
        ({"activation": "relu"}, "activación"),
        ({"outputs": [{"name": "x", "shape": [1], "meaning": "displacement"}]}, "logits"),
        (
            {"outputs": [{"name": "l", "meaning": "logits"}, {"name": "h", "meaning": "heatmap"}]},
            "heatmap",
        ),
        ({"artifacts__onnx__sha256": None}, "sha256"),
    ],
)
def test_una_ficha_que_no_sirve_dice_por_que(cambios, motivo):
    with pytest.raises(FichaError, match=motivo):
        parse_ficha("tdeed-snb", _roto(**cambios))


def test_el_registro_sin_el_modelo_lo_dice(tmp_path):
    with pytest.raises(FichaError, match="no tiene el modelo «otro»"):
        load_ficha(REGISTRO, "otro")
    with pytest.raises(FichaError, match="no existe"):
        load_ficha(tmp_path / "registry.yaml", "tdeed-snb")


def test_el_modelo_se_comprueba_byte_a_byte(tmp_path):
    modelo = tmp_path / "m.onnx"
    modelo.write_bytes(b"no es un modelo" * 1000)
    assert sha256_of(modelo) == hashlib.sha256(modelo.read_bytes()).hexdigest()

    with pytest.raises(FichaError, match="no es el modelo de la ficha"):
        verify_model(ficha_real(), modelo)
    with pytest.raises(FichaError, match="no existe"):
        verify_model(ficha_real(), tmp_path / "falta.onnx")
    verify_model(replace(ficha_real(), sha256=sha256_of(modelo)), modelo)


def _salidas(clases: int = 13) -> list[NodeArg]:
    return [NodeArg("logits", [1, 100, clases]), NodeArg("displacement", [1, 100])]


def test_la_sesion_buena_pasa_y_un_eje_dinamico_no_se_compara():
    check_session(ficha_real(), [NodeArg("clip", [1, 100, 3, 448, 796])], _salidas())
    check_session(ficha_real(), [NodeArg("clip", ["batch", 100, 3, 448, 796])], _salidas())


@pytest.mark.parametrize(
    ("entradas", "salidas", "motivo"),
    [
        ([NodeArg("input", [1, 100, 3, 448, 796])], _salidas(), "entrada"),
        ([NodeArg("clip", [1, 3, 100, 448, 796])], _salidas(), "entrada"),
        ([NodeArg("clip", [1, 100, 3, 448])], _salidas(), "entrada"),
        ([], _salidas(), "entradas"),
        (
            [NodeArg("clip", [1, 100, 3, 448, 796])],
            [NodeArg("logits", [1, 100, 13])],
            "displacement",
        ),
        ([NodeArg("clip", [1, 100, 3, 448, 796])], _salidas(clases=18), "18 clases"),
    ],
)
def test_una_sesion_que_no_casa_con_la_ficha_se_rechaza(entradas, salidas, motivo):
    with pytest.raises(FichaError, match=motivo):
        check_session(ficha_real(), entradas, salidas)
