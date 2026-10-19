from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from fanalysis.clips import ClipPlan, ClipResult
from fanalysis.goals import Finish, Goal, match_goals
from fanalysis.report import (
    action_rows,
    clip_row,
    format_instant,
    goal_row,
    match_row,
    write_csv,
    write_json,
)
from fanalysis.spotting import Action


@pytest.mark.parametrize(
    ("segundos", "texto"),
    [(0.0, "00:00.0"), (264.32, "04:24.3"), (59.96, "01:00.0"), (3864.25, "1:04:24.2")],
)
def test_instantes_legibles(segundos, texto):
    assert format_instant(segundos) == texto


def test_la_fila_de_un_gol_lleva_lo_que_se_comprobo():
    gol = Goal(
        number=2,
        frame=0,
        t_s=264.0,
        score=0.9623,
        finish=Finish("shot", 262.4, 0.71),
        finish_checked=True,
        merged=(270.0,),
    )
    fila = goal_row(gol)
    assert fila["gol"] == 2
    assert fila["instante"] == "04:24.0"
    assert fila["estado"] == "confirmado"
    assert (fila["remate"], fila["remate_antes_s"], fila["remate_confianza"]) == ("shot", 1.6, 0.71)
    assert fila["fundidos"] == ["04:30.0"]

    sin = goal_row(Goal(1, 0, 10.0, 0.3, None, True))
    assert (sin["estado"], sin["remate"], sin["remate_antes_s"]) == ("sin_remate", None, None)


def test_filas_de_acciones_clips_y_medida():
    filas = action_rows([Action(10, 30.0, "goal", 0.5), Action(5, 20.0, "shot", 0.25)])
    assert [f["clase"] for f in filas] == ["shot", "goal"]

    clip = clip_row(
        ClipResult(ClipPlan(0.0, 18.0, (1,), (8.0,), True, False), Path("c.mp4"), 18.02)
    )
    assert clip["antes_recortado"] is True
    assert clip["duracion_s"] == 18.02

    medida = match_row(match_goals([60.5], [60.0, 300.0]))
    assert medida["aciertos"] == 1
    assert medida["no_detectados"] == ["05:00.0"]
    assert (medida["precision"], medida["recall"]) == (1.0, 0.5)


def test_csv_y_json_se_escriben_en_utf8(tmp_path):
    ruta = write_csv(action_rows([Action(5, 20.0, "free_kick", 0.25)]), tmp_path / "a" / "x.csv")
    with ruta.open(encoding="utf-8") as fichero:
        assert next(iter(csv.DictReader(fichero)))["clase"] == "free_kick"
    assert write_csv([], tmp_path / "vacio.csv").read_text(encoding="utf-8") == ""

    destino = write_json({"aviso": "el vídeo no tiene audio"}, tmp_path / "informe.json")
    assert "vídeo" in destino.read_text(encoding="utf-8")
    assert json.loads(destino.read_text(encoding="utf-8")) == {"aviso": "el vídeo no tiene audio"}
