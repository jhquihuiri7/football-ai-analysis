from __future__ import annotations

import pytest
from ayudas import timeline_con

from fanalysis.goals import (
    STATUS_CONFIRMED,
    STATUS_NO_FINISH,
    STATUS_UNCHECKED,
    GoalError,
    best_finish,
    find_goals,
    match_goals,
    parse_instant,
    parse_marks,
    parse_numbers,
    select_for_highlights,
)


def test_un_gol_con_su_tiro_delante_queda_confirmado():
    # A 12,5 fps un frame son 80 ms: los instantes de los tests caen en frames exactos.
    linea = timeline_con({"goal": [(60.0, 0.9)], "shot": [(58.4, 0.5)]})
    (gol,) = find_goals(linea)
    assert gol.number == 1
    assert gol.t_s == pytest.approx(60.0)
    assert gol.status == STATUS_CONFIRMED
    assert gol.finish is not None
    assert gol.finish.kind == "shot"
    assert gol.t_s - gol.finish.t_s == pytest.approx(1.6)


def test_un_cabezazo_tambien_es_remate_y_gana_el_mas_seguro():
    linea = timeline_con({"goal": [(60.0, 0.9)], "shot": [(59.0, 0.3)], "header": [(59.5, 0.6)]})
    (gol,) = find_goals(linea)
    assert gol.finish is not None
    assert (gol.finish.kind, gol.finish.score) == ("header", pytest.approx(0.6))


@pytest.mark.parametrize(
    "tiro",
    [
        (56.0, 0.9),  # 4 s antes: es otra jugada
        (61.0, 0.9),  # después del gol: no lo trae
        (59.0, 0.1),  # por debajo del umbral
    ],
)
def test_sin_tiro_en_los_2_5_s_anteriores_el_gol_queda_sin_remate(tiro):
    linea = timeline_con({"goal": [(60.0, 0.9)], "shot": [tiro]})
    (gol,) = find_goals(linea)
    assert gol.finish is None
    assert gol.status == STATUS_NO_FINISH


def test_el_remate_se_busca_en_la_puntuacion_por_frame_y_no_en_los_picos():
    # Un tiro más fuerte justo fuera de la ventana apagaría al de dentro si se buscara
    # entre los picos ya suprimidos (están a menos de 2 s).
    linea = timeline_con({"goal": [(60.0, 0.9)], "shot": [(56.9, 0.95), (58.0, 0.4)]})
    remate = best_finish(linea, linea.frame_of(60.0))
    assert remate is not None
    assert remate.t_s == pytest.approx(58.0)


def test_dos_picos_a_menos_de_20_s_son_el_mismo_gol():
    linea = timeline_con(
        {"goal": [(30.0, 0.5), (36.0, 0.8), (80.0, 0.4), (112.0, 0.3)], "shot": [(35.0, 0.5)]}
    )
    goles = find_goals(linea)
    assert [round(g.t_s) for g in goles] == [36, 80, 112]  # 80 y 112: 32 s, como en la línea base
    assert goles[0].merged == (pytest.approx(30.0),)
    assert [g.number for g in goles] == [1, 2, 3]
    assert [g.status for g in goles] == [STATUS_CONFIRMED, STATUS_NO_FINISH, STATUS_NO_FINISH]


def test_un_modelo_sin_clases_de_remate_no_puede_validar():
    linea = timeline_con({"goal": [(60.0, 0.9)]}, kinds=("normal_play", "goal"))
    (gol,) = find_goals(linea)
    assert gol.status == STATUS_UNCHECKED
    assert select_for_highlights([gol]) == [gol]


def test_sin_clase_de_gol_no_hay_goles():
    linea = timeline_con({"shot": [(60.0, 0.9)]}, kinds=("normal_play", "shot"))
    with pytest.raises(GoalError, match="clase de gol"):
        find_goals(linea)


def test_parametros_de_los_goles():
    linea = timeline_con({"goal": [(60.0, 0.9)]})
    with pytest.raises(GoalError, match="negativa"):
        find_goals(linea, dedup_s=-1.0)
    with pytest.raises(ValueError, match="umbral del remate"):
        find_goals(linea, finish_threshold=1.5)


def test_la_revision_quita_y_pone_goles_por_numero():
    linea = timeline_con(
        {"goal": [(20.0, 0.9), (60.0, 0.8), (100.0, 0.7)], "shot": [(19.0, 0.5), (99.0, 0.5)]}
    )
    goles = find_goals(linea)
    assert [g.number for g in select_for_highlights(goles)] == [1, 3]
    assert [g.number for g in select_for_highlights(goles, require_finish=False)] == [1, 2, 3]
    assert [g.number for g in select_for_highlights(goles, include=[2], exclude=[1])] == [2, 3]
    with pytest.raises(GoalError, match=r"\[7\]"):
        select_for_highlights(goles, exclude=[7])
    with pytest.raises(GoalError, match="a la vez"):
        select_for_highlights(goles, exclude=[2], include=[2])


@pytest.mark.parametrize(
    ("texto", "segundos"),
    [("264", 264.0), ("264.5", 264.5), ("04:24", 264.0), ("4:24.5", 264.5), ("1:04:24", 3864.0)],
)
def test_instantes(texto, segundos):
    assert parse_instant(texto) == pytest.approx(segundos)


@pytest.mark.parametrize("texto", ["", "4:24:10:1", "a:10", "-5", "4m24s"])
def test_instantes_que_no_lo_son(texto):
    with pytest.raises(GoalError, match="no es un instante"):
        parse_instant(texto)


def test_las_marcas_admiten_comentarios_comas_y_texto_detras():
    texto = """
    # goles del partido
    05:28 de cabeza
    02:05, 03:24   # dos en la primera parte
    """
    assert parse_marks(texto) == [125.0, 204.0, 328.0]
    assert parse_marks("") == []


def test_los_numeros_de_la_revision():
    assert parse_numbers("2, 5") == [2, 5]
    assert parse_numbers(" 5 2 2 ") == [2, 5]
    assert parse_numbers("") == []
    for malo in ("0", "dos", "2.5", "-1"):
        with pytest.raises(GoalError, match="número de gol"):
            parse_numbers(malo)


def test_el_emparejado_es_uno_a_uno_y_por_cercania():
    # 60,5 está a 0,5 de 60 y a 1,5 de 62: se lleva el 60, y el 61,9 se queda con el 62.
    informe = match_goals([60.5, 61.9, 200.0], [60.0, 62.0, 300.0], tolerance_s=2.0)
    assert informe.pairs == ((60.5, 60.0), (61.9, 62.0))
    assert informe.false_positives == (200.0,)
    assert informe.missed == (300.0,)
    assert informe.precision == pytest.approx(2 / 3)
    assert informe.recall == pytest.approx(2 / 3)
    assert informe.median_error_s == pytest.approx(0.3)


def test_el_emparejado_sin_nada_que_contar():
    vacio = match_goals([], [])
    assert (vacio.precision, vacio.recall, vacio.median_error_s) == (None, None, None)
    solo_reales = match_goals([], [10.0])
    assert solo_reales.recall == 0.0
    assert solo_reales.precision is None
    with pytest.raises(GoalError, match="negativa"):
        match_goals([1.0], [1.0], tolerance_s=-1.0)
