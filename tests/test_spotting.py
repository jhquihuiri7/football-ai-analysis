from __future__ import annotations

import numpy as np
import pytest
from ayudas import CorredorFalso, FuenteFalsa, ficha_pequena, timeline_con

from fanalysis.spotting import (
    SpottingError,
    Timeline,
    activate,
    aggregate,
    apply_displacement,
    background_share,
    detect,
    find_actions,
    looks_blind,
    peaks,
    probe_starts,
    window_starts,
)


def test_softmax_por_frame_y_sigmoid():
    logits = np.array([[0.0, 0.0], [10.0, 0.0], [1000.0, 0.0]], np.float32)
    scores = activate(logits, "softmax")
    np.testing.assert_allclose(scores.sum(axis=1), 1.0, rtol=1e-6)
    assert scores[0, 0] == pytest.approx(0.5)
    assert scores[2, 0] == pytest.approx(1.0)  # estable con logits enormes
    np.testing.assert_allclose(activate(np.zeros((1, 2), np.float32), "sigmoid"), 0.5)


def _desplazamiento_de_tdeed(scores: np.ndarray, desplazamiento: np.ndarray) -> np.ndarray:
    """El post-proceso original, escrito como su doble bucle, para comparar."""
    salida = np.zeros_like(scores)
    ultimo = scores.shape[0] - 1
    for t in range(scores.shape[0]):
        destino = max(0, min(ultimo, t - round(float(desplazamiento[t]))))
        salida[destino] = np.maximum(salida[destino], scores[t])
    return salida


def test_el_desplazamiento_es_el_de_tdeed():
    rng = np.random.default_rng(3)
    scores = rng.random((100, 13), dtype=np.float32)
    desplazamiento = rng.uniform(-4.4, 4.4, size=100).astype(np.float32)
    np.testing.assert_array_equal(
        apply_displacement(scores, desplazamiento),
        _desplazamiento_de_tdeed(scores, desplazamiento),
    )


def test_el_desplazamiento_mueve_hacia_atras_agrega_por_maximo_y_se_queda_en_el_borde():
    scores = np.array([[0.1], [0.2], [0.9], [0.4]], np.float32)
    desplazamiento = np.array([3.0, 1.0, 1.0, 0.0], np.float32)
    movidas = apply_displacement(scores, desplazamiento)
    # el 0 se sale por delante y se queda en el 0; el 1 y el 2 van al 0 y al 1.
    np.testing.assert_allclose(movidas[:, 0], [0.2, 0.9, 0.0, 0.4])
    with pytest.raises(SpottingError, match="desplazamiento"):
        apply_displacement(scores, np.zeros(3, np.float32))


@pytest.mark.parametrize(
    ("frames", "esperadas"),
    [(100, [0]), (124, [0, 24]), (125, [0, 25]), (130, [0, 25, 30]), (150, [0, 25, 50])],
)
def test_las_ventanas_acaban_alineadas_con_el_final(frames, esperadas):
    assert window_starts(frames, 100, 25) == esperadas


def test_ventanas_imposibles():
    with pytest.raises(SpottingError, match="necesita 100"):
        window_starts(99, 100, 25)
    with pytest.raises(SpottingError, match="salto"):
        window_starts(500, 100, 101)
    with pytest.raises(SpottingError, match="salto"):
        window_starts(500, 100, 0)


def test_la_media_cuenta_solo_las_ventanas_que_dejaron_algo():
    a = np.array([[1.0, 0.0], [0.0, 0.0], [0.5, 0.5]], np.float32)
    b = np.array([[0.0, 1.0], [0.2, 0.8]], np.float32)
    medias, soporte = aggregate([(0, a), (1, b)], 4, 2)
    np.testing.assert_array_equal(soporte, [1, 1, 2, 0])
    # el frame 1 quedó a cero en `a`: su media es la de `b` sola, no la mitad.
    np.testing.assert_allclose(medias, [[1.0, 0.0], [0.0, 1.0], [0.35, 0.65], [0.0, 0.0]])
    with pytest.raises(SpottingError, match="no cabe"):
        aggregate([(3, a)], 4, 2)


def test_picos_por_clase_con_supresion_por_puntuacion():
    kinds = ("normal_play", "shot", "goal")
    scores = np.zeros((40, 3), np.float32)
    scores[:, 0] = 0.9
    scores[[10, 12, 30], 2] = [0.6, 0.8, 0.5]
    scores[11, 1] = 0.15
    encontrados = peaks(scores, kinds, threshold=0.2, min_gap_frames=5)
    # el 12 gana al 10 por puntuación aunque llegue después; el tiro no pasa el umbral;
    # el fondo nunca da picos.
    assert encontrados == [(12, "goal", pytest.approx(0.8)), (30, "goal", pytest.approx(0.5))]
    with pytest.raises(SpottingError, match="puntuaciones"):
        peaks(scores, kinds[:2])


def test_detect_ve_cada_frame_en_orden_y_situa_el_evento():
    ficha = ficha_pequena(frames=10)
    corredor = CorredorFalso(ficha, eventos={17: "GOAL", 3: "SHOT"})
    linea = detect(
        FuenteFalsa(28, ficha.height, ficha.width), corredor, ficha, hop=5, offset_s=60.0
    )

    inicios = [ventana[0] for ventana in corredor.ventanas]
    assert inicios == window_starts(28, 10, 5) == [0, 5, 10, 15, 18]
    for ventana in corredor.ventanas:
        assert ventana == list(range(ventana[0], ventana[0] + 10))
    assert linea.n_frames == 28
    assert linea.windows == 5
    assert linea.support.min() >= 1
    assert linea.time_of(17) == pytest.approx(60.0 + 17 / ficha.sample_fps)

    acciones = find_actions(linea, nms_s=0.0)
    assert [(a.frame, a.kind) for a in acciones] == [(3, "shot"), (17, "goal")]
    assert acciones[1].t_s == pytest.approx(60.0 + 17 / ficha.sample_fps)


def test_detect_sin_frames_para_una_ventana():
    ficha = ficha_pequena(frames=10)
    with pytest.raises(SpottingError, match="ventanas de 10"):
        detect(FuenteFalsa(9, ficha.height, ficha.width), CorredorFalso(ficha), ficha, hop=5)


def test_detect_rechaza_una_salida_de_otra_forma():
    ficha = ficha_pequena(frames=10)

    class Corto(CorredorFalso):
        def infer(self, frames):
            salidas = super().infer(frames)
            salidas["logits"] = salidas["logits"][:, :2]
            return salidas

    with pytest.raises(SpottingError, match="logits"):
        detect(FuenteFalsa(10, ficha.height, ficha.width), Corto(ficha), ficha, hop=5)


def test_la_linea_de_tiempo_se_guarda_y_se_recupera(tmp_path):
    linea = timeline_con({"goal": [(30.0, 0.7)]}, offset_s=5.0, duracion_s=60.0)
    ruta = linea.save(tmp_path / "sub" / "puntuaciones.npz")
    vuelta = Timeline.load(ruta)
    np.testing.assert_array_equal(vuelta.scores, linea.scores)
    np.testing.assert_array_equal(vuelta.support, linea.support)
    assert (vuelta.kinds, vuelta.fps, vuelta.offset_s, vuelta.hop) == (
        linea.kinds,
        linea.fps,
        linea.offset_s,
        linea.hop,
    )
    assert vuelta.frame_of(30.0) == linea.frame_of(30.0) == 312
    assert vuelta.frame_of(-100.0) == 0
    assert vuelta.end_s == pytest.approx(65.0)


def test_find_actions_valida_y_filtra():
    linea = timeline_con({"goal": [(30.0, 0.7)], "shot": [(29.0, 0.4), (29.5, 0.3)]})
    assert [a.kind for a in find_actions(linea)] == ["shot", "goal"]  # 0,5 s < 2 s: un tiro
    assert [a.kind for a in find_actions(linea, nms_s=0.0)] == ["shot", "shot", "goal"]
    assert [a.kind for a in find_actions(linea, kinds=("goal",))] == ["goal"]
    with pytest.raises(SpottingError, match="umbral"):
        find_actions(linea, threshold=0.0)
    with pytest.raises(SpottingError, match="negativa"):
        find_actions(linea, nms_s=-1.0)


def _picos_ingenuos(scores, kinds, threshold, gap):
    """La supresión de football-ai tal cual: cada candidato contra todos los elegidos."""
    salida = []
    for indice, nombre in enumerate(kinds):
        if nombre == "normal_play":
            continue
        columna = scores[:, indice]
        candidatos = sorted(
            (int(t) for t in np.flatnonzero(columna >= threshold)),
            key=lambda t: float(columna[t]),
            reverse=True,
        )
        elegidos = []
        for frame in candidatos:
            if all(abs(frame - otro) >= gap for otro in elegidos):
                elegidos.append(frame)
                salida.append((frame, nombre, float(columna[frame])))
    return sorted(salida)


@pytest.mark.parametrize("gap", [0, 1, 2, 7, 25])
def test_la_mascara_suprime_igual_que_comparar_con_todos(gap):
    rng = np.random.default_rng(gap)
    kinds = ("normal_play", "pass", "shot", "goal")
    scores = rng.random((600, 4), dtype=np.float32)
    scores[100:110, 3] = 0.5  # una meseta: empates que deciden el orden estable
    assert peaks(scores, kinds, threshold=0.3, min_gap_frames=gap) == _picos_ingenuos(
        scores, kinds, 0.3, gap
    )


def test_find_actions_con_una_clase_que_no_existe():
    with pytest.raises(SpottingError, match="no tiene las clases"):
        find_actions(timeline_con({}), kinds=("gol",))


def test_el_fondo_medio_dice_si_el_modelo_ve():
    kinds = ("normal_play", "pass", "goal")
    scores = np.array([[0.9, 0.1, 0.0], [0.3, 0.7, 0.0]], np.float32)
    assert background_share(scores, kinds) == pytest.approx(0.6)
    with pytest.raises(SpottingError, match="fondo"):
        background_share(scores, ("pass", "goal", "shot"))
    # Las medidas del 2026-10-19: estadio 0,44-0,46; panorámica amateur 0,76.
    assert not looks_blind([0.44, 0.46, 0.45])
    assert looks_blind([0.76, 0.79, 0.71])
    assert not looks_blind([0.9, 0.45, 0.44])  # manda la mediana, no una ventana suelta
    with pytest.raises(SpottingError, match="ventanas de prueba"):
        looks_blind([])


def test_las_ventanas_de_prueba_se_reparten_por_el_tramo():
    assert probe_starts(0.0, 30.0, 8.0, 3) == pytest.approx([5.5, 11.0, 16.5])
    assert probe_starts(100.0, 108.0, 8.0, 1) == [100.0]
    with pytest.raises(SpottingError, match="una ventana"):
        probe_starts(0.0, 5.0, 8.0, 3)
    with pytest.raises(SpottingError, match="al menos una"):
        probe_starts(0.0, 30.0, 8.0, 0)
