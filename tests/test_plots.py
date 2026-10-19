from __future__ import annotations

import pytest
from ayudas import timeline_con

from fanalysis.goals import find_goals

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from fanalysis.plots import plot_timeline  # noqa: E402 - después de elegir el backend


def test_la_linea_de_tiempo_pinta_gol_y_tiro_con_los_goles_numerados():
    linea = timeline_con({"goal": [(60.0, 0.9), (100.0, 0.4)], "shot": [(58.4, 0.5)]})
    goles = find_goals(linea)

    figura = plot_timeline(linea, goles, threshold=0.2)

    ejes = figura.axes[0]
    assert [t.get_text() for t in ejes.get_legend().get_texts()] == ["gol", "tiro"]
    assert {"1", "2", "umbral 0.2"} <= {t.get_text() for t in ejes.texts}
    # El gol sin remate va hueco: su relleno es el de la superficie, no el de la serie.
    puntos = [trazo for trazo in ejes.lines if trazo.get_marker() == "o"]
    assert len(puntos) == 2
    assert puntos[0].get_markerfacecolor() != puntos[1].get_markerfacecolor()
    matplotlib.pyplot.close(figura)


def test_sin_goles_tambien_se_pinta():
    figura = plot_timeline(timeline_con({}, duracion_s=30.0), [], threshold=0.5)
    assert figura.axes[0].get_xlabel() == "segundo del vídeo"
    matplotlib.pyplot.close(figura)
