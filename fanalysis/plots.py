"""La gráfica de la línea de tiempo: cuándo cree el modelo que hay gol y tiro.

Sirve para revisar a ojo lo que el umbral decide solo: un gol real que se quedó por
debajo, o una cresta que no es nada. Va en el cuaderno debajo de la tabla de goles, que
es la vista con los números.

matplotlib se importa al llamar: el resto del paquete no lo necesita.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from fanalysis.constants import GOAL_KIND, SECONDS_PER_MINUTE
from fanalysis.goals import STATUS_NO_FINISH
from fanalysis.report import format_instant

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fanalysis.goals import Goal
    from fanalysis.spotting import Timeline

__all__ = ["plot_timeline"]

# Paleta de referencia de la guía de visualización (categórica, modo claro), validada
# para estas dos series: ΔE CVD 24,7 y contraste ≥ 3:1 sobre la superficie.
_SURFACE = "#fcfcfb"
_TEXT_PRIMARY = "#0b0b0b"
_TEXT_SECONDARY = "#52514e"
_GRID = "#e4e3df"
_SERIES = {"goal": "#2a78d6", "shot": "#eb6834"}
_LABELS = {"goal": "gol", "shot": "tiro"}

_LINE_WIDTH_PT = 1.5
"""Puntos tipográficos: unos 2 px a la resolución de pantalla del cuaderno."""
_MARKER_PT = 7.0
"""Diámetro del punto de cada gol, en puntos: algo más de 8 px."""
_RING_PT = 1.5
"""Anillo del color de la superficie alrededor de cada punto, para que se lea sobre la línea."""
_FIGURE_IN = (12.0, 3.6)
"""Pulgadas: ancha, porque es una línea de tiempo de un partido entero."""
_LABEL_OFFSET_PT = 6
"""Puntos por encima del punto a los que va el número del gol."""
_Y_MAX = 1.05
"""Un poco por encima de 1 para que el punto de un gol seguro no se corte."""


def plot_timeline(
    timeline: Timeline,
    goals: Sequence[Goal],
    *,
    threshold: float,
    kinds: Sequence[str] = ("goal", "shot"),
) -> Any:
    """Las puntuaciones de gol y tiro a lo largo del tramo, con los goles numerados.

    Los goles confirmados van rellenos y los que no tienen remate, huecos: la diferencia
    no depende solo del color. Devuelve la `Figure` de matplotlib.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.ticker import FuncFormatter, MultipleLocator  # noqa: PLC0415

    figura, ejes = plt.subplots(figsize=_FIGURE_IN, facecolor=_SURFACE)
    ejes.set_facecolor(_SURFACE)
    tiempos = timeline.offset_s + np.arange(timeline.n_frames) / timeline.fps
    for kind in kinds:
        if kind not in timeline.kinds:
            continue
        ejes.plot(
            tiempos,
            timeline.column(kind),
            color=_SERIES.get(kind, _TEXT_SECONDARY),
            linewidth=_LINE_WIDTH_PT,
            solid_joinstyle="round",
            solid_capstyle="round",
            label=_LABELS.get(kind, kind),
        )

    ejes.axhline(threshold, color=_TEXT_SECONDARY, linewidth=1, zorder=1)
    ejes.annotate(
        f"umbral {threshold:g}",
        xy=(1, threshold),
        xycoords=("axes fraction", "data"),
        xytext=(-4, 4),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color=_TEXT_SECONDARY,
        fontsize=9,
    )

    color_gol = _SERIES[GOAL_KIND]
    for gol in goals:
        hueco = gol.status == STATUS_NO_FINISH
        ejes.plot(
            [gol.t_s],
            [gol.score],
            marker="o",
            markersize=_MARKER_PT,
            markerfacecolor=_SURFACE if hueco else color_gol,
            markeredgecolor=color_gol if hueco else _SURFACE,
            markeredgewidth=_RING_PT,
            linestyle="none",
            zorder=3,
        )
        ejes.annotate(
            str(gol.number),
            xy=(gol.t_s, gol.score),
            xytext=(0, _LABEL_OFFSET_PT),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=_TEXT_PRIMARY,
            fontsize=9,
            fontweight="bold",
        )

    ejes.set_xlim(timeline.offset_s, timeline.end_s)
    ejes.set_ylim(0, _Y_MAX)
    ejes.yaxis.set_major_locator(MultipleLocator(0.5))
    ejes.xaxis.set_major_formatter(FuncFormatter(lambda x, _: format_instant(x)[:-2]))
    ejes.grid(visible=True, color=_GRID, linewidth=0.75, linestyle="-")
    ejes.set_axisbelow(True)
    for lado in ("top", "right"):
        ejes.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ejes.spines[lado].set_color(_GRID)
    ejes.tick_params(colors=_TEXT_SECONDARY, labelsize=9, length=0)
    ejes.set_ylabel("puntuación", color=_TEXT_SECONDARY, fontsize=9)
    ejes.set_xlabel(
        "minuto del vídeo" if timeline.end_s >= SECONDS_PER_MINUTE else "segundo del vídeo",
        color=_TEXT_SECONDARY,
        fontsize=9,
    )
    ejes.set_title("Puntuación de gol y de tiro", loc="left", color=_TEXT_PRIMARY, fontsize=11)
    # La leyenda va fuera, en la línea del título: dentro taparía las crestas del principio.
    leyenda = ejes.legend(
        loc="lower right", bbox_to_anchor=(1, 1), frameon=False, fontsize=9, ncols=len(kinds)
    )
    for texto in leyenda.get_texts():
        texto.set_color(_TEXT_PRIMARY)
    figura.tight_layout()
    return figura
