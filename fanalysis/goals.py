"""Los goles: de los picos de la clase de gol a una lista validada.

Un pico de GOAL por encima del umbral es un **candidato**. Para que cuente se le hacen
dos comprobaciones, las dos con números medidos en la línea base de T-DEED:

1. **El remate.** Un gol de verdad viene de un remate: en `videoGP.MP4`, los 10 GOAL que
   detectó T-DEED llevaban un SHOT entre 0,24 y 2,16 s antes. Se busca la puntuación más
   alta de las clases de remate en los `FINISH_WINDOW_S` anteriores. Si pasa el umbral, el
   gol queda **confirmado**; si no, **sin remate**, y no entra en los highlights salvo que
   se pida.
2. **Los duplicados.** Dos picos de gol a menos de `GOAL_DEDUP_S` son el mismo gol: entre
   dos goles reales hay celebración y saque de centro. Se queda el más seguro y el otro
   se anota en `merged`.

Nada de esto saca nada al aire: es una sugerencia para que la revise una persona (ADR 0013
§1 de football-ai), y el cuaderno lo enseña antes de montar los highlights.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from fanalysis.constants import (
    ACTION_THRESHOLD,
    FINISH_KINDS,
    FINISH_WINDOW_S,
    GOAL_DEDUP_S,
    GOAL_KIND,
    MATCH_TOLERANCE_S,
    NMS_S,
    SECONDS_PER_HOUR,
    SECONDS_PER_MINUTE,
)
from fanalysis.spotting import Action, check_threshold, find_actions

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from fanalysis.spotting import Timeline

__all__ = [
    "STATUS_CONFIRMED",
    "STATUS_NO_FINISH",
    "STATUS_UNCHECKED",
    "Finish",
    "Goal",
    "GoalError",
    "MatchReport",
    "best_finish",
    "find_goals",
    "match_goals",
    "parse_instant",
    "parse_marks",
    "parse_numbers",
    "select_for_highlights",
]

STATUS_CONFIRMED = "confirmado"
"""Gol con su remate delante."""
STATUS_NO_FINISH = "sin_remate"
"""Gol sin remate en la ventana: más probable que sea un falso positivo."""
STATUS_UNCHECKED = "sin_validar"
"""El modelo no tiene clases de remate y la comprobación no se puede hacer."""

_TIME_PART = re.compile(r"^\d+(\.\d+)?$")


class GoalError(ValueError):
    """Un parámetro o una anotación que no se puede usar. El mensaje dice cuál."""


@dataclass(frozen=True, slots=True)
class Finish:
    """El remate que precede a un gol."""

    kind: str
    t_s: float
    score: float


@dataclass(frozen=True, slots=True)
class Goal:
    """Un gol candidato, con lo que se comprobó de él."""

    number: int
    """De 1 en adelante, en orden de partido. Es el número con el que se revisa."""
    frame: int
    t_s: float
    score: float
    finish: Finish | None
    finish_checked: bool
    merged: tuple[float, ...] = field(default=())
    """Instantes de los picos que se fundieron con este por estar a menos de `dedup_s`."""

    @property
    def status(self) -> str:
        if not self.finish_checked:
            return STATUS_UNCHECKED
        return STATUS_CONFIRMED if self.finish is not None else STATUS_NO_FINISH


def best_finish(
    timeline: Timeline,
    goal_frame: int,
    *,
    window_s: float = FINISH_WINDOW_S,
    threshold: float = ACTION_THRESHOLD,
) -> Finish | None:
    """El remate más seguro en los `window_s` anteriores al gol, si pasa el umbral.

    Se mira la puntuación por frame, no los picos ya filtrados: la supresión de picos
    podría haber apagado un remate dentro de la ventana por otro más fuerte de fuera.
    """
    desde = max(0, goal_frame - round(window_s * timeline.fps))
    mejor: Finish | None = None
    for kind in FINISH_KINDS:
        if kind not in timeline.kinds:
            continue
        tramo = timeline.column(kind)[desde : goal_frame + 1]
        if tramo.size == 0:
            continue
        donde = int(np.argmax(tramo))
        score = float(tramo[donde])
        if score >= threshold and (mejor is None or score > mejor.score):
            mejor = Finish(kind=kind, t_s=timeline.time_of(desde + donde), score=score)
    return mejor


def find_goals(
    timeline: Timeline,
    *,
    threshold: float = ACTION_THRESHOLD,
    finish_threshold: float = ACTION_THRESHOLD,
    nms_s: float = NMS_S,
    finish_window_s: float = FINISH_WINDOW_S,
    dedup_s: float = GOAL_DEDUP_S,
) -> list[Goal]:
    """Los goles del tramo, sin duplicados y con su remate comprobado."""
    if GOAL_KIND not in timeline.kinds:
        msg = f"el modelo no tiene clase de gol (clases: {', '.join(timeline.kinds)})"
        raise GoalError(msg)
    check_threshold(finish_threshold, "umbral del remate")
    for nombre, valor in (("ventana del remate", finish_window_s), ("separación", dedup_s)):
        if valor < 0:
            msg = f"la {nombre} no puede ser negativa: {valor}"
            raise GoalError(msg)

    picos = find_actions(timeline, threshold=threshold, nms_s=nms_s, kinds=(GOAL_KIND,))
    elegidos: list[Action] = []
    fundidos: dict[int, list[float]] = {}
    for pico in sorted(picos, key=lambda a: a.score, reverse=True):
        cercano = min(elegidos, key=lambda a: abs(a.t_s - pico.t_s), default=None)
        if cercano is not None and abs(cercano.t_s - pico.t_s) < dedup_s:
            fundidos.setdefault(cercano.frame, []).append(pico.t_s)
            continue
        elegidos.append(pico)

    comprobable = any(kind in timeline.kinds for kind in FINISH_KINDS)
    goles = []
    for numero, pico in enumerate(sorted(elegidos, key=lambda a: a.t_s), start=1):
        remate = (
            best_finish(timeline, pico.frame, window_s=finish_window_s, threshold=finish_threshold)
            if comprobable
            else None
        )
        goles.append(
            Goal(
                number=numero,
                frame=pico.frame,
                t_s=pico.t_s,
                score=pico.score,
                finish=remate,
                finish_checked=comprobable,
                merged=tuple(sorted(fundidos.get(pico.frame, []))),
            )
        )
    return goles


def select_for_highlights(
    goals: Sequence[Goal],
    *,
    require_finish: bool = True,
    exclude: Iterable[int] = (),
    include: Iterable[int] = (),
) -> list[Goal]:
    """Los goles que van a los highlights, en orden de partido.

    Con `require_finish`, los goles sin remate se quedan fuera. Después manda la revisión:
    `exclude` quita goles por su número y `include` mete los que se habían quedado fuera.
    """
    numeros = {g.number for g in goals}
    quitar, poner = set(exclude), set(include)
    desconocidos = (quitar | poner) - numeros
    if desconocidos:
        msg = (
            f"no hay goles con los números {sorted(desconocidos)}: los que hay son "
            f"{sorted(numeros) or 'ninguno'}"
        )
        raise GoalError(msg)
    if quitar & poner:
        msg = f"los goles {sorted(quitar & poner)} están a la vez para quitar y para poner"
        raise GoalError(msg)
    elegidos = [
        g
        for g in goals
        if (g.number in poner)
        or (g.number not in quitar and not (require_finish and g.status == STATUS_NO_FINISH))
    ]
    return sorted(elegidos, key=lambda g: g.t_s)


# ------------------------------------------------------------- medir contra a mano


def parse_instant(text: str) -> float:
    """`264`, `264.5`, `04:24` o `1:04:24` a segundos."""
    partes = text.strip().split(":")
    if not 1 <= len(partes) <= len((SECONDS_PER_HOUR, SECONDS_PER_MINUTE, 1)) or not all(
        _TIME_PART.match(p) for p in partes
    ):
        msg = f"«{text}» no es un instante: se escribe como 264, 04:24 o 1:04:24"
        raise GoalError(msg)
    pesos = (SECONDS_PER_HOUR, SECONDS_PER_MINUTE, 1)[-len(partes) :]
    return sum(float(p) * w for p, w in zip(partes, pesos, strict=True))


def parse_marks(text: str) -> list[float]:
    """Los instantes de los goles reales, uno por línea o separados por comas.

    Lo que va detrás de `#` es comentario, y lo que va detrás del instante en la misma
    línea se ignora, así que vale «04:24 gol de cabeza».
    """
    instantes: list[float] = []
    for linea in text.splitlines():
        for trozo in linea.split("#", 1)[0].split(","):
            palabras = trozo.split()
            if palabras:
                instantes.append(parse_instant(palabras[0]))
    return sorted(instantes)


def parse_numbers(text: str) -> list[int]:
    """Los números de gol de la revisión: `"2, 5"` o `"2 5"` → `[2, 5]`."""
    numeros: list[int] = []
    for trozo in text.replace(",", " ").split():
        if not trozo.isdigit() or int(trozo) < 1:
            msg = f"«{trozo}» no es un número de gol: se escriben como 2, 5"
            raise GoalError(msg)
        numeros.append(int(trozo))
    return sorted(set(numeros))


@dataclass(frozen=True, slots=True)
class MatchReport:
    """Precisión y recall de los goles detectados frente a los anotados a mano."""

    tolerance_s: float
    pairs: tuple[tuple[float, float], ...]
    """`(detectado, real)` emparejados uno a uno."""
    false_positives: tuple[float, ...]
    missed: tuple[float, ...]

    @property
    def precision(self) -> float | None:
        detectados = len(self.pairs) + len(self.false_positives)
        return len(self.pairs) / detectados if detectados else None

    @property
    def recall(self) -> float | None:
        reales = len(self.pairs) + len(self.missed)
        return len(self.pairs) / reales if reales else None

    @property
    def median_error_s(self) -> float | None:
        """Mediana y no media: un acierto en el límite no arrastra la cifra."""
        errores = [abs(d - r) for d, r in self.pairs]
        return statistics.median(errores) if errores else None


def match_goals(
    detected: Sequence[float], truth: Sequence[float], *, tolerance_s: float = MATCH_TOLERANCE_S
) -> MatchReport:
    """Empareja uno a uno, empezando por los pares más cercanos, dentro de la tolerancia."""
    if tolerance_s < 0:
        msg = f"la tolerancia no puede ser negativa: {tolerance_s}"
        raise GoalError(msg)
    candidatos = sorted(
        (abs(d - r), i, j)
        for i, d in enumerate(detected)
        for j, r in enumerate(truth)
        if abs(d - r) <= tolerance_s
    )
    usados_d: set[int] = set()
    usados_r: set[int] = set()
    pares: list[tuple[float, float]] = []
    for _, i, j in candidatos:
        if i not in usados_d and j not in usados_r:
            usados_d.add(i)
            usados_r.add(j)
            pares.append((detected[i], truth[j]))
    return MatchReport(
        tolerance_s=tolerance_s,
        pairs=tuple(sorted(pares)),
        false_positives=tuple(d for i, d in enumerate(detected) if i not in usados_d),
        missed=tuple(r for j, r in enumerate(truth) if j not in usados_r),
    )
