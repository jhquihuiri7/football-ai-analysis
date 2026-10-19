"""Lo que se guarda y se enseña: instantes legibles, tablas y el informe JSON."""

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING, Any

from fanalysis.constants import SECONDS_PER_HOUR, SECONDS_PER_MINUTE

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from fanalysis.clips import ClipResult
    from fanalysis.goals import Goal, MatchReport
    from fanalysis.spotting import Action

__all__ = [
    "action_rows",
    "clip_row",
    "format_instant",
    "goal_row",
    "match_row",
    "write_csv",
    "write_json",
]


def format_instant(t_s: float) -> str:
    """`264.3` → `04:24.3`; con horas, `1:04:24.3`. Con una décima: el modelo va a 80 ms."""
    decimas = round(t_s * 10)
    segundos, decima = divmod(decimas, 10)
    horas, resto = divmod(segundos, SECONDS_PER_HOUR)
    minutos, segundos = divmod(resto, SECONDS_PER_MINUTE)
    if horas:
        return f"{horas}:{minutos:02d}:{segundos:02d}.{decima}"
    return f"{minutos:02d}:{segundos:02d}.{decima}"


def action_rows(actions: Sequence[Action]) -> list[dict[str, Any]]:
    """Una fila por acción, en orden de partido."""
    return [
        {
            "instante": format_instant(a.t_s),
            "segundos": round(a.t_s, 2),
            "clase": a.kind,
            "confianza": round(a.score, 3),
        }
        for a in sorted(actions, key=lambda a: a.t_s)
    ]


def goal_row(goal: Goal) -> dict[str, Any]:
    """Un gol con todo lo que se comprobó, para la tabla y el informe."""
    remate = goal.finish
    return {
        "gol": goal.number,
        "instante": format_instant(goal.t_s),
        "segundos": round(goal.t_s, 2),
        "confianza": round(goal.score, 3),
        "estado": goal.status,
        "remate": remate.kind if remate else None,
        "remate_antes_s": round(goal.t_s - remate.t_s, 2) if remate else None,
        "remate_confianza": round(remate.score, 3) if remate else None,
        "fundidos": [format_instant(t) for t in goal.merged],
    }


def clip_row(clip: ClipResult) -> dict[str, Any]:
    plan = clip.plan
    return {
        "clip": clip.path.name,
        "goles": list(plan.goals),
        "desde": format_instant(plan.start_s),
        "hasta": format_instant(plan.end_s),
        "duracion_s": round(clip.duration_s, 2),
        "antes_recortado": plan.pre_truncated,
        "despues_recortado": plan.post_truncated,
    }


def match_row(report: MatchReport) -> dict[str, Any]:
    def ratio(value: float | None) -> float | None:
        return None if value is None else round(value, 3)

    return {
        "tolerancia_s": report.tolerance_s,
        "aciertos": len(report.pairs),
        "falsos_positivos": [format_instant(t) for t in report.false_positives],
        "no_detectados": [format_instant(t) for t in report.missed],
        "precision": ratio(report.precision),
        "recall": ratio(report.recall),
        "error_mediano_s": ratio(report.median_error_s),
    }


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> Path:
    """Las filas en CSV, con las columnas de la primera. Sin filas, solo la cabecera vacía."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fichero:
        if rows:
            escritor = csv.DictWriter(fichero, fieldnames=list(rows[0]))
            escritor.writeheader()
            escritor.writerows(rows)
    return path


def write_json(data: Mapping[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
