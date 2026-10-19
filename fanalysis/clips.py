"""Antes y después del gol: los clips y los highlights.

Cada gol se recorta con `PRE_ROLL_S` antes y `POST_ROLL_S` después, que es lo que decidió
el propietario para los clips del directo (ADR 0013 de football-ai): la jugada que trae
el gol, el gol y la celebración. Tres cosas de este repo:

- **Se recorta contra el vídeo, no contra la jugada.** Un gol a los 8 s del principio no
  tiene 20 s de antes: el clip empieza en el 0 y lo dice (`pre_truncated`). Igual al final.
- **Dos goles cuyos clips se pisan salen en un clip solo.** Si no, los highlights
  repetirían los mismos segundos dos veces.
- **Se recodifica.** En el directo se cortaba con `-c copy` porque cada segmento empezaba
  en un keyframe. Un vídeo cualquiera no: copiar cortaría en el keyframe anterior, a
  veces segundos antes, o empezaría con frames congelados. Recodificar con libx264 corta
  en el frame y deja todos los clips con el mismo formato, así que los highlights sí se
  pegan copiando.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fanalysis.constants import (
    AUDIO_BITRATE,
    CLIP_DURATION_TOLERANCE_S,
    FFMPEG,
    FFPROBE,
    POST_ROLL_S,
    PRE_ROLL_S,
    PREVIEW_CRF,
    PREVIEW_WIDTH_PX,
    SECONDS_PER_MINUTE,
    SNAPSHOT_QUALITY,
    SNAPSHOT_WIDTH_PX,
    STDERR_TAIL_CHARS,
    X264_CRF,
    X264_PRESET,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from fanalysis.goals import Goal

__all__ = [
    "ClipError",
    "ClipPlan",
    "ClipResult",
    "build_highlights",
    "clip_name",
    "concat_command",
    "cut_clips",
    "cut_command",
    "media_duration",
    "plan_clips",
    "preview_command",
    "run_command",
    "run_or_fail",
    "snapshot_command",
    "write_listfile",
]


class ClipError(RuntimeError):
    """Un clip o los highlights no se pudieron hacer. El mensaje dice por qué."""


@dataclass(frozen=True, slots=True)
class ClipPlan:
    """El tramo de un clip, en segundos del vídeo, y los goles que lleva."""

    start_s: float
    end_s: float
    goals: tuple[int, ...]
    """Los números de los goles, en orden."""
    goal_times: tuple[float, ...]
    pre_truncated: bool
    """El vídeo empezaba antes de que se cumplieran los segundos de antes del primer gol."""
    post_truncated: bool
    """El vídeo acababa antes de que se cumplieran los segundos de después del último."""

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True, slots=True)
class ClipResult:
    """Un clip ya cortado y comprobado."""

    plan: ClipPlan
    path: Path
    duration_s: float


def plan_clips(
    goals: Sequence[Goal],
    *,
    duration_s: float,
    pre_s: float = PRE_ROLL_S,
    post_s: float = POST_ROLL_S,
) -> list[ClipPlan]:
    """Un tramo por gol, recortado a lo que hay de vídeo y con los que se pisan fundidos."""
    if pre_s < 0 or post_s < 0:
        msg = f"los segundos de antes y de después no pueden ser negativos: {pre_s} y {post_s}"
        raise ClipError(msg)
    if pre_s + post_s <= 0:
        msg = "un clip necesita algo de antes o de después del gol"
        raise ClipError(msg)
    if duration_s <= 0:
        msg = f"el vídeo dura {duration_s} s"
        raise ClipError(msg)
    fuera = [g.number for g in goals if not 0 <= g.t_s <= duration_s]
    if fuera:
        msg = f"los goles {fuera} caen fuera del vídeo, que dura {duration_s:.1f} s"
        raise ClipError(msg)

    planes: list[ClipPlan] = []
    for gol in sorted(goals, key=lambda g: g.t_s):
        inicio, fin = gol.t_s - pre_s, gol.t_s + post_s
        if planes and inicio <= planes[-1].end_s:
            previo = planes[-1]
            planes[-1] = ClipPlan(
                start_s=previo.start_s,
                end_s=min(fin, duration_s),
                goals=(*previo.goals, gol.number),
                goal_times=(*previo.goal_times, gol.t_s),
                pre_truncated=previo.pre_truncated,
                post_truncated=fin > duration_s,
            )
            continue
        planes.append(
            ClipPlan(
                start_s=max(0.0, inicio),
                end_s=min(fin, duration_s),
                goals=(gol.number,),
                goal_times=(gol.t_s,),
                pre_truncated=inicio < 0,
                post_truncated=fin > duration_s,
            )
        )
    return planes


def clip_name(plan: ClipPlan) -> str:
    """`gol_01_04m24s.mp4`, o `goles_02-03_05m28s.mp4` si el clip lleva varios."""
    numeros = "-".join(f"{n:02d}" for n in plan.goals)
    primero = plan.goal_times[0]
    minutos, segundos = divmod(int(primero), SECONDS_PER_MINUTE)
    prefijo = "gol" if len(plan.goals) == 1 else "goles"
    return f"{prefijo}_{numeros}_{minutos:02d}m{segundos:02d}s.mp4"


def cut_command(
    source: Path, target: Path, *, start_s: float, duration_s: float, has_audio: bool
) -> list[str]:
    """La orden de ffmpeg que corta un tramo en el frame, recodificando.

    `-ss` va delante de `-i`: busca rápido al keyframe anterior y, al recodificar, ffmpeg
    descarta lo que sobra hasta el instante exacto.
    """
    orden = [
        FFMPEG, "-v", "error", "-nostdin", "-y",
        "-ss", f"{start_s:.3f}", "-i", str(source), "-t", f"{duration_s:.3f}",
        "-map", "0:v:0",
    ]  # fmt: skip
    if has_audio:
        orden += ["-map", "0:a:0?", "-c:a", "aac", "-b:a", AUDIO_BITRATE]
    orden += [
        "-c:v", "libx264", "-preset", X264_PRESET, "-crf", str(X264_CRF),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(target),
    ]  # fmt: skip
    return orden


def write_listfile(paths: Sequence[Path], listfile: Path) -> Path:
    """La lista del demuxer `concat`: rutas absolutas en POSIX, con la comilla escapada."""
    lineas = []
    for ruta in paths:
        texto = ruta.resolve().as_posix().replace("'", "'\\''")
        lineas.append(f"file '{texto}'")
    listfile.parent.mkdir(parents=True, exist_ok=True)
    listfile.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return listfile


def concat_command(listfile: Path, target: Path) -> list[str]:
    """Pega los clips copiando: salieron todos del mismo codificador con los mismos ajustes."""
    return [
        FFMPEG, "-v", "error", "-nostdin", "-y",
        "-f", "concat", "-safe", "0", "-i", str(listfile),
        "-c", "copy", "-movflags", "+faststart", str(target),
    ]  # fmt: skip


def snapshot_command(
    source: Path, target: Path, *, t_s: float, width: int = SNAPSHOT_WIDTH_PX
) -> list[str]:
    """Un fotograma del instante `t_s`, en JPEG y con `width` de ancho, para la revisión."""
    return [
        FFMPEG, "-v", "error", "-nostdin", "-y",
        "-ss", f"{max(0.0, t_s):.3f}", "-i", str(source),
        "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", str(SNAPSHOT_QUALITY),
        str(target),
    ]  # fmt: skip


def preview_command(source: Path, target: Path, *, width: int = PREVIEW_WIDTH_PX) -> list[str]:
    """Una copia ligera de un clip para verla dentro del cuaderno sin cargar el original."""
    return [
        FFMPEG, "-v", "error", "-nostdin", "-y", "-i", str(source),
        "-vf", f"scale={width}:-2", "-c:v", "libx264", "-preset", X264_PRESET,
        "-crf", str(PREVIEW_CRF), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", AUDIO_BITRATE, "-movflags", "+faststart",
        str(target),
    ]  # fmt: skip


def run_or_fail(command: list[str], what: str) -> None:
    """Ejecuta una orden de ffmpeg y, si falla, dice qué se intentaba y por qué."""
    codigo, motivo = run_command(command)
    if codigo != 0:
        msg = f"ffmpeg no pudo {what} (código {codigo}): {motivo.strip()}"
        raise ClipError(msg)


def run_command(command: list[str]) -> tuple[int, str]:
    """Ejecuta una orden y devuelve su código y el final de su salida de error."""
    try:
        hecho = subprocess.run(command, capture_output=True, check=False)
    except FileNotFoundError as error:
        msg = f"no se encuentra {command[0]}: hace falta ffmpeg instalado y en el PATH"
        raise ClipError(msg) from error
    return hecho.returncode, hecho.stderr.decode("utf-8", "replace")[-STDERR_TAIL_CHARS:]


def media_duration(path: Path) -> float:
    """Lo que dura un fichero según ffprobe, en segundos."""
    orden = [FFPROBE, "-v", "error", "-print_format", "json", "-show_format", str(path)]
    try:
        hecho = subprocess.run(orden, capture_output=True, check=False)
    except FileNotFoundError as error:
        msg = f"no se encuentra {FFPROBE}: hace falta ffmpeg instalado y en el PATH"
        raise ClipError(msg) from error
    if hecho.returncode != 0:
        texto = hecho.stderr.decode("utf-8", "replace")[-STDERR_TAIL_CHARS:]
        msg = f"ffprobe no pudo leer {path.name}: {texto.strip()}"
        raise ClipError(msg)
    try:
        return float(json.loads(hecho.stdout)["format"]["duration"])
    except (KeyError, TypeError, ValueError) as error:
        msg = f"ffprobe no da la duración de {path.name}"
        raise ClipError(msg) from error


def cut_clips(
    source: Path,
    plans: Sequence[ClipPlan],
    out_dir: Path,
    *,
    has_audio: bool,
    run: Callable[[list[str]], tuple[int, str]] = run_command,
    duration_of: Callable[[Path], float] = media_duration,
) -> list[ClipResult]:
    """Corta cada tramo y comprueba que el clip existe y dura lo que se pidió."""
    out_dir.mkdir(parents=True, exist_ok=True)
    hechos: list[ClipResult] = []
    for plan in plans:
        destino = out_dir / clip_name(plan)
        codigo, motivo = run(
            cut_command(
                source,
                destino,
                start_s=plan.start_s,
                duration_s=plan.duration_s,
                has_audio=has_audio,
            )
        )
        if codigo != 0:
            msg = f"ffmpeg no pudo cortar {destino.name} (código {codigo}): {motivo.strip()}"
            raise ClipError(msg)
        duracion = _check(destino, plan.duration_s, duration_of)
        hechos.append(ClipResult(plan=plan, path=destino, duration_s=duracion))
    return hechos


def build_highlights(
    clips: Sequence[ClipResult],
    target: Path,
    *,
    run: Callable[[list[str]], tuple[int, str]] = run_command,
    duration_of: Callable[[Path], float] = media_duration,
) -> Path:
    """Pega los clips en orden de partido y comprueba que dura lo que suman."""
    if not clips:
        msg = "no hay clips que pegar: ningún gol pasó la revisión"
        raise ClipError(msg)
    ordenados = sorted(clips, key=lambda c: c.plan.start_s)
    listfile = write_listfile([c.path for c in ordenados], target.with_suffix(".txt"))
    codigo, motivo = run(concat_command(listfile, target))
    if codigo != 0:
        msg = f"ffmpeg no pudo montar los highlights (código {codigo}): {motivo.strip()}"
        raise ClipError(msg)
    _check(target, sum(c.duration_s for c in ordenados), duration_of)
    return target


def _check(path: Path, expected_s: float, duration_of: Callable[[Path], float]) -> float:
    if not path.is_file() or path.stat().st_size == 0:
        msg = f"ffmpeg acabó sin error pero {path.name} no existe o está vacío"
        raise ClipError(msg)
    real = duration_of(path)
    if abs(real - expected_s) > CLIP_DURATION_TOLERANCE_S:
        msg = f"{path.name} dura {real:.2f} s y se pidieron {expected_s:.2f} s: el corte salió mal"
        raise ClipError(msg)
    return real
