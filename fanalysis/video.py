"""El vídeo de entrada: qué es, si sirve, y sus frames al ritmo del modelo.

La trampa que más caro sale con T-DEED es el ritmo. Se entrenó con frames a 12,5 fps; si
le llegan a otro ritmo no da error, ve la jugada acelerada o a cámara lenta y acierta
menos (PROGRESS de football-ai-training, T1). Aquí los frames los saca ffmpeg con su
filtro `fps`, que remuestrea por tiempo, así que da igual que el vídeo vaya a 25, a 29,97
o a ritmo variable, como los de los móviles.

Lo que no se arregla solo se valida antes de gastar una hora de GPU: un vídeo vertical,
uno más corto que una ventana o uno que ffprobe no sabe leer.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import IO, TYPE_CHECKING, Any, cast

import numpy as np

from fanalysis.constants import (
    ASPECT_TOLERANCE,
    CHANNELS,
    FFMPEG,
    FFPROBE,
    HALF_TURN_DEG,
    MIN_DECODED_FRACTION,
    QUARTER_TURN_DEG,
    STDERR_TAIL_CHARS,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from io import BufferedReader
    from pathlib import Path

    import numpy.typing as npt
    from typing_extensions import Self

    from fanalysis.ficha import Ficha

__all__ = [
    "FfmpegFrameReader",
    "Segment",
    "VideoError",
    "VideoInfo",
    "expected_frames",
    "parse_probe",
    "probe",
    "read_window",
    "validate",
]


class VideoError(ValueError):
    """El vídeo no se puede analizar. El mensaje dice qué hacer."""


@dataclass(frozen=True, slots=True)
class VideoInfo:
    """Lo que dice ffprobe del vídeo, con el ancho y el alto ya como se ven."""

    path: Path
    duration_s: float
    width: int
    height: int
    """Con la rotación aplicada: un móvil en vertical da más alto que ancho."""
    fps: float
    rotation: int
    """Grados de los metadatos de rotación. ffmpeg los aplica solo al decodificar."""
    codec: str
    has_audio: bool

    @property
    def aspect(self) -> float:
        return self.width / self.height


@dataclass(frozen=True, slots=True)
class Segment:
    """El tramo del vídeo que se analiza, en segundos, y los avisos de la validación."""

    start_s: float
    end_s: float
    warnings: tuple[str, ...] = field(default=())

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def probe(path: Path) -> VideoInfo:
    """Lee el vídeo con ffprobe. Falla con un mensaje claro si no es un vídeo."""
    if not path.is_file():
        msg = f"no existe el vídeo: {path}"
        raise VideoError(msg)
    orden = [FFPROBE, "-v", "error", "-print_format", "json", "-show_format", "-show_streams"]
    salida = _run([*orden, str(path)])
    try:
        datos = json.loads(salida)
    except json.JSONDecodeError as error:
        msg = f"ffprobe no devolvió JSON para {path.name}"
        raise VideoError(msg) from error
    return parse_probe(datos, path)


def parse_probe(data: Mapping[str, Any], path: Path) -> VideoInfo:
    """Convierte la salida JSON de ffprobe en un `VideoInfo`."""
    streams = data.get("streams") or []
    videos = [s for s in streams if s.get("codec_type") == "video" and not _is_cover(s)]
    if not videos:
        msg = f"{path.name} no tiene pista de vídeo"
        raise VideoError(msg)
    video = videos[0]
    ancho, alto = int(video.get("width") or 0), int(video.get("height") or 0)
    if ancho <= 0 or alto <= 0:
        msg = f"{path.name}: ffprobe no sabe el tamaño del vídeo"
        raise VideoError(msg)
    rotacion = _rotation(video)
    if abs(rotacion) % HALF_TURN_DEG == QUARTER_TURN_DEG:
        ancho, alto = alto, ancho

    duracion = _float(data.get("format", {}).get("duration")) or _float(video.get("duration"))
    if duracion is None or duracion <= 0:
        msg = f"{path.name}: ffprobe no sabe cuánto dura el vídeo"
        raise VideoError(msg)
    fps = _rate(video.get("avg_frame_rate")) or _rate(video.get("r_frame_rate"))
    if fps is None:
        msg = f"{path.name}: ffprobe no sabe a cuántos frames por segundo va el vídeo"
        raise VideoError(msg)
    return VideoInfo(
        path=path,
        duration_s=duracion,
        width=ancho,
        height=alto,
        fps=fps,
        rotation=rotacion,
        codec=str(video.get("codec_name", "?")),
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
    )


def validate(
    info: VideoInfo, ficha: Ficha, *, start_s: float = 0.0, end_s: float | None = None
) -> Segment:
    """Comprueba que el vídeo y el tramo sirven para el modelo.

    Lo que hace el análisis inútil es un error (`VideoError`). Lo que lo empeora sin
    invalidarlo sale como aviso en el `Segment`, para enseñarlo antes de empezar.
    """
    fin = info.duration_s if end_s is None or end_s <= 0 else end_s
    if start_s < 0 or start_s >= info.duration_s:
        msg = f"el tramo empieza en {start_s:g} s y el vídeo dura {info.duration_s:.1f} s"
        raise VideoError(msg)
    if fin <= start_s:
        msg = f"el tramo acaba ({fin:g} s) antes de empezar ({start_s:g} s)"
        raise VideoError(msg)
    fin = min(fin, info.duration_s)
    if fin - start_s < ficha.window_s:
        msg = (
            f"el tramo dura {fin - start_s:.1f} s y el modelo mira ventanas de "
            f"{ficha.window_s:g} s: no le cabe ni una"
        )
        raise VideoError(msg)

    if info.height > info.width:
        msg = (
            f"el vídeo es vertical ({info.width}×{info.height}). El modelo se entrenó con "
            "retransmisiones apaisadas y, estirado, no ve nada: gíralo o recórtalo a horizontal"
        )
        raise VideoError(msg)

    avisos: list[str] = []
    modelo = ficha.width / ficha.height
    if abs(info.aspect - modelo) / modelo > ASPECT_TOLERANCE:
        avisos.append(
            f"la proporción del vídeo ({info.aspect:.2f}) no es la del modelo ({modelo:.2f}): "
            "los frames se estiran al reducirlos y el modelo acertará menos"
        )
    if info.fps < ficha.sample_fps:
        avisos.append(
            f"el vídeo va a {info.fps:.2f} fps y el modelo pide {ficha.sample_fps:g}: se "
            "repetirán frames y las jugadas se verán a saltos"
        )
    if info.height < ficha.height:
        avisos.append(
            f"el vídeo tiene {info.height} px de alto y el modelo trabaja a {ficha.height}: "
            "se amplía y se pierde detalle"
        )
    if not info.has_audio:
        avisos.append("el vídeo no tiene audio: los clips saldrán mudos")
    if info.rotation:
        avisos.append(f"el vídeo lleva {info.rotation}° de rotación; ffmpeg la aplica al leerlo")
    return Segment(start_s=start_s, end_s=fin, warnings=tuple(avisos))


def expected_frames(segment: Segment, fps: float) -> int:
    """Los frames que debería dar el tramo al ritmo del modelo."""
    return round(segment.duration_s * fps)


def read_window(info: VideoInfo, ficha: Ficha, start_s: float) -> npt.NDArray[np.uint8]:
    """Una ventana del modelo que empieza en `start_s`, para las pruebas antes del análisis."""
    tramo = Segment(start_s=start_s, end_s=start_s + ficha.window_s)
    frames = np.zeros((ficha.clip_frames, ficha.height, ficha.width, CHANNELS), np.uint8)
    leidos = 0
    with FfmpegFrameReader(info, ficha, tramo) as lector:
        while leidos < ficha.clip_frames and lector.read_into(frames[leidos]):
            leidos += 1
    if leidos < ficha.clip_frames:
        msg = (
            f"a partir de {start_s:.1f} s solo hay {leidos} frames y una ventana necesita "
            f"{ficha.clip_frames}"
        )
        raise VideoError(msg)
    return frames


class FfmpegFrameReader:
    """Los frames del tramo, reducidos al tamaño del modelo y a su ritmo, por una tubería.

    Implementa `FrameSource`. ffmpeg decodifica, remuestrea con `fps` y reduce; aquí solo
    se leen bytes. Es un gestor de contexto: al salir, ffmpeg se cierra pase lo que pase.
    """

    def __init__(self, info: VideoInfo, ficha: Ficha, segment: Segment) -> None:
        self._ficha = ficha
        self._segment = segment
        self._shape = (ficha.height, ficha.width, CHANNELS)
        self._frame_bytes = ficha.height * ficha.width * CHANNELS
        # Se lee aquí y se copia al destino solo si el frame llegó entero: un frame a medias
        # al final del vídeo no puede pisar el hueco más viejo de la ventana.
        self._staging = bytearray(self._frame_bytes)
        self._staging_view = np.frombuffer(self._staging, dtype=np.uint8).reshape(self._shape)
        filtros = (
            f"fps={ficha.sample_fps:g},"
            f"scale={ficha.width}:{ficha.height}:flags=bilinear,format=rgb24"
        )
        self._command = [
            FFMPEG, "-v", "error", "-nostdin",
            "-ss", f"{segment.start_s:.3f}", "-t", f"{segment.duration_s:.3f}",
            "-i", str(info.path),
            "-map", "0:v:0", "-an", "-sn", "-vf", filtros,
            "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
        ]  # fmt: skip
        self._process: subprocess.Popen[bytes] | None = None
        self._stderr: IO[bytes] | None = None
        self.frames_read = 0

    @property
    def command(self) -> list[str]:
        """La orden de ffmpeg, para enseñarla si algo falla."""
        return list(self._command)

    def __enter__(self) -> Self:
        # Los errores van a un fichero y no a una tubería: si ffmpeg escribiera mucho en
        # una tubería que nadie lee hasta el final, se quedaría bloqueado y esto también.
        self._stderr = tempfile.TemporaryFile()
        self._process = subprocess.Popen(self._command, stdout=subprocess.PIPE, stderr=self._stderr)
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def read_into(self, dst: npt.NDArray[np.uint8]) -> bool:
        """El siguiente frame en `dst`. `False` al acabar; si ffmpeg falló, `VideoError`."""
        if self._process is None or self._process.stdout is None:
            msg = "el lector no está abierto: úsalo con `with`"
            raise VideoError(msg)
        # Con el búfer por defecto, la salida de Popen es un BufferedReader.
        salida = cast("BufferedReader", self._process.stdout)
        leidos = salida.readinto(self._staging)
        while leidos and leidos < self._frame_bytes:
            # Una tubería entrega lo que tiene, no lo que se pide: se completa el frame.
            resto = memoryview(self._staging)[leidos:]
            nuevos = salida.readinto(resto)
            if not nuevos:
                break
            leidos += nuevos
        if leidos != self._frame_bytes:
            self._finish()
            return False
        np.copyto(dst, self._staging_view)
        self.frames_read += 1
        return True

    def check_complete(self) -> None:
        """Después de leer: ffmpeg entregó casi todos los frames que pedía el tramo."""
        esperados = expected_frames(self._segment, self._ficha.sample_fps)
        if self.frames_read < MIN_DECODED_FRACTION * esperados:
            msg = (
                f"ffmpeg entregó {self.frames_read} frames y el tramo pedía unos {esperados}: "
                "el vídeo está dañado o cortado, y lo que falta no se ha analizado"
            )
            raise VideoError(msg)

    def close(self) -> None:
        if self._process is not None:
            if self._process.poll() is None:
                self._process.kill()
                self._process.wait()
            if self._process.stdout is not None:
                self._process.stdout.close()
        if self._stderr is not None:
            self._stderr.close()
            self._stderr = None

    def _finish(self) -> None:
        """Al acabar los frames: si ffmpeg salió con error, se dice por qué."""
        if self._process is None:
            return
        codigo = self._process.wait()
        if codigo != 0:
            texto = ""
            if self._stderr is not None:
                self._stderr.seek(0)
                texto = self._stderr.read().decode("utf-8", "replace")[-STDERR_TAIL_CHARS:]
            msg = f"ffmpeg falló leyendo el vídeo (código {codigo}): {texto.strip()}"
            raise VideoError(msg)


# ---------------------------------------------------------------------- ayudas


def _run(command: list[str]) -> str:
    try:
        hecho = subprocess.run(command, capture_output=True, check=False)
    except FileNotFoundError as error:
        msg = f"no se encuentra {command[0]}: hace falta ffmpeg instalado y en el PATH"
        raise VideoError(msg) from error
    if hecho.returncode != 0:
        texto = hecho.stderr.decode("utf-8", "replace")[-STDERR_TAIL_CHARS:]
        msg = f"{command[0]} no pudo leer el fichero: {texto.strip()}"
        raise VideoError(msg)
    return hecho.stdout.decode("utf-8", "replace")


def _is_cover(stream: Mapping[str, Any]) -> bool:
    """Una carátula incrustada se declara como vídeo, pero es una imagen."""
    disposicion = stream.get("disposition") or {}
    return bool(disposicion.get("attached_pic"))


def _rotation(stream: Mapping[str, Any]) -> int:
    for dato in stream.get("side_data_list") or []:
        if "rotation" in dato:
            return int(float(dato["rotation"]))
    rotar = (stream.get("tags") or {}).get("rotate")
    return int(float(rotar)) if rotar is not None else 0


def _rate(value: Any) -> float | None:
    """`30000/1001` → 29,97. `0/0` y lo ilegible → `None`."""
    if not isinstance(value, str) or "/" not in value:
        return _float(value)
    numerador, denominador = value.split("/", 1)
    num, den = _float(numerador), _float(denominador)
    if num is None or not den:
        return None
    return num / den if num > 0 else None


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
