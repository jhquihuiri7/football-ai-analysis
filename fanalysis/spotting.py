"""De frames a acciones: la ventana deslizante de T-DEED y sus picos.

Replica la inferencia original de T-DEED sobre un vídeo, que es con la que se midió la
línea base (6 goles de 6 en `videoGP.MP4`, PROGRESS de football-ai-training):

1. ventanas de `clip_frames` frames a `sample_fps`, que empiezan cada `hop` frames (25
   por defecto: 75 de 100 de solape, como su `inference.py`);
2. en cada ventana, la activación de la ficha por frame y después el desplazamiento
   temporal, agregado por máximo (`apply_displacement`);
3. cada frame del vídeo se queda con la **media** de las ventanas que lo vieron, contando
   solo las que le dejaron algo: las filas a cero que deja el desplazamiento no diluyen;
4. los picos, por clase, con un umbral y una supresión de ±`NMS_S` (`peaks`).

Dos diferencias, a propósito:
- T-DEED tira los frames del final que no llenan una ventana. Aquí la última ventana se
  alinea con el final, así que un gol en los últimos segundos también se ve.
- T-DEED rellena el principio con cinco frames negros. Aquí no: la primera ventana
  empieza en el primer frame.

`apply_displacement` y `peaks` son los de `libs/vision/spotter.py` de football-ai, con
la misma semántica. Si allí cambian, se cambian aquí.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from fanalysis.constants import (
    ACTION_THRESHOLD,
    BACKGROUND_KIND,
    CHANNELS,
    DEFAULT_HOP_FRAMES,
    DISPLACEMENT,
    LOGITS,
    MIN_HOP_FRAMES,
    NMS_S,
    SOFTMAX,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
    from pathlib import Path

    import numpy.typing as npt

    from fanalysis.ficha import Ficha
    from fanalysis.runner import ClipRunner

__all__ = [
    "Action",
    "FrameSource",
    "SpottingError",
    "Timeline",
    "activate",
    "aggregate",
    "apply_displacement",
    "check_hop",
    "check_threshold",
    "detect",
    "find_actions",
    "peaks",
    "window_starts",
]


class SpottingError(ValueError):
    """Algo impide detectar: parámetros fuera de rango, vídeo corto o salida rara."""


class FrameSource(Protocol):
    """Una fuente de frames ya reducidos al tamaño del modelo, en RGB y al ritmo de la ficha."""

    def read_into(self, dst: npt.NDArray[np.uint8]) -> bool:
        """Escribe el siguiente frame en `dst`. `False` al acabar, sin tocar `dst`."""
        ...


@dataclass(frozen=True, slots=True)
class Action:
    """Una acción que el modelo cree haber visto."""

    frame: int
    t_s: float
    """Segundos desde el principio del vídeo, no desde el principio del tramo analizado."""
    kind: str
    score: float


def check_threshold(threshold: float, name: str = "umbral") -> None:
    """Un umbral es una probabilidad: de 0 (excluido) a 1."""
    if not 0.0 < threshold <= 1.0:
        msg = f"el {name} tiene que estar entre 0 y 1 (0 excluido), y es {threshold}"
        raise SpottingError(msg)


def check_hop(hop: int, clip_frames: int) -> None:
    """El salto va de un frame a una ventana entera: más, y quedarían frames sin ver."""
    if not MIN_HOP_FRAMES <= hop <= clip_frames:
        msg = (
            f"el salto entre ventanas tiene que estar entre {MIN_HOP_FRAMES} y {clip_frames} "
            f"frames, y es {hop}: con más, habría frames que no vería ninguna ventana"
        )
        raise SpottingError(msg)


def window_starts(n_frames: int, clip_frames: int, hop: int) -> list[int]:
    """Dónde empieza cada ventana, con la última alineada al final del vídeo."""
    check_hop(hop, clip_frames)
    if n_frames < clip_frames:
        msg = f"hay {n_frames} frames y una ventana necesita {clip_frames}"
        raise SpottingError(msg)
    starts = list(range(0, n_frames - clip_frames + 1, hop))
    if starts[-1] != n_frames - clip_frames:
        starts.append(n_frames - clip_frames)
    return starts


def activate(logits: npt.NDArray[np.float32], activation: str) -> npt.NDArray[np.float32]:
    """Logits a puntuaciones, con la activación que diga la ficha."""
    if activation == SOFTMAX:
        estables = logits - logits.max(axis=-1, keepdims=True)
        exponenciales = np.exp(estables)
        return np.asarray(exponenciales / exponenciales.sum(axis=-1, keepdims=True), np.float32)
    return np.asarray(1.0 / (1.0 + np.exp(-logits)), np.float32)


def apply_displacement(
    scores: npt.NDArray[np.float32], displacement: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Mueve cada frame a donde el modelo dice que está de verdad el evento.

    T-DEED no puntúa el frame donde ve la jugada: puntúa el que mira y dice a cuántos
    frames está el evento. Sin aplicarlo, todo sale corrido hasta cuatro frames (un tercio
    de segundo). Se agrega por máximo, no por suma, y va después de la activación, como
    en el post-proceso original. Lo que se sale de la ventana se queda en su borde.
    """
    if scores.shape[0] != displacement.shape[0]:
        msg = (
            f"el desplazamiento tiene {displacement.shape[0]} frames y las puntuaciones "
            f"{scores.shape[0]}"
        )
        raise SpottingError(msg)
    ultimo = scores.shape[0] - 1
    destino = np.clip(np.arange(scores.shape[0]) - np.rint(displacement).astype(np.intp), 0, ultimo)
    movidas = np.zeros_like(scores)
    np.maximum.at(movidas, destino, scores)
    return movidas


def aggregate(
    windows: Iterable[tuple[int, npt.NDArray[np.float32]]], n_frames: int, n_classes: int
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]:
    """La media por frame de las ventanas que lo vieron, y cuántas fueron.

    Una fila a cero —un frame al que el desplazamiento no le dejó nada— no cuenta como
    ventana que lo vio, igual que en T-DEED: si contara, diluiría la media del frame.
    """
    total = np.zeros((n_frames, n_classes), dtype=np.float32)
    soporte = np.zeros(n_frames, dtype=np.int32)
    for start, scores in windows:
        end = start + scores.shape[0]
        if start < 0 or end > n_frames or scores.shape[1:] != (n_classes,):
            msg = f"una ventana de forma {scores.shape} en {start} no cabe en {n_frames} frames"
            raise SpottingError(msg)
        total[start:end] += scores
        soporte[start:end] += scores.sum(axis=1) != 0
    media = total / np.maximum(soporte, 1)[:, None]
    return media.astype(np.float32), soporte


def peaks(
    scores: npt.NDArray[np.float32],
    kinds: Sequence[str],
    *,
    threshold: float = ACTION_THRESHOLD,
    min_gap_frames: int = 1,
    background: str = BACKGROUND_KIND,
) -> list[tuple[int, str, float]]:
    """Máximos locales por clase, ya filtrados y separados: `(frame, clase, score)`.

    Un evento enciende una cresta de frames, no uno. Se recorren por puntuación
    descendente, y cada elegido apaga a los que tiene a menos de `min_gap_frames`: entre
    dos crestas pegadas gana la más segura, no la primera. El fondo no da acciones.

    Lo apagado se marca en una máscara en vez de comparar cada candidato con todos los
    elegidos: en un partido entero, PASS pasa de 0,2 en decenas de miles de frames.
    """
    if scores.ndim != 2 or scores.shape[1] != len(kinds):  # noqa: PLR2004 - (frames, clases)
        msg = f"se esperaban puntuaciones (frames, {len(kinds)}) y llegó {scores.shape}"
        raise SpottingError(msg)
    radio = min_gap_frames - 1
    encontrados: list[tuple[int, str, float]] = []
    for indice, nombre in enumerate(kinds):
        if nombre == background:
            continue
        columna = scores[:, indice]
        candidatos = sorted(
            (int(t) for t in np.flatnonzero(columna >= threshold)),
            key=lambda t: float(columna[t]),
            reverse=True,
        )
        apagados = np.zeros(columna.shape[0], dtype=bool)
        for frame in candidatos:
            if apagados[frame]:
                continue
            apagados[max(0, frame - radio) : frame + radio + 1] = True
            encontrados.append((frame, nombre, float(columna[frame])))
    return sorted(encontrados)


@dataclass(frozen=True)
class Timeline:
    """Las puntuaciones de cada clase en cada frame analizado, y cómo se obtuvieron."""

    scores: npt.NDArray[np.float32]
    """`(frames, clases)`, la media de las ventanas."""
    support: npt.NDArray[np.int32]
    """`(frames,)`, cuántas ventanas puntuaron cada frame."""
    kinds: tuple[str, ...]
    fps: float
    offset_s: float
    """Dónde empieza el tramo analizado dentro del vídeo, en segundos."""
    model: str
    sha256: str
    hop: int
    windows: int
    source: str = ""
    """El nombre del vídeo, para no reutilizar por error las puntuaciones de otro."""

    def __post_init__(self) -> None:
        frames = self.scores.shape[0]
        if self.scores.shape != (frames, len(self.kinds)) or self.support.shape != (frames,):
            msg = f"puntuaciones {self.scores.shape} y soporte {self.support.shape} no casan"
            raise SpottingError(msg)

    @property
    def n_frames(self) -> int:
        return int(self.scores.shape[0])

    @property
    def end_s(self) -> float:
        """Dónde acaba el tramo analizado dentro del vídeo, en segundos."""
        return self.time_of(self.n_frames)

    def time_of(self, frame: float) -> float:
        """Segundos desde el principio del vídeo de un frame del tramo."""
        return self.offset_s + frame / self.fps

    def frame_of(self, t_s: float) -> int:
        """El frame del tramo más cercano a un instante del vídeo, dentro del tramo."""
        return int(np.clip(round((t_s - self.offset_s) * self.fps), 0, self.n_frames - 1))

    def column(self, kind: str) -> npt.NDArray[np.float32]:
        """Las puntuaciones de una clase a lo largo del tramo."""
        return self.scores[:, self.kinds.index(kind)]

    def save(self, path: Path) -> Path:
        """Guarda las puntuaciones: recalcularlas en un partido cuesta media hora de GPU."""
        meta = {
            "kinds": list(self.kinds),
            "fps": self.fps,
            "offset_s": self.offset_s,
            "model": self.model,
            "sha256": self.sha256,
            "hop": self.hop,
            "windows": self.windows,
            "source": self.source,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fichero:
            np.savez_compressed(
                fichero, scores=self.scores, support=self.support, meta=np.array(json.dumps(meta))
            )
        return path

    @classmethod
    def load(cls, path: Path) -> Timeline:
        with np.load(path, allow_pickle=False) as datos:
            meta = json.loads(str(datos["meta"]))
            return cls(
                scores=datos["scores"].astype(np.float32),
                support=datos["support"].astype(np.int32),
                kinds=tuple(meta["kinds"]),
                fps=float(meta["fps"]),
                offset_s=float(meta["offset_s"]),
                model=str(meta["model"]),
                sha256=str(meta["sha256"]),
                hop=int(meta["hop"]),
                windows=int(meta["windows"]),
                source=str(meta.get("source", "")),
            )


def detect(
    source: FrameSource,
    runner: ClipRunner,
    ficha: Ficha,
    *,
    hop: int = DEFAULT_HOP_FRAMES,
    offset_s: float = 0.0,
    name: str = "",
    on_window: Callable[[int, int], None] | None = None,
) -> Timeline:
    """Recorre la fuente con la ventana deslizante y devuelve las puntuaciones por frame.

    Los frames no se guardan todos —un partido serían decenas de GB—: solo la última
    ventana, en un anillo. `on_window(ventanas, frames)` avisa después de cada inferencia,
    para enseñar el progreso.
    """
    t = ficha.clip_frames
    check_hop(hop, t)
    anillo = np.empty((t, ficha.height, ficha.width, CHANNELS), dtype=np.uint8)
    ordenada = np.empty_like(anillo)
    ventanas: list[tuple[int, npt.NDArray[np.float32]]] = []
    leidos = 0
    ultima: int | None = None

    def inferir() -> None:
        nonlocal ultima
        inicio = leidos - t
        corte = leidos % t
        ordenada[: t - corte] = anillo[corte:]
        ordenada[t - corte :] = anillo[:corte]
        ventanas.append((inicio, _window_scores(runner.infer(ordenada), ficha)))
        ultima = inicio
        if on_window is not None:
            on_window(len(ventanas), leidos)

    while source.read_into(anillo[leidos % t]):
        leidos += 1
        if leidos >= t and (leidos - t) % hop == 0:
            inferir()
    if leidos < t:
        msg = (
            f"el tramo da {leidos} frames a {ficha.sample_fps:g} fps y el modelo necesita "
            f"ventanas de {t} ({ficha.window_s:g} s)"
        )
        raise SpottingError(msg)
    if ultima != leidos - t:
        inferir()

    medias, soporte = aggregate(ventanas, leidos, len(ficha.classes))
    return Timeline(
        scores=medias,
        support=soporte,
        kinds=ficha.kinds,
        fps=ficha.sample_fps,
        offset_s=offset_s,
        model=ficha.name,
        sha256=ficha.sha256,
        hop=hop,
        windows=len(ventanas),
        source=name,
    )


def find_actions(
    timeline: Timeline,
    *,
    threshold: float = ACTION_THRESHOLD,
    nms_s: float = NMS_S,
    kinds: Sequence[str] | None = None,
) -> list[Action]:
    """Las acciones del tramo, ordenadas por tiempo. `kinds` limita a unas clases."""
    check_threshold(threshold)
    if nms_s < 0:
        msg = f"la separación entre picos no puede ser negativa: {nms_s}"
        raise SpottingError(msg)
    elegidas = list(timeline.kinds) if kinds is None else list(kinds)
    desconocidas = [k for k in elegidas if k not in timeline.kinds]
    if desconocidas:
        msg = f"el modelo no tiene las clases {desconocidas} (tiene: {', '.join(timeline.kinds)})"
        raise SpottingError(msg)
    columnas = [timeline.kinds.index(k) for k in elegidas]
    separacion = max(1, round(nms_s * timeline.fps))
    return [
        Action(frame=frame, t_s=timeline.time_of(frame), kind=kind, score=score)
        for frame, kind, score in peaks(
            timeline.scores[:, columnas], elegidas, threshold=threshold, min_gap_frames=separacion
        )
    ]


def _window_scores(
    salidas: dict[str, npt.NDArray[np.float32]], ficha: Ficha
) -> npt.NDArray[np.float32]:
    """Las puntuaciones de una ventana, con la forma comprobada y el desplazamiento aplicado."""
    forma = (ficha.clip_frames, len(ficha.classes))
    logits = salidas.get(LOGITS)
    if logits is None or logits.shape != forma:
        recibida = None if logits is None else logits.shape
        msg = f"{ficha.name}: se esperaban logits {forma} por ventana y llegó {recibida}"
        raise SpottingError(msg)
    scores = activate(logits, ficha.activation)
    desplazamiento = salidas.get(DISPLACEMENT)
    if DISPLACEMENT in ficha.outputs:
        if desplazamiento is None or desplazamiento.shape != (ficha.clip_frames,):
            recibida = None if desplazamiento is None else desplazamiento.shape
            msg = f"{ficha.name}: se esperaba un desplazamiento ({forma[0]},) y llegó {recibida}"
            raise SpottingError(msg)
        scores = apply_displacement(scores, desplazamiento)
    return scores
