"""Dobles y fábricas que comparten los tests: fichas pequeñas, sesiones y fuentes falsas."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from fanalysis.ficha import Ficha, load_ficha

if TYPE_CHECKING:
    from fanalysis.spotting import Timeline

REGISTRO = Path(__file__).resolve().parents[1] / "models" / "registry.yaml"
FPS = 12.5
CLASES = ("normal_play", "SHOT", "HEADER", "GOAL")


def ficha_real() -> Ficha:
    return load_ficha(REGISTRO, "tdeed-snb")


def ficha_pequena(
    *,
    frames: int = 4,
    alto: int = 2,
    ancho: int = 3,
    clases: tuple[str, ...] = CLASES,
    layout: str = "NTCHW",
    **cambios: object,
) -> Ficha:
    """La ficha de T-DEED con una entrada diminuta, para no mover 428 MB por test."""
    tamanos = {"N": 1, "T": frames, "C": 3, "H": alto, "W": ancho}
    return replace(
        ficha_real(),
        input_shape=tuple(tamanos[eje] for eje in layout),
        layout=layout,
        classes=clases,
        **cambios,  # type: ignore[arg-type]
    )


@dataclass
class NodeArg:
    """Lo que devuelven `get_inputs` y `get_outputs` de onnxruntime: nombre y forma."""

    name: str
    shape: list[object]


@dataclass
class SesionFalsa:
    """Una sesión de onnxruntime sin modelo: guarda lo que le entra y devuelve lo pedido."""

    ficha: Ficha
    salidas: list[np.ndarray] | None = None
    forma_entrada: list[object] | None = None
    clases_logits: int | None = None
    proveedores: tuple[str, ...] = ("CPUExecutionProvider",)
    entradas: list[np.ndarray] = field(default_factory=list)

    def get_inputs(self) -> list[NodeArg]:
        forma = self.forma_entrada or list(self.ficha.input_shape)
        return [NodeArg(self.ficha.input_name, forma)]

    def get_outputs(self) -> list[NodeArg]:
        t = self.ficha.clip_frames
        c = self.clases_logits or len(self.ficha.classes)
        formas = {"logits": [1, t, c], "displacement": [1, t]}
        return [NodeArg(nombre, formas[sig]) for sig, nombre in self.ficha.outputs.items()]

    def get_providers(self) -> list[str]:
        return list(self.proveedores)

    def run(self, names: list[str], feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.entradas.append(feed[self.ficha.input_name].copy())
        if self.salidas is not None:
            return self.salidas
        t, c = self.ficha.clip_frames, len(self.ficha.classes)
        por_nombre = {
            self.ficha.outputs["logits"]: np.zeros((1, t, c), np.float32),
            self.ficha.outputs.get("displacement", "-"): np.zeros((1, t), np.float32),
        }
        return [por_nombre[n] for n in names]


def frame_con_indice(indice: int, alto: int, ancho: int) -> np.ndarray:
    """Un frame que lleva su número escrito en los dos primeros píxeles."""
    frame = np.zeros((alto, ancho, 3), np.uint8)
    frame[0, 0, 0], frame[0, 0, 1] = indice % 256, indice // 256
    return frame


def indice_de(frame: np.ndarray) -> int:
    return int(frame[0, 0, 0]) + 256 * int(frame[0, 0, 1])


@dataclass
class FuenteFalsa:
    """`FrameSource` con `total` frames numerados."""

    total: int
    alto: int
    ancho: int
    leidos: int = 0

    def read_into(self, dst: np.ndarray) -> bool:
        if self.leidos >= self.total:
            return False
        dst[...] = frame_con_indice(self.leidos, self.alto, self.ancho)
        self.leidos += 1
        return True


@dataclass
class CorredorFalso:
    """`ClipRunner` que apunta qué frames vio y enciende una clase en frames concretos.

    `eventos` va de frame absoluto a clase: en ese frame la clase gana con claridad. El
    resto del tiempo gana el fondo, que es la columna 0.
    """

    ficha: Ficha
    eventos: dict[int, str] = field(default_factory=dict)
    ventanas: list[list[int]] = field(default_factory=list)

    def infer(self, frames: np.ndarray) -> dict[str, np.ndarray]:
        indices = [indice_de(f) for f in frames]
        self.ventanas.append(indices)
        clases = [c.upper() for c in self.ficha.classes]
        logits = np.zeros((len(indices), len(clases)), np.float32)
        logits[:, 0] = 5.0
        for fila, absoluto in enumerate(indices):
            if absoluto in self.eventos:
                logits[fila, clases.index(self.eventos[absoluto].upper())] = 10.0
        return {"logits": logits, "displacement": np.zeros(len(indices), np.float32)}


def timeline_con(
    picos: dict[str, list[tuple[float, float]]],
    *,
    duracion_s: float = 120.0,
    offset_s: float = 0.0,
    kinds: tuple[str, ...] = ("normal_play", "shot", "header", "goal"),
) -> Timeline:
    """Una línea de tiempo a 12,5 fps con picos `(instante, puntuación)` por clase."""
    from fanalysis.spotting import Timeline  # noqa: PLC0415 - llega después que la ficha

    n = round(duracion_s * FPS)
    scores = np.zeros((n, len(kinds)), np.float32)
    scores[:, 0] = 1.0
    for kind, lista in picos.items():
        for t, s in lista:
            scores[round((t - offset_s) * FPS), kinds.index(kind)] = s
    return Timeline(
        scores=scores,
        support=np.ones(n, np.int32),
        kinds=kinds,
        fps=FPS,
        offset_s=offset_s,
        model="prueba",
        sha256="0" * 64,
        hop=25,
        windows=1,
    )
