"""La ficha de un modelo: cómo se ejecuta, leído de `models/registry.yaml`.

Un `.onnx` no dice cómo usarse. El orden de las clases, el color de entrada, la
normalización y qué significa cada salida no están dentro del fichero, y adivinarlos
falla **en silencio**: el modelo traga lo que le des si la forma cuadra. Por eso aquí se
lee la ficha que declara football-ai y se comprueba contra el `.onnx` antes de usarlo.

Solo se lee lo que necesita un modelo temporal (un *spotter*). El lector completo, con
los detectores de imagen y Core ML, es `libs/vision/registry.py` de football-ai.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import yaml

from fanalysis.constants import (
    ACTIVATIONS,
    BATCH_AXIS,
    BGR,
    CHANNEL_AXIS,
    CHANNELS,
    DISPLACEMENT,
    GOAL_KIND,
    HASH_CHUNK_BYTES,
    HEIGHT_AXIS,
    LOGITS,
    RGB,
    TIME_AXIS,
    WIDTH_AXIS,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

__all__ = [
    "Ficha",
    "FichaError",
    "canonical",
    "check_session",
    "load_ficha",
    "sha256_of",
    "verify_model",
]

_VIDEO_AXES = frozenset({BATCH_AXIS, TIME_AXIS, CHANNEL_AXIS, HEIGHT_AXIS, WIDTH_AXIS})


class FichaError(ValueError):
    """La ficha no se puede usar, o no casa con el `.onnx`. El mensaje dice por qué."""


def canonical(name: str) -> str:
    """El nombre de una clase en el vocabulario del repo: `HIGH PASS` → `high_pass`."""
    return "_".join(name.strip().lower().split())


def sha256_of(path: Path) -> str:
    """El sha256 de un fichero, leído por trozos."""
    digest = hashlib.sha256()
    with path.open("rb") as fichero:
        while bloque := fichero.read(HASH_CHUNK_BYTES):
            digest.update(bloque)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class Ficha:
    """Lo que hace falta para ejecutar un modelo temporal y leer su salida."""

    name: str
    version: str
    sha256: str
    license: str
    input_name: str
    input_shape: tuple[int, ...]
    layout: str
    color: str
    scale: float
    mean: tuple[float, ...]
    std: tuple[float, ...]
    sample_fps: float
    activation: str
    outputs: Mapping[str, str]
    """Nombre de cada salida por su significado (`logits`, `displacement`)."""
    classes: tuple[str, ...]

    @property
    def clip_frames(self) -> int:
        """Frames de una ventana."""
        return self.input_shape[self.layout.index(TIME_AXIS)]

    @property
    def height(self) -> int:
        return self.input_shape[self.layout.index(HEIGHT_AXIS)]

    @property
    def width(self) -> int:
        return self.input_shape[self.layout.index(WIDTH_AXIS)]

    @property
    def window_s(self) -> float:
        """Segundos de partido que ve una ventana: 8 en T-DEED."""
        return self.clip_frames / self.sample_fps

    @property
    def kinds(self) -> tuple[str, ...]:
        """Las clases en el vocabulario del repo, en el orden de la salida."""
        return tuple(canonical(c) for c in self.classes)


def load_ficha(registry: Path, name: str) -> Ficha:
    """Lee la ficha `name` del registro y comprueba que se puede usar aquí."""
    if not registry.is_file():
        msg = f"no existe el registro de modelos: {registry}"
        raise FichaError(msg)
    datos = yaml.safe_load(registry.read_text(encoding="utf-8"))
    modelos = _map(_map(datos, str(registry)).get("models"), f"{registry}: models")
    if name not in modelos:
        msg = f"el registro no tiene el modelo «{name}» (hay: {', '.join(modelos) or 'ninguno'})"
        raise FichaError(msg)
    return parse_ficha(name, modelos[name])


def parse_ficha(name: str, value: Any) -> Ficha:
    """Convierte un bloque del registro en una `Ficha`, o explica qué le falta."""
    donde = f"ficha {name}"
    bloque = _map(value, donde)
    entrada = _map(bloque.get("input"), f"{donde}: input")
    onnx = _map(_map(bloque.get("artifacts"), f"{donde}: artifacts").get("onnx"), donde)

    forma = tuple(_ints(entrada, "shape", donde))
    layout = _str(entrada, "layout", donde)
    if set(layout) != _VIDEO_AXES or len(layout) != len(forma) or layout[0] != BATCH_AXIS:
        msg = (
            f"{donde}: el layout «{layout}» no describe un clip de vídeo de {len(forma)} ejes "
            "con el batch delante. Aquí solo se ejecutan modelos temporales (NTCHW o NCTHW)"
        )
        raise FichaError(msg)
    if forma[layout.index(BATCH_AXIS)] != 1 or forma[layout.index(CHANNEL_AXIS)] != CHANNELS:
        msg = f"{donde}: se espera batch 1 y {CHANNELS} canales, y la forma es {list(forma)}"
        raise FichaError(msg)

    color = _str(entrada, "color", donde).upper()
    if color not in (RGB, BGR):
        msg = f"{donde}: el color «{color}» no es {RGB} ni {BGR}"
        raise FichaError(msg)
    fps = entrada.get("sample_fps")
    if (
        not isinstance(fps, int | float)
        or isinstance(fps, bool)
        or not math.isfinite(fps)
        or fps <= 0
    ):
        msg = (
            f"{donde}: un modelo temporal necesita `sample_fps` positivo. Darle frames a otro "
            "ritmo no da error, solo acierta menos"
        )
        raise FichaError(msg)
    activacion = _str(bloque, "activation", donde)
    if activacion not in ACTIVATIONS:
        msg = f"{donde}: la activación «{activacion}» no es {' ni '.join(ACTIVATIONS)}"
        raise FichaError(msg)

    salidas = _outputs(bloque.get("outputs"), donde)
    clases = tuple(_strs(bloque, "classes", donde))
    if canonical(GOAL_KIND) not in (canonical(c) for c in clases):
        msg = f"{donde}: no tiene clase de gol, y sin ella no hay highlights (clases: {clases})"
        raise FichaError(msg)

    ficha = Ficha(
        name=name,
        version=str(bloque.get("version", "")),
        sha256=_str(onnx, "sha256", donde).lower(),
        license=" ".join(str(bloque.get("license", "")).split()),
        input_name=_str(entrada, "name", donde),
        input_shape=forma,
        layout=layout,
        color=color,
        scale=_positive(entrada, "scale", donde),
        mean=tuple(_floats(entrada, "mean", donde)),
        std=tuple(_floats(entrada, "std", donde)),
        sample_fps=float(fps),
        activation=activacion,
        outputs=salidas,
        classes=clases,
    )
    if len(ficha.mean) != CHANNELS or len(ficha.std) != CHANNELS or min(ficha.std) <= 0:
        msg = f"{donde}: `mean` y `std` llevan {CHANNELS} valores, y `std` positivos"
        raise FichaError(msg)
    return ficha


def verify_model(ficha: Ficha, model_path: Path) -> None:
    """El `.onnx` existe y es el de la ficha, byte a byte."""
    if not model_path.is_file():
        msg = f"no existe el modelo: {model_path}"
        raise FichaError(msg)
    real = sha256_of(model_path)
    if real != ficha.sha256:
        msg = (
            f"{model_path.name} no es el modelo de la ficha {ficha.name}: sha256 {real[:12]}… "
            f"frente a {ficha.sha256[:12]}…. Si es otro export, necesita su propia ficha"
        )
        raise FichaError(msg)


def check_session(ficha: Ficha, inputs: Sequence[Any], outputs: Sequence[Any]) -> None:
    """La entrada y las salidas de la sesión de onnxruntime son las de la ficha.

    Se comprueba al abrir y no al primer clip: una ficha equivocada no rompe nada visible,
    y el resultado sale malo sin que nadie sepa por qué. `inputs` y `outputs` son los
    `NodeArg` de `session.get_inputs()` y `session.get_outputs()`: con `name` y `shape`.
    """
    if len(inputs) != 1:
        msg = f"{ficha.name}: el .onnx tiene {len(inputs)} entradas y aquí se espera una"
        raise FichaError(msg)
    entrada = inputs[0]
    real = list(entrada.shape)
    declarada = list(ficha.input_shape)
    # Un eje dinámico llega como texto o `None`: ese no se compara, lo fija la ficha.
    distinta = len(real) != len(declarada) or any(
        isinstance(r, int) and r != d for r, d in zip(real, declarada, strict=False)
    )
    if entrada.name != ficha.input_name or distinta:
        msg = (
            f"{ficha.name}: la ficha dice entrada «{ficha.input_name}» {declarada} y el .onnx "
            f"trae «{entrada.name}» {list(entrada.shape)}"
        )
        raise FichaError(msg)

    por_nombre = {salida.name: salida for salida in outputs}
    for significado, nombre in ficha.outputs.items():
        if nombre not in por_nombre:
            msg = (
                f"{ficha.name}: la ficha declara la salida «{nombre}» ({significado}) y el .onnx "
                f"trae {', '.join(sorted(por_nombre))}"
            )
            raise FichaError(msg)
    clases = por_nombre[ficha.outputs[LOGITS]].shape[-1]
    if isinstance(clases, int) and clases != len(ficha.classes):
        msg = (
            f"{ficha.name}: los logits del .onnx tienen {clases} clases y la ficha nombra "
            f"{len(ficha.classes)}. Con el orden equivocado, las etiquetas salen cambiadas"
        )
        raise FichaError(msg)


# ---------------------------------------------------------------------- lectura


def _outputs(value: Any, where: str) -> dict[str, str]:
    if not isinstance(value, list) or not value:
        msg = f"{where}: `outputs` tiene que ser una lista con al menos los logits"
        raise FichaError(msg)
    salidas: dict[str, str] = {}
    for n, bloque in enumerate(value):
        datos = _map(bloque, f"{where}: outputs[{n}]")
        significado = _str(datos, "meaning", where)
        if significado in salidas:
            msg = f"{where}: dos salidas con el significado «{significado}»"
            raise FichaError(msg)
        salidas[significado] = _str(datos, "name", where)
    if LOGITS not in salidas:
        msg = f"{where}: ninguna salida significa «{LOGITS}» (hay {', '.join(salidas)})"
        raise FichaError(msg)
    desconocidas = set(salidas) - {LOGITS, DISPLACEMENT}
    if desconocidas:
        msg = f"{where}: este repo no sabe usar las salidas {', '.join(sorted(desconocidas))}"
        raise FichaError(msg)
    return salidas


def _map(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        msg = f"{where}: se esperaba un bloque con claves"
        raise FichaError(msg)
    return value


def _str(data: Mapping[str, Any], key: str, where: str) -> str:
    valor = data.get(key)
    if not isinstance(valor, str) or not valor.strip():
        msg = f"{where}: falta `{key}` o no es texto"
        raise FichaError(msg)
    return valor.strip()


def _strs(data: Mapping[str, Any], key: str, where: str) -> list[str]:
    valor = data.get(key)
    if not isinstance(valor, list) or not valor or not all(isinstance(v, str) for v in valor):
        msg = f"{where}: `{key}` tiene que ser una lista de textos"
        raise FichaError(msg)
    if len({canonical(v) for v in valor}) != len(valor):
        msg = f"{where}: `{key}` repite nombres"
        raise FichaError(msg)
    return list(valor)


def _ints(data: Mapping[str, Any], key: str, where: str) -> list[int]:
    valor = data.get(key)
    if (
        not isinstance(valor, list)
        or not valor
        or not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in valor)
    ):
        msg = f"{where}: `{key}` tiene que ser una lista de enteros positivos"
        raise FichaError(msg)
    return list(valor)


def _floats(data: Mapping[str, Any], key: str, where: str) -> list[float]:
    valor = data.get(key)
    if not isinstance(valor, list) or not all(
        isinstance(v, int | float) and not isinstance(v, bool) for v in valor
    ):
        msg = f"{where}: `{key}` tiene que ser una lista de números"
        raise FichaError(msg)
    return [float(v) for v in valor]


def _positive(data: Mapping[str, Any], key: str, where: str) -> float:
    valor = data.get(key)
    if not isinstance(valor, int | float) or isinstance(valor, bool) or valor <= 0:
        msg = f"{where}: `{key}` tiene que ser un número positivo"
        raise FichaError(msg)
    return float(valor)
