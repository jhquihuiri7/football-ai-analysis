"""Ejecuta el `.onnx` de un modelo temporal sobre una ventana de frames.

Todo lo que hace falta para ejecutarlo sale de la ficha: la forma, el orden de los ejes,
el color y la normalización. Y al abrir la sesión se comprueba que la ficha y el `.onnx`
dicen lo mismo (`check_session`), porque si no coinciden el síntoma no es un error sino
un resultado malo.

El tensor de entrada se reserva una sola vez: una ventana de T-DEED son 428 MB en fp32, y
el bucle la rellena miles de veces en un partido.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from fanalysis.constants import (
    BATCH_AXIS,
    BGR,
    CHANNEL_AXIS,
    CHANNELS,
    CPU_PROVIDER,
    CUDA_PROVIDER,
    HEIGHT_AXIS,
    TIME_AXIS,
    WIDTH_AXIS,
)
from fanalysis.ficha import Ficha, FichaError, check_session, verify_model

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt

__all__ = ["ClipRunner", "OnnxClipRunner", "select_providers"]


class ClipRunner(Protocol):
    """Lo que el detector de acciones necesita de un modelo: una ventana dentro, salidas fuera.

    Es un `Protocol` para que los tests usen un doble sin `.onnx` delante.
    """

    def infer(self, frames: npt.NDArray[np.uint8]) -> dict[str, npt.NDArray[np.float32]]:
        """Salidas por significado y sin el eje del batch, a partir de `(T, H, W, 3)` en RGB."""
        ...


def select_providers(available: list[str]) -> list[str]:
    """CUDA si onnxruntime la ofrece, y la CPU detrás como respaldo."""
    if CUDA_PROVIDER in available:
        return [CUDA_PROVIDER, CPU_PROVIDER]
    return [CPU_PROVIDER]


class OnnxClipRunner:
    """Un `.onnx` temporal detrás de la interfaz `ClipRunner`."""

    def __init__(
        self,
        ficha: Ficha,
        model_path: Path,
        *,
        providers: list[str] | None = None,
        session: Any = None,
    ) -> None:
        """Verifica el sha256, abre la sesión y la comprueba contra la ficha.

        `session` es para los tests: un objeto con `get_inputs`, `get_outputs`,
        `get_providers` y `run`. Sin él, se abre onnxruntime de verdad.
        """
        if session is None:
            verify_model(ficha, model_path)
            session = _open_session(model_path, providers)
        check_session(ficha, session.get_inputs(), session.get_outputs())
        self._ficha = ficha
        self._session = session
        self.provider: str = session.get_providers()[0]
        """El proveedor con el que abrió de verdad. Caer a CPU no avisa: hay que mirarlo."""

        # La ficha garantiza el batch delante (`parse_ficha`).
        sin_batch = ficha.layout.removeprefix(BATCH_AXIS)
        self._tensor = np.empty(ficha.input_shape, dtype=np.float32)
        # Una vista (T, C, H, W) del tensor, sea cual sea su layout: escribir en ella es
        # escribir en el tensor. Así NTCHW y NCTHW se rellenan con el mismo código.
        self._view = np.moveaxis(
            self._tensor[0],
            [sin_batch.index(e) for e in (TIME_AXIS, CHANNEL_AXIS, HEIGHT_AXIS, WIDTH_AXIS)],
            [0, 1, 2, 3],
        )
        self._identity = ficha.scale == 1.0 and set(ficha.mean) == {0.0} and set(ficha.std) == {1.0}
        forma_canal = (1, CHANNELS, 1, 1)
        self._mean = np.asarray(ficha.mean, dtype=np.float32).reshape(forma_canal)
        self._std = np.asarray(ficha.std, dtype=np.float32).reshape(forma_canal)
        self._names = list(ficha.outputs.values())
        self._meanings = list(ficha.outputs)

    @property
    def ficha(self) -> Ficha:
        return self._ficha

    def infer(self, frames: npt.NDArray[np.uint8]) -> dict[str, npt.NDArray[np.float32]]:
        """Rellena el tensor con la ventana y ejecuta la red."""
        ficha = self._ficha
        esperada = (ficha.clip_frames, ficha.height, ficha.width, CHANNELS)
        if frames.shape != esperada or frames.dtype != np.uint8:
            msg = f"{ficha.name} espera una ventana {esperada} uint8 y llegó {frames.shape}"
            raise FichaError(msg)

        origen = frames[..., ::-1] if ficha.color == BGR else frames
        # (T, H, W, C) → (T, C, H, W) directamente sobre la vista, sin copias intermedias.
        self._view[...] = origen.transpose(0, 3, 1, 2)
        if not self._identity:
            self._view /= ficha.scale
            self._view -= self._mean
            self._view /= self._std

        crudas = self._session.run(self._names, {ficha.input_name: self._tensor})
        salidas: dict[str, npt.NDArray[np.float32]] = {}
        for significado, valor in zip(self._meanings, crudas, strict=True):
            array = np.asarray(valor, dtype=np.float32)
            if array.ndim and array.shape[0] == 1:
                array = array[0]
            if not np.isfinite(array).all():
                msg = f"{ficha.name}: la salida «{significado}» trae NaN o infinitos"
                raise FichaError(msg)
            salidas[significado] = array
        return salidas


def _open_session(model_path: Path, providers: list[str] | None) -> Any:
    """Abre onnxruntime con CUDA si la hay. Importa tarde: los tests no lo necesitan."""
    import onnxruntime as ort  # noqa: PLC0415

    disponibles = list(ort.get_available_providers())
    # Desde onnxruntime 1.21, `preload_dlls` carga CUDA y cuDNN de las ruedas de NVIDIA que
    # Colab instala con torch. Sin él, a veces la sesión cae a CPU sin decir nada; si aun
    # así falla, `provider` lo enseña y el cuaderno avisa.
    precarga = getattr(ort, "preload_dlls", None)
    if precarga is not None and CUDA_PROVIDER in disponibles:
        with contextlib.suppress(OSError, RuntimeError):
            precarga()
    elegidos = providers or select_providers(disponibles)
    return ort.InferenceSession(str(model_path), providers=elegidos)
