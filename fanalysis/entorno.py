"""Qué onnxruntime instalar en Colab, según la CUDA que admite el driver de su GPU.

Desde la 1.27, el `onnxruntime-gpu` de PyPI se compila contra CUDA 13, y hasta la 1.26
contra CUDA 12 (documentación del CUDA Execution Provider). Si el driver no llega a la CUDA
del paquete, la sesión no falla: cae a la CPU sin avisar y va unas 25 veces más lenta.
Por eso la versión se elige mirando el driver, que es lo único que no se puede cambiar
desde el cuaderno.

Los extras `[cuda,cudnn]` traen las librerías de NVIDIA que ese paquete necesita, como
ruedas de pip. Así no depende de la CUDA que tenga instalada Colab, y `preload_dlls`
(desde la 1.21) las carga al abrir la sesión.

Este módulo solo usa la biblioteca estándar: el cuaderno lo importa desde el clon antes
de instalar nada.
"""

from __future__ import annotations

import re
from typing import Final

__all__ = [
    "CPU_REQUIREMENT",
    "EntornoError",
    "driver_cuda_version",
    "onnxruntime_requirement",
]

CPU_REQUIREMENT: Final = "onnxruntime>=1.21"
"""Sin GPU, el onnxruntime de siempre."""

CUDA13_MAJOR: Final = 13
CUDA12_MAJOR: Final = 12
"""Las dos CUDA para las que hay `onnxruntime-gpu` en PyPI con `preload_dlls`."""

CUDA13_REQUIREMENT: Final = "onnxruntime-gpu[cuda,cudnn]>=1.27"
"""Para drivers de CUDA 13 o más: desde la 1.27, el paquete de PyPI es de CUDA 13.0."""

CUDA12_REQUIREMENT: Final = "onnxruntime-gpu[cuda,cudnn]>=1.21,<1.27"
"""Para drivers de CUDA 12: de la 1.21 (la primera con `preload_dlls`) a la 1.26 (la
última de PyPI compilada contra CUDA 12)."""

_CUDA_VERSION = re.compile(r"CUDA Version:\s*(\d+)\.(\d+)")


class EntornoError(RuntimeError):
    """El entorno no puede ejecutar el modelo en su GPU. El mensaje dice por qué."""


def driver_cuda_version(nvidia_smi: str) -> tuple[int, int] | None:
    """La CUDA más alta que admite el driver, de la cabecera de `nvidia-smi`. `None` sin GPU."""
    encontrada = _CUDA_VERSION.search(nvidia_smi)
    if encontrada is None:
        return None
    return int(encontrada.group(1)), int(encontrada.group(2))


def onnxruntime_requirement(cuda: tuple[int, int] | None) -> str:
    """El paquete de onnxruntime que casa con esa CUDA, listo para `pip install`."""
    if cuda is None:
        return CPU_REQUIREMENT
    mayor, menor = cuda
    if mayor >= CUDA13_MAJOR:
        return CUDA13_REQUIREMENT
    if mayor == CUDA12_MAJOR:
        return CUDA12_REQUIREMENT
    msg = (
        f"el driver de la GPU solo admite CUDA {mayor}.{menor}, y no hay onnxruntime-gpu en "
        "PyPI para ella: cambia de entorno o ejecuta en CPU"
    )
    raise EntornoError(msg)
