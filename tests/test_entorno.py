from __future__ import annotations

import pytest

from fanalysis.entorno import (
    CPU_REQUIREMENT,
    EntornoError,
    driver_cuda_version,
    onnxruntime_requirement,
)

CABECERA_T4 = """
Mon Oct 19 10:00:00 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
"""


def test_la_cuda_del_driver_sale_de_la_cabecera():
    assert driver_cuda_version(CABECERA_T4) == (12, 4)
    assert driver_cuda_version("Driver Version: 580.82   CUDA Version: 13.0") == (13, 0)
    assert driver_cuda_version("") is None
    assert driver_cuda_version("NVIDIA-SMI has failed because it couldn't communicate") is None


@pytest.mark.parametrize(
    ("cuda", "esperado"),
    [
        (None, CPU_REQUIREMENT),
        ((12, 4), "onnxruntime-gpu[cuda,cudnn]>=1.21,<1.27"),
        ((12, 8), "onnxruntime-gpu[cuda,cudnn]>=1.21,<1.27"),
        ((13, 0), "onnxruntime-gpu[cuda,cudnn]>=1.27"),
        ((13, 2), "onnxruntime-gpu[cuda,cudnn]>=1.27"),
    ],
)
def test_el_paquete_casa_con_el_driver(cuda, esperado):
    assert onnxruntime_requirement(cuda) == esperado


def test_un_driver_de_cuda_11_no_tiene_paquete():
    with pytest.raises(EntornoError, match=r"CUDA 11\.8"):
        onnxruntime_requirement((11, 8))
