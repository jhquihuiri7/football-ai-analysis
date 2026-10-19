"""El `.onnx` de verdad: abre, casa con su ficha y devuelve lo que la ficha promete.

Necesita el modelo, que no va en git. Se busca en `FAI_MODELO_ONNX` o en el repo de
football-ai junto a este. Una ventana tarda unos 25 s en una CPU de portátil.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from ayudas import ficha_real

from fanalysis.runner import OnnxClipRunner

MODELO = Path(
    os.environ.get(
        "FAI_MODELO_ONNX",
        Path(__file__).resolve().parents[2] / "football-ai" / "models" / "onnx" / "tdeed-snb.onnx",
    )
)

pytestmark = [
    pytest.mark.modelo,
    pytest.mark.skipif(not MODELO.is_file(), reason=f"no está el modelo en {MODELO}"),
]


def test_una_ventana_por_el_modelo_real():
    pytest.importorskip("onnxruntime")
    ficha = ficha_real()
    corredor = OnnxClipRunner(ficha, MODELO)
    rng = np.random.default_rng(0)
    ventana = rng.integers(0, 256, size=(100, 448, 796, 3), dtype=np.uint8)

    salidas = corredor.infer(ventana)

    assert salidas["logits"].shape == (100, 13)
    assert salidas["displacement"].shape == (100,)
    assert np.isfinite(salidas["logits"]).all()
