from __future__ import annotations

import numpy as np
import pytest
from ayudas import SesionFalsa, ficha_pequena

from fanalysis.ficha import FichaError
from fanalysis.runner import OnnxClipRunner, select_providers


def _ventana(ficha, semilla: int = 0) -> np.ndarray:
    rng = np.random.default_rng(semilla)
    forma = (ficha.clip_frames, ficha.height, ficha.width, 3)
    return rng.integers(0, 256, size=forma, dtype=np.uint8)


def test_ntchw_en_rgb_y_sin_normalizar_es_el_pixel_tal_cual():
    ficha = ficha_pequena()
    sesion = SesionFalsa(ficha)
    corredor = OnnxClipRunner(ficha, model_path=None, session=sesion)  # type: ignore[arg-type]
    frames = _ventana(ficha)

    corredor.infer(frames)

    tensor = sesion.entradas[0]
    assert tensor.dtype == np.float32
    assert tensor.shape == ficha.input_shape
    # [n, t, c, y, x] del tensor es [t, y, x, c] del frame: T-DEED normaliza por dentro.
    np.testing.assert_array_equal(tensor[0], frames.transpose(0, 3, 1, 2).astype(np.float32))


def test_ncthw_lleva_el_tiempo_detras_del_canal():
    ficha = ficha_pequena(layout="NCTHW")
    sesion = SesionFalsa(ficha)
    corredor = OnnxClipRunner(ficha, model_path=None, session=sesion)  # type: ignore[arg-type]
    frames = _ventana(ficha)

    corredor.infer(frames)

    np.testing.assert_array_equal(sesion.entradas[0][0], frames.transpose(3, 0, 1, 2))


def test_bgr_invierte_el_color_y_la_normalizacion_de_la_ficha_se_aplica():
    ficha = ficha_pequena(color="BGR", scale=255.0, mean=(0.5, 0.25, 0.0), std=(0.5, 0.5, 0.25))
    sesion = SesionFalsa(ficha)
    corredor = OnnxClipRunner(ficha, model_path=None, session=sesion)  # type: ignore[arg-type]
    frames = _ventana(ficha)

    corredor.infer(frames)

    bgr = frames[..., ::-1].transpose(0, 3, 1, 2).astype(np.float32)
    media = np.array([0.5, 0.25, 0.0], np.float32).reshape(1, 3, 1, 1)
    desviacion = np.array([0.5, 0.5, 0.25], np.float32).reshape(1, 3, 1, 1)
    np.testing.assert_allclose(sesion.entradas[0][0], (bgr / 255 - media) / desviacion, rtol=1e-6)


def test_las_salidas_vuelven_por_significado_y_sin_el_batch():
    ficha = ficha_pequena()
    logits = np.arange(ficha.clip_frames * 4, dtype=np.float32).reshape(1, ficha.clip_frames, 4)
    desplazamiento = np.ones((1, ficha.clip_frames), np.float32)
    sesion = SesionFalsa(ficha, salidas=[logits, desplazamiento])
    corredor = OnnxClipRunner(ficha, model_path=None, session=sesion)  # type: ignore[arg-type]

    salidas = corredor.infer(_ventana(ficha))

    assert set(salidas) == {"logits", "displacement"}
    np.testing.assert_array_equal(salidas["logits"], logits[0])
    np.testing.assert_array_equal(salidas["displacement"], desplazamiento[0])
    assert corredor.provider == "CPUExecutionProvider"


def test_una_salida_con_nan_no_pasa():
    ficha = ficha_pequena()
    logits = np.full((1, ficha.clip_frames, 4), np.nan, np.float32)
    sesion = SesionFalsa(ficha, salidas=[logits, np.zeros((1, ficha.clip_frames), np.float32)])
    corredor = OnnxClipRunner(ficha, model_path=None, session=sesion)  # type: ignore[arg-type]
    with pytest.raises(FichaError, match="NaN"):
        corredor.infer(_ventana(ficha))


def test_una_ventana_de_otra_forma_no_entra():
    ficha = ficha_pequena()
    corredor = OnnxClipRunner(ficha, model_path=None, session=SesionFalsa(ficha))  # type: ignore[arg-type]
    with pytest.raises(FichaError, match="espera una ventana"):
        corredor.infer(np.zeros((ficha.clip_frames + 1, ficha.height, ficha.width, 3), np.uint8))
    with pytest.raises(FichaError, match="espera una ventana"):
        corredor.infer(np.zeros((ficha.clip_frames, ficha.height, ficha.width, 3), np.float32))  # type: ignore[arg-type]


def test_la_sesion_se_comprueba_al_abrir():
    ficha = ficha_pequena()
    with pytest.raises(FichaError, match="clases"):
        OnnxClipRunner(ficha, model_path=None, session=SesionFalsa(ficha, clases_logits=9))  # type: ignore[arg-type]


def test_cuda_delante_si_la_hay():
    assert select_providers(["CUDAExecutionProvider", "CPUExecutionProvider"]) == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert select_providers(["TensorrtExecutionProvider", "CPUExecutionProvider"]) == [
        "CPUExecutionProvider"
    ]
