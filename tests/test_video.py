from __future__ import annotations

import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ayudas import ficha_real

from fanalysis.spotting import detect, find_actions, window_starts
from fanalysis.video import (
    FfmpegFrameReader,
    VideoError,
    VideoInfo,
    expected_frames,
    parse_probe,
    probe,
    read_window,
    validate,
)

TIENE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
RUTA = Path("partido.mp4")


def sondeo(**video: Any) -> dict[str, Any]:
    """Lo que devuelve ffprobe de un vídeo de móvil típico, con lo que se quiera cambiar."""
    pista = {
        "codec_type": "video",
        "codec_name": "h264",
        "width": 1280,
        "height": 720,
        "avg_frame_rate": "30000/1001",
        "r_frame_rate": "30000/1001",
        **video,
    }
    return {
        "streams": [pista, {"codec_type": "audio", "codec_name": "aac"}],
        "format": {"duration": "710.710000"},
    }


def info(**cambios: Any) -> VideoInfo:
    base = {
        "path": RUTA,
        "duration_s": 710.0,
        "width": 1280,
        "height": 720,
        "fps": 29.97,
        "rotation": 0,
        "codec": "h264",
        "has_audio": True,
    }
    return VideoInfo(**{**base, **cambios})


def test_el_sondeo_de_un_video_normal():
    leido = parse_probe(sondeo(), RUTA)
    assert (leido.width, leido.height) == (1280, 720)
    assert leido.fps == pytest.approx(29.97, abs=0.01)
    assert leido.duration_s == pytest.approx(710.71)
    assert leido.has_audio
    assert leido.rotation == 0


def test_la_rotacion_de_un_movil_cambia_ancho_y_alto():
    girado = parse_probe(sondeo(side_data_list=[{"rotation": -90}]), RUTA)
    assert (girado.width, girado.height, girado.rotation) == (720, 1280, -90)
    volteado = parse_probe(sondeo(side_data_list=[{"rotation": -180}]), RUTA)
    assert (volteado.width, volteado.height) == (1280, 720)
    viejo = parse_probe(sondeo(tags={"rotate": "90"}), RUTA)
    assert (viejo.width, viejo.height) == (720, 1280)


def test_el_ritmo_medio_vacio_cae_al_nominal_y_la_caratula_no_es_video():
    leido = parse_probe(sondeo(avg_frame_rate="0/0", r_frame_rate="25/1"), RUTA)
    assert leido.fps == 25.0
    caratula = {
        "codec_type": "video",
        "width": 600,
        "height": 600,
        "disposition": {"attached_pic": 1},
    }
    con_caratula = sondeo()
    con_caratula["streams"].insert(0, caratula)
    assert parse_probe(con_caratula, RUTA).width == 1280


@pytest.mark.parametrize(
    ("datos", "motivo"),
    [
        ({"streams": [{"codec_type": "audio"}], "format": {"duration": "10"}}, "pista de vídeo"),
        ({**sondeo(), "format": {}}, "cuánto dura"),
        (sondeo(avg_frame_rate="0/0", r_frame_rate="0/0"), "frames por segundo"),
        (sondeo(width=0), "tamaño"),
    ],
)
def test_un_sondeo_que_no_sirve_lo_dice(datos, motivo):
    with pytest.raises(VideoError, match=motivo):
        parse_probe(datos, RUTA)


def test_un_video_que_no_existe(tmp_path):
    with pytest.raises(VideoError, match="no existe"):
        probe(tmp_path / "nada.mp4")


def test_un_video_bueno_pasa_sin_avisos():
    tramo = validate(info(), ficha_real())
    assert (tramo.start_s, tramo.end_s, tramo.warnings) == (0.0, 710.0, ())
    assert expected_frames(tramo, 12.5) == 8875


def test_el_tramo_se_recorta_al_video():
    tramo = validate(info(), ficha_real(), start_s=250.0, end_s=900.0)
    assert (tramo.start_s, tramo.end_s) == (250.0, 710.0)
    assert validate(info(), ficha_real(), end_s=0).end_s == 710.0


@pytest.mark.parametrize(
    ("cambios", "tramo", "motivo"),
    [
        ({}, {"start_s": 800.0}, "empieza"),
        ({}, {"start_s": -1.0}, "empieza"),
        ({}, {"start_s": 300.0, "end_s": 200.0}, "antes de empezar"),
        ({}, {"start_s": 300.0, "end_s": 305.0}, "no le cabe ni una"),
        ({"duration_s": 6.0}, {}, "no le cabe ni una"),
        ({"width": 720, "height": 1280}, {}, "vertical"),
    ],
)
def test_lo_que_hace_inutil_el_analisis_es_un_error(cambios, tramo, motivo):
    with pytest.raises(VideoError, match=motivo):
        validate(info(**cambios), ficha_real(), **tramo)


@pytest.mark.parametrize(
    ("cambios", "aviso"),
    [
        ({"width": 1440, "height": 1080}, "proporción"),
        ({"fps": 10.0}, "repetirán frames"),
        ({"width": 640, "height": 360}, "se amplía"),
        ({"has_audio": False}, "mudos"),
        ({"rotation": 180}, "rotación"),
    ],
)
def test_lo_que_lo_empeora_es_un_aviso(cambios, aviso):
    avisos = validate(info(**cambios), ficha_real()).warnings
    assert any(aviso in a for a in avisos), avisos


# ------------------------------------------------------------ con ffmpeg de verdad


def _video(ruta: Path, *, segundos: int, tamano: str = "1280x720", fps: int = 30) -> Path:
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size={tamano}:rate={fps}:duration={segundos}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(ruta),
        ],
        check=True,
    )
    return ruta


@pytest.mark.ffmpeg
@pytest.mark.skipif(not TIENE_FFMPEG, reason="hace falta ffmpeg y ffprobe en el PATH")
def test_los_frames_salen_al_ritmo_y_al_tamano_del_modelo(tmp_path):
    ficha = ficha_real()
    leido = probe(_video(tmp_path / "v.mp4", segundos=10))
    assert (leido.width, leido.height, round(leido.fps)) == (1280, 720, 30)
    assert not leido.has_audio

    tramo = validate(leido, ficha, start_s=1.0, end_s=9.0)
    destino = np.zeros((ficha.height, ficha.width, 3), np.uint8)
    with FfmpegFrameReader(leido, ficha, tramo) as lector:
        frames = 0
        while lector.read_into(destino):
            frames += 1
        lector.check_complete()
        # Al acabar, el destino se queda con el último frame entero, sin pisar.
        ultimo = destino.copy()
        assert not lector.read_into(destino)
        np.testing.assert_array_equal(destino, ultimo)
    # 8 s a 12,5 fps son 100 frames; el filtro `fps` puede redondear uno en el borde.
    assert frames == lector.frames_read
    assert abs(frames - expected_frames(tramo, ficha.sample_fps)) <= 1
    assert expected_frames(tramo, ficha.sample_fps) == 100
    assert destino.std() > 0  # hay imagen, no un frame negro


@pytest.mark.ffmpeg
@pytest.mark.skipif(not TIENE_FFMPEG, reason="hace falta ffmpeg y ffprobe en el PATH")
def test_un_video_cortado_no_pasa_por_bueno(tmp_path):
    ficha = ficha_real()
    leido = probe(_video(tmp_path / "v.mp4", segundos=10))
    # Se le dice que dura más de lo que dura: faltarán frames.
    largo = replace(leido, duration_s=20.0)
    tramo = validate(largo, ficha)
    destino = np.zeros((ficha.height, ficha.width, 3), np.uint8)
    with FfmpegFrameReader(largo, ficha, tramo) as lector:
        while lector.read_into(destino):
            pass
        with pytest.raises(VideoError, match="dañado o cortado"):
            lector.check_complete()


@pytest.mark.ffmpeg
@pytest.mark.skipif(not TIENE_FFMPEG, reason="hace falta ffmpeg y ffprobe en el PATH")
def test_un_fichero_que_no_es_video(tmp_path):
    falso = tmp_path / "falso.mp4"
    falso.write_bytes(b"esto no es un mp4" * 100)
    with pytest.raises(VideoError, match="no pudo leer"):
        probe(falso)


@pytest.mark.ffmpeg
@pytest.mark.skipif(not TIENE_FFMPEG, reason="hace falta ffmpeg y ffprobe en el PATH")
def test_la_cadena_entera_sin_el_modelo(tmp_path):
    """ffmpeg, el lector, el anillo de ventanas y la media, con un runner falso del tamaño
    real de T-DEED. Comprueba la cadena entera salvo el modelo."""
    ficha = ficha_real()
    leido = probe(_video(tmp_path / "v.mp4", segundos=13))
    tramo = validate(leido, ficha, start_s=2.0)

    class Fondo:
        def __init__(self) -> None:
            self.ventanas: list[np.ndarray] = []

        def infer(self, frames: np.ndarray) -> dict[str, np.ndarray]:
            assert frames.shape == (100, 448, 796, 3)
            assert frames.dtype == np.uint8
            self.ventanas.append(frames[:, 0, 0].copy())
            logits = np.zeros((100, 13), np.float32)
            logits[:, 0] = 4.0
            return {"logits": logits, "displacement": np.zeros(100, np.float32)}

    corredor = Fondo()
    with FfmpegFrameReader(leido, ficha, tramo) as lector:
        linea = detect(lector, corredor, ficha, offset_s=tramo.start_s, name="v.mp4")
        lector.check_complete()

    # 11 s a 12,5 fps: unos 137 frames. Ventanas en 0, 25 y la última pegada al final.
    assert abs(linea.n_frames - expected_frames(tramo, ficha.sample_fps)) <= 1
    assert linea.windows == len(window_starts(linea.n_frames, 100, 25)) == 3
    assert linea.support.min() >= 1
    assert linea.time_of(0) == 2.0
    assert linea.source == "v.mp4"
    # El fondo gana en todo el tramo: ni una acción.
    assert find_actions(linea) == []


@pytest.mark.ffmpeg
@pytest.mark.skipif(not TIENE_FFMPEG, reason="hace falta ffmpeg y ffprobe en el PATH")
def test_una_ventana_de_prueba(tmp_path):
    ficha = ficha_real()
    leido = probe(_video(tmp_path / "v.mp4", segundos=10))
    ventana = read_window(leido, ficha, 1.0)
    assert ventana.shape == (100, 448, 796, 3)
    assert ventana.std() > 0
    with pytest.raises(VideoError, match="solo hay"):
        read_window(leido, ficha, 5.0)
