from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from fanalysis.clips import (
    ClipError,
    ClipPlan,
    ClipResult,
    build_highlights,
    clip_name,
    concat_command,
    cut_clips,
    cut_command,
    media_duration,
    plan_clips,
    preview_command,
    run_or_fail,
    snapshot_command,
    write_listfile,
)
from fanalysis.goals import Goal

TIENE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def gol(numero: int, t_s: float) -> Goal:
    return Goal(number=numero, frame=0, t_s=t_s, score=0.9, finish=None, finish_checked=True)


def test_veinte_segundos_antes_y_diez_despues():
    (plan,) = plan_clips([gol(1, 264.0)], duration_s=710.0)
    assert (plan.start_s, plan.end_s) == (244.0, 274.0)
    assert plan.goals == (1,)
    assert not plan.pre_truncated
    assert not plan.post_truncated


def test_un_gol_pegado_a_los_bordes_se_recorta_y_lo_dice():
    inicio, final = plan_clips([gol(1, 8.0), gol(2, 705.0)], duration_s=710.0)
    assert (inicio.start_s, inicio.end_s, inicio.pre_truncated) == (0.0, 18.0, True)
    assert (final.start_s, final.end_s, final.post_truncated) == (685.0, 710.0, True)


def test_dos_goles_cuyos_clips_se_pisan_salen_en_uno():
    planes = plan_clips([gol(2, 100.0), gol(1, 90.0), gol(3, 200.0)], duration_s=710.0)
    assert [(p.start_s, p.end_s, p.goals) for p in planes] == [
        (70.0, 110.0, (1, 2)),
        (180.0, 210.0, (3,)),
    ]
    assert planes[0].goal_times == (90.0, 100.0)


def test_los_goles_de_la_linea_base_que_no_se_pisan_van_aparte():
    # 05:28 y 06:00: 328 + 10 = 338 y 360 - 20 = 340. Dos segundos de hueco.
    planes = plan_clips([gol(1, 328.0), gol(2, 360.0)], duration_s=710.0)
    assert len(planes) == 2


@pytest.mark.parametrize(
    ("kwargs", "motivo"),
    [
        ({"pre_s": -1.0}, "negativos"),
        ({"pre_s": 0.0, "post_s": 0.0}, "algo de antes"),
        ({"duration_s": 0.0}, "dura"),
    ],
)
def test_parametros_de_los_clips(kwargs, motivo):
    with pytest.raises(ClipError, match=motivo):
        plan_clips([gol(1, 10.0)], **{"duration_s": 60.0, **kwargs})


def test_un_gol_fuera_del_video_no_se_corta():
    with pytest.raises(ClipError, match=r"\[2\]"):
        plan_clips([gol(1, 10.0), gol(2, 70.0)], duration_s=60.0)


def test_el_nombre_dice_los_goles_y_el_minuto():
    uno = ClipPlan(244.0, 274.0, (1,), (264.4,), False, False)
    dos = ClipPlan(300.0, 340.0, (2, 3), (328.0, 330.0), False, False)
    assert clip_name(uno) == "gol_01_04m24s.mp4"
    assert clip_name(dos) == "goles_02-03_05m28s.mp4"


def test_el_corte_busca_antes_de_abrir_y_recodifica():
    orden = cut_command(
        Path("p.mp4"), Path("c.mp4"), start_s=244.0, duration_s=30.0, has_audio=True
    )
    assert orden.index("-ss") < orden.index("-i")
    assert orden[orden.index("-ss") + 1] == "244.000"
    assert orden[orden.index("-t") + 1] == "30.000"
    assert "libx264" in orden
    assert "0:a:0?" in orden
    sin_audio = cut_command(
        Path("p.mp4"), Path("c.mp4"), start_s=0.0, duration_s=5.0, has_audio=False
    )
    assert "0:a:0?" not in sin_audio
    assert "aac" not in sin_audio


def test_el_fotograma_y_la_vista_previa_son_ligeros():
    foto = snapshot_command(Path("p.mp4"), Path("g.jpg"), t_s=-0.5)
    assert foto[foto.index("-ss") + 1] == "0.000"  # nunca antes del principio
    assert foto[foto.index("-frames:v") + 1] == "1"
    assert "scale=640:-2" in foto
    vista = preview_command(Path("c.mp4"), Path("v.mp4"))
    assert "scale=640:-2" in vista
    assert vista[vista.index("-crf") + 1] == "30"


def test_run_or_fail_dice_que_se_intentaba(monkeypatch):
    monkeypatch.setattr("fanalysis.clips.run_command", lambda _: (1, "No such file"))
    with pytest.raises(ClipError, match=r"no pudo sacar el fotograma .*No such file"):
        run_or_fail(["ffmpeg"], "sacar el fotograma")


def test_la_lista_de_concat_escapa_las_comillas(tmp_path):
    lista = write_listfile([tmp_path / "gol de O'Neill.mp4"], tmp_path / "lista.txt")
    linea = lista.read_text(encoding="utf-8").strip()
    assert linea.startswith("file '")
    assert "O'\\''Neill" in linea
    assert "\\" not in linea.replace("'\\''", "")  # rutas en POSIX
    assert concat_command(lista, tmp_path / "h.mp4")[-1].endswith("h.mp4")


class FfmpegFalso:
    """Apunta las órdenes y escribe el fichero de salida, como haría ffmpeg."""

    def __init__(self, codigo: int = 0) -> None:
        self.ordenes: list[list[str]] = []
        self.codigo = codigo

    def __call__(self, orden: list[str]) -> tuple[int, str]:
        self.ordenes.append(orden)
        if self.codigo == 0:
            Path(orden[-1]).write_bytes(b"clip")
        return self.codigo, "Invalid data found when processing input"


def test_cada_clip_se_corta_y_se_comprueba(tmp_path):
    planes = plan_clips([gol(1, 60.0), gol(2, 200.0)], duration_s=710.0)
    ffmpeg = FfmpegFalso()
    hechos = cut_clips(
        Path("p.mp4"),
        planes,
        tmp_path,
        has_audio=False,
        run=ffmpeg,
        duration_of=lambda _: 30.1,
    )
    assert [h.path.name for h in hechos] == ["gol_01_01m00s.mp4", "gol_02_03m20s.mp4"]
    assert all(h.duration_s == 30.1 for h in hechos)
    assert len(ffmpeg.ordenes) == 2


def test_un_clip_que_no_dura_lo_pedido_es_un_error(tmp_path):
    planes = plan_clips([gol(1, 60.0)], duration_s=710.0)
    with pytest.raises(ClipError, match=r"dura 12\.00 s y se pidieron 30\.00"):
        cut_clips(
            Path("p.mp4"),
            planes,
            tmp_path,
            has_audio=False,
            run=FfmpegFalso(),
            duration_of=lambda _: 12.0,
        )


def test_si_ffmpeg_falla_se_dice_por_que(tmp_path):
    planes = plan_clips([gol(1, 60.0)], duration_s=710.0)
    with pytest.raises(ClipError, match="Invalid data"):
        cut_clips(Path("p.mp4"), planes, tmp_path, has_audio=False, run=FfmpegFalso(codigo=1))


def test_si_ffmpeg_no_deja_fichero_es_un_error(tmp_path):
    planes = plan_clips([gol(1, 60.0)], duration_s=710.0)
    with pytest.raises(ClipError, match="no existe o está vacío"):
        cut_clips(
            Path("p.mp4"),
            planes,
            tmp_path,
            has_audio=False,
            run=lambda _: (0, ""),
            duration_of=lambda _: 30.0,
        )


def test_los_highlights_van_en_orden_de_partido_y_duran_lo_que_suman(tmp_path):
    def resultado(nombre: str, inicio: float) -> ClipResult:
        ruta = tmp_path / nombre
        ruta.write_bytes(b"x")
        return ClipResult(
            ClipPlan(inicio, inicio + 30, (1,), (inicio + 20,), False, False), ruta, 30.0
        )

    ffmpeg = FfmpegFalso()
    destino = build_highlights(
        [resultado("b.mp4", 300.0), resultado("a.mp4", 100.0)],
        tmp_path / "highlights.mp4",
        run=ffmpeg,
        duration_of=lambda _: 60.0,
    )
    assert destino.is_file()
    lista = (tmp_path / "highlights.txt").read_text(encoding="utf-8").splitlines()
    assert [linea.rsplit("/", 1)[-1] for linea in lista] == ["a.mp4'", "b.mp4'"]
    with pytest.raises(ClipError, match="ningún gol"):
        build_highlights([], tmp_path / "h.mp4", run=ffmpeg)


# ------------------------------------------------------------ con ffmpeg de verdad


def _video_de_prueba(ruta: Path, segundos: int = 12) -> Path:
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size=640x360:rate=30:duration={segundos}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={segundos}",
            "-c:v",
            "libx264",
            "-g",
            "120",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(ruta),
        ],
        check=True,
    )
    return ruta


@pytest.mark.ffmpeg
@pytest.mark.skipif(not TIENE_FFMPEG, reason="hace falta ffmpeg y ffprobe en el PATH")
def test_cortes_reales_al_frame_y_highlights(tmp_path):
    fuente = _video_de_prueba(tmp_path / "partido.mp4")
    planes = plan_clips([gol(1, 4.0), gol(2, 10.5)], duration_s=12.0, pre_s=2.0, post_s=1.0)
    assert [(p.start_s, p.end_s) for p in planes] == [(2.0, 5.0), (8.5, 11.5)]

    hechos = cut_clips(fuente, planes, tmp_path / "clips", has_audio=True)
    for hecho in hechos:
        # El keyframe más cercano está a 4 s: copiar no daría 3 s justos, recodificar sí.
        assert hecho.duration_s == pytest.approx(3.0, abs=0.1)

    total = build_highlights(hechos, tmp_path / "clips" / "highlights.mp4")
    assert media_duration(total) == pytest.approx(6.0, abs=0.2)

    foto = tmp_path / "gol_01.jpg"
    run_or_fail(snapshot_command(fuente, foto, t_s=4.0), "sacar el fotograma")
    assert foto.read_bytes()[:2] == b"\xff\xd8"  # cabecera JPEG
    vista = tmp_path / "vista.mp4"
    run_or_fail(preview_command(hechos[0].path, vista), "hacer la vista previa")
    assert media_duration(vista) == pytest.approx(3.0, abs=0.1)
