"""El cuaderno de Colab: que sus celdas compilen y que no se separe del paquete.

No se ejecuta aquí: necesita Colab, una GPU y un partido. Lo que se comprueba es lo que
se rompe sin que nadie lo vea hasta que alguien lo abre: una celda con un error de
sintaxis, un nombre del paquete que ya no existe, o un valor por defecto que deja de ser
el medido.
"""

from __future__ import annotations

import ast
import importlib
import json
import re
from pathlib import Path

from fanalysis import constants

CUADERNO = Path(__file__).resolve().parents[1] / "notebooks" / "highlights_goles.ipynb"


def _celdas_de_codigo() -> list[str]:
    datos = json.loads(CUADERNO.read_text(encoding="utf-8"))
    assert datos["nbformat"] == 4
    return ["".join(c["source"]) for c in datos["cells"] if c["cell_type"] == "code"]


def _sin_magias(codigo: str) -> str:
    """Las líneas `!orden` y `%magia` de IPython no son Python: se quitan para compilar."""
    return "\n".join(
        "" if linea.lstrip().startswith(("!", "%")) else linea for linea in codigo.splitlines()
    )


def _compila(codigo: str) -> str | None:
    """El error de sintaxis de una celda, o `None` si compila."""
    try:
        ast.parse(_sin_magias(codigo))
    except SyntaxError as error:
        return str(error)
    return None


def test_cada_celda_compila():
    errores = {n: e for n, c in enumerate(_celdas_de_codigo(), start=1) if (e := _compila(c))}
    assert not errores, f"celdas que no compilan: {errores}"


def test_lo_que_importa_del_paquete_existe():
    for codigo in _celdas_de_codigo():
        for nodo in ast.walk(ast.parse(_sin_magias(codigo))):
            if isinstance(nodo, ast.ImportFrom) and (nodo.module or "").startswith("fanalysis"):
                modulo = importlib.import_module(nodo.module or "")
                for nombre in nodo.names:
                    assert hasattr(modulo, nombre.name), f"{nodo.module}.{nombre.name}"


def _parametros() -> dict[str, object]:
    valores: dict[str, object] = {}
    for codigo in _celdas_de_codigo():
        for nodo in ast.parse(_sin_magias(codigo)).body:
            if (
                isinstance(nodo, ast.Assign)
                and len(nodo.targets) == 1
                and isinstance(nodo.targets[0], ast.Name)
                and isinstance(nodo.value, ast.Constant)
            ):
                valores[nodo.targets[0].id] = nodo.value.value
    return valores


def test_los_valores_por_defecto_son_los_medidos():
    parametros = _parametros()
    assert parametros["UMBRAL_GOL"] == constants.ACTION_THRESHOLD
    assert parametros["UMBRAL_ACCIONES"] == constants.ACTION_THRESHOLD
    assert parametros["SEGUNDOS_ANTES"] == constants.PRE_ROLL_S
    assert parametros["SEGUNDOS_DESPUES"] == constants.POST_ROLL_S
    assert parametros["SALTO_FRAMES"] == constants.DEFAULT_HOP_FRAMES
    assert parametros["EXIGIR_REMATE"] is True


def test_los_parametros_son_formularios_de_colab():
    texto = "\n".join(_celdas_de_codigo())
    for nombre in ("VIDEO", "MODELO_ONNX", "SALIDA", "UMBRAL_GOL", "GOLES_REALES", "QUITAR"):
        assert re.search(rf"^{nombre} = .*#@param", texto, re.MULTILINE), nombre


def test_ningun_secreto_en_el_cuaderno():
    texto = CUADERNO.read_text(encoding="utf-8")
    assert not re.search(r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}", texto)


def test_el_cuaderno_se_guarda_sin_salidas():
    datos = json.loads(CUADERNO.read_text(encoding="utf-8"))
    for celda in datos["cells"]:
        if celda["cell_type"] == "code":
            assert celda["outputs"] == []
            assert celda["execution_count"] is None
