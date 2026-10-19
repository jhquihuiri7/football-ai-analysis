# Dependencias y licencias

Política: toda dependencia nueva se anota aquí en el mismo commit, con su licencia.
`ultralytics` (AGPL) y `torch` no entran. Lo que solo sirve para medir lo dice.

## Del paquete

| Paquete | Licencia | Para qué |
|---|---|---|
| numpy | BSD-3-Clause | puntuaciones, ventanas y frames |
| PyYAML | MIT | leer `models/registry.yaml` |
| onnxruntime / onnxruntime-gpu | MIT | ejecutar el `.onnx`. Se instala una de las dos (extras `cpu` y `gpu`), porque las dos dan el mismo módulo. En Colab, la versión la elige `fanalysis/entorno.py` por la CUDA del driver |
| Ruedas de NVIDIA (CUDA runtime, cuBLAS, cuFFT, cuRAND, NVRTC y cuDNN), por los extras `[cuda,cudnn]` de onnxruntime-gpu | licencia de NVIDIA para sus librerías redistribuibles | solo en la sesión de Colab, para que onnxruntime-gpu no dependa de la CUDA que traiga. No se redistribuyen |
| matplotlib (extra `plots`) | licencia de matplotlib, tipo PSF y compatible con BSD | la gráfica de la línea de tiempo |

## De desarrollo

| Paquete | Licencia | Para qué |
|---|---|---|
| pytest | MIT | tests |
| ruff | MIT | lint y formato |
| mypy | MIT | tipado estricto |
| types-PyYAML | Apache-2.0 | los tipos de PyYAML para mypy |
| hatchling | MIT | construir el paquete (Colab lo instala con pip desde el clon) |

## Herramientas externas

| Herramienta | Licencia | Cómo se usa |
|---|---|---|
| ffmpeg y ffprobe | LGPL-2.1+, o GPL si la build lleva libx264 | se **ejecutan** como programas, sin enlazarse ni redistribuirse. En Colab vienen instalados |
| Google Colab | términos de Google | donde corre el cuaderno |

## Modelos

| Modelo | Licencia | Aquí |
|---|---|---|
| `tdeed-snb.onnx` | código T-DEED GPL-3.0; pesos afinados sobre SoccerNet Ball Action Spotting, datos de investigación | **Solo para analizar y medir.** Lo que salga con él no se publica ni se vende. No va en git: entra por Drive con su ficha |
