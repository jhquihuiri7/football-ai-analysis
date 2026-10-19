# football-ai-analysis — instrucciones permanentes

Se carga en todas las sesiones de este repo. Es normativo, no informativo.

## 0. Qué es esto y qué no

El repo donde se **analizan partidos grabados** con los modelos que ya están entrenados. Lo
primero es el cuaderno de Colab de los highlights de goles (ADR 0001). La lógica vive en
el paquete `fanalysis/`, con sus tests. El cuaderno solo la llama.

Lo que **no** vive aquí:
- el entrenamiento, que es `football-ai-training`;
- la referencia, el VPS y el diferido, que son `football-ai`;
- el directo de los iPhone, que es `football-ai-capture`.

**Fuente de verdad.** Cómo se ejecuta un modelo lo dice su ficha en `models/registry.yaml`
de football-ai, y aquí se copia literal. La arquitectura del producto la fijan el
`docs/BLUEPRINT.md` de football-ai y sus ADR: el 0013 para clips y repeticiones, y el
0021 para los eventos. Si algo de aquí choca con ellos, para: escribe un ADR en
`docs/DECISIONS/` con el problema, las opciones y tu recomendación, y pregunta.

## 1. Reglas duras

- **Licencias.** T-DEED es GPL-3.0, y SoccerNet es de investigación. Los dos sirven aquí
  para analizar y para medir. Nada de lo que salga con ellos se publica ni se vende. Están
  prohibidos `ultralytics` (AGPL) y `torch`, porque los modelos entran en ONNX. Toda
  dependencia nueva se anota en `docs/DEPENDENCIES.md`, con su licencia, en el mismo commit.
- **Fichas.** Ningún `.onnx` se usa sin su ficha. El sha256 y la entrada y las salidas se
  comprueban antes de ejecutarlo. Los valores se leen del fichero exportado, nunca de
  memoria.
- **Sin números mágicos.** Cada constante de comportamiento va en `fanalysis/constants.py`,
  con sus unidades y su origen: medida (dónde), decisión (qué ADR) o punto de partida.
- **`fanalysis/` no imprime** (ruff T201). Los errores son excepciones con un mensaje que
  dice qué hacer. El cuaderno sí imprime.
- **Tipado:** `mypy --strict` pasa sobre `fanalysis/`.
- **Secretos** solo por variable de entorno o en los Secretos de Colab. Nunca en el
  cuaderno, en un log ni en un commit. Lo comprueba `tests/test_notebook.py`.
- **El sistema sugiere y una persona decide** (ADR 0013 §1 de football-ai). Los highlights
  salen de la revisión del cuaderno, no directamente del umbral.
- **El cuaderno se guarda sin salidas**, y sus valores por defecto son las constantes
  medidas. Lo comprueba el test del cuaderno.

## 2. Protocolo

- Una tarea, un commit, con Conventional Commits: `feat(goles): ... (AN-05)`.
- Ninguna tarea está hecha sin tests. Antes de darla por hecha se ejecutan los comandos de
  la §4 y se pega la salida real. Si algo falla, se dice.
- Lo que no se puede ejecutar en la máquina de turno (Colab, GPU) se implementa igual y
  se dice que no se ejecutó. No se reporta verde lo que no se ha corrido.
- Al cerrar una tarea se actualiza `docs/PROGRESS.md`: qué se hizo, qué quedó fuera y el
  siguiente paso. **Lo que se mide va ahí**: es lo único que luego no se recuerda.
- Rama `main`. La regla de que «main no recibe nada» es de los tres repos de la migración
  a la arquitectura B, y este repo no forma parte del plan.

## 3. Entorno

- Gestor: `uv`. Nunca `pip install` suelto ni conda. En Colab, el cuaderno instala el
  paquete con pip desde el clon, y es la única excepción.
- El cuaderno corre en Colab con una T4. En local, todo salvo la GPU.
- En Windows hay que exportar `UV_LINK_MODE=copy` antes de cualquier `uv`.
- En la máquina Windows del propietario, el Control de aplicaciones bloquea el
  `python.exe` de los venv (os error 4551). Por eso pytest y mypy se lanzan con el
  intérprete base de uv, cargando `.venv/Lib/site-packages` con `site.addsitedir`.

## 4. Comandos

```bash
uv sync
uv run ruff check . && uv run ruff format --check .
uv run mypy                      # --strict sobre fanalysis/
uv run pytest                    # -m "not modelo" para saltar el .onnx real (25 s por ventana en CPU)
```
