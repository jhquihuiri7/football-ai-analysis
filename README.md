# football-ai-analysis

Análisis de partidos grabados con los modelos ya entrenados de football-ai. Lo primero es
un cuaderno de Colab que **detecta las acciones de un vídeo y monta los highlights de los
goles**, con la jugada que trae cada gol y la celebración.

Aquí no se entrena ni se emite. Los modelos los entrena `football-ai-training` y su ficha
la declara `football-ai` en `models/registry.yaml`. Este repo los carga y los aplica a
vídeos ya grabados.

## El cuaderno

[`notebooks/highlights_goles.ipynb`](notebooks/highlights_goles.ipynb):
[abrir en Colab](https://colab.research.google.com/github/jhquihuiri7/football-ai-analysis/blob/main/notebooks/highlights_goles.ipynb).

**Qué necesita**
- Un entorno con GPU: *Entorno de ejecución → Cambiar tipo de entorno → T4 GPU*. En CPU
  también funciona, pero cada ventana de 8 s tarda unos 21-25 s en una CPU de portátil
  (medido): un partido se va a horas.
- El modelo `tdeed-snb.onnx` (50 MB) en tu Google Drive. El cuaderno comprueba su sha256
  contra la ficha antes de usarlo.
- El vídeo del partido, apaisado y con un plano que siga el juego (ver abajo), en Drive o
  subido a la sesión.

**La GPU de Colab.**
- **Qué onnxruntime se instala.** El cuaderno lee en `nvidia-smi` la CUDA que admite el
  driver e instala el `onnxruntime-gpu` que le toca: desde la 1.27 para CUDA 13, y de la
  1.21 a la 1.26 para CUDA 12. Lo instala con sus propias librerías de NVIDIA, así que no
  depende de la CUDA que traiga Colab.
- **Si hay GPU y aun así abre en CPU, se para** y dice por qué. Seguir costaría horas.

**Qué vídeos sirven.** T-DEED se entrenó con retransmisiones de televisión. Medido el
2026-10-19:
- **Con una cámara de estadio** (un feed de scouting de la Bundesliga) ve el juego: 23
  acciones en 30 s, con pases y conducciones a 0,8-0,9.
- **Con una panorámica amateur desde la grada** (`videoGP.MP4`), en la que los jugadores
  miden unos pocos píxeles, no ve nada: responde como a una imagen negra.

Por eso, antes del análisis largo, el cuaderno pasa tres ventanas de prueba y se para si
el modelo no está viendo el partido. Con las grabaciones del soporte, usa la del
**programa** (la cámara virtual que sale a YouTube), no la panorámica en bruto.

**Qué sale**, en la carpeta de salida (en Drive, si la ruta empieza por `/content/drive`):

| Fichero | Qué es |
|---|---|
| `gol_01_04m24s.mp4`… | un clip por gol: 20 s antes y 10 s después |
| `highlights.mp4` | los clips revisados, pegados en orden de partido |
| `acciones.csv` | todas las acciones detectadas, con instante y confianza |
| `informe.json` | el vídeo, el modelo, los parámetros, cada gol con lo que se comprobó, los clips y los avisos |
| `puntuaciones.npz` | las puntuaciones por frame. Si existe, el cuaderno no vuelve a pasar el modelo |

## Cómo decide qué es un gol

El detalle y el porqué de cada número están en el
[ADR 0001](docs/DECISIONS/0001-highlights-de-goles-con-tdeed.md).

1. **Acciones.** T-DEED mira ventanas de 8 s (100 frames a 12,5 fps) que empiezan cada
   2 s. Es exactamente su inferencia original, con la que se midió la línea base. Cada
   frame se queda con la media de las ventanas que lo vieron. Una acción es un pico por
   encima de 0,2, separado al menos 2 s de otro de su misma clase.
2. **Candidatos a gol.** Son los picos de la clase GOAL. Dos picos a menos de 20 s son el
   mismo gol, porque entre dos goles hay celebración y saque de centro.
3. **Validación por remate.** Un gol confirmado lleva un tiro (o un cabezazo) en los 2,5 s
   anteriores. En la línea base, los 10 GOAL de T-DEED llevaban un SHOT entre 0,24 y
   2,16 s antes. Los goles sin remate se enseñan, pero no entran en los highlights salvo
   que los metas en la revisión.
4. **Revisión.** Antes de montar los highlights, el cuaderno enseña el fotograma de cada
   gol, para quitar o añadir goles por su número. El sistema sugiere y una persona decide
   (ADR 0013 de football-ai).
5. **Validación contra los goles reales (opcional).** Si apuntas a mano cuándo hubo gol,
   el cuaderno calcula la precisión y el recall con ±2 s, la misma tolerancia con la que
   se mide el resto de eventos del proyecto.

## Antes y después del gol

Cada clip empieza **20 s antes** del gol y acaba **10 s después**. Es la decisión del
propietario para los clips del directo: sin el pase y el tiro, el gol no se entiende.
Los dos valores se cambian en el formulario del cuaderno.

- **Un gol pegado al principio o al final del vídeo** tiene menos de 20 s antes o de 10 s
  después. El clip se recorta a lo que hay, y el informe lo marca (`antes_recortado`,
  `despues_recortado`).
- **Dos goles cuyos clips se pisan** salen en un único clip, para no repetir segundos en
  los highlights.
- **El corte es exacto al frame**: se recodifica con libx264 y se comprueba que el clip
  dure lo que se pidió.

## Modelos

| Modelo | Qué hace | Aquí |
|---|---|---|
| `tdeed-snb` | Acciones de balón en ventanas de 8 s: 12 clases, entre ellas GOAL y SHOT | **El que se usa.** Es el único entrenado que detecta goles |
| `rfdetr-small` | Jugadores, porteros, árbitros y balón en un frame | No detecta acciones. Servirá para seguir el balón |
| N3 y N4 | Los eventos propios del producto | Todavía sin pesos que sirvan para el gol |

El pipeline no depende de T-DEED: lee la ficha. Cuando N4 tenga pesos, se cambia la ficha
y el modelo, no el código.

## ⚠️ Licencia del modelo

**T-DEED es GPL-3.0 y sus pesos se afinaron sobre SoccerNet, que es solo para
investigación.** Su ficha dice «SOLO PARA MEDIR». Los highlights que salgan de aquí sirven
para analizar y para enseñar internamente. No se publican ni se venden mientras el modelo
sea T-DEED. El camino comercial es N4, entrenado con datos propios
(ADR 0021 de football-ai).

## Desarrollo local

```bash
export UV_LINK_MODE=copy      # en Windows, antes de cualquier uv
uv sync
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest
```

Algunos tests necesitan cosas de fuera y se saltan solos si faltan:
- los que llevan la marca `ffmpeg` necesitan `ffmpeg` y `ffprobe` en el PATH;
- los que llevan la marca `modelo` necesitan el `.onnx` real, en `FAI_MODELO_ONNX` o en
  `../football-ai/models/onnx/tdeed-snb.onnx`.

Ver [CLAUDE.md](CLAUDE.md) para las normas del repo y [docs/PROGRESS.md](docs/PROGRESS.md)
para el estado y lo medido.
