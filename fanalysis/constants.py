"""Constantes de comportamiento, con sus unidades y de dónde sale cada una.

Ninguna se escribe suelta en una expresión: es la misma regla que en football-ai y en
football-ai-training. Cada una dice si está **medida** (y dónde), si es **decisión** del
propietario (y qué ADR la firma) o si es un **punto de partida** que falta medir.
"""

from __future__ import annotations

from typing import Final

# --------------------------------------------------------------------------- #
# Vocabulario de clases
#
# Las clases de una ficha se canonizan (minúsculas, espacios a `_`) antes de usarlas,
# así que lo de abajo vale igual para el `GOAL` de T-DEED que para el `goal` de N4
# (ADR 0004 de football-ai-training).
# --------------------------------------------------------------------------- #

GOAL_KIND: Final = "goal"
"""La clase del gol, ya canonizada. Sin ella en la ficha no hay highlights que hacer."""

FINISH_KINDS: Final = ("shot", "header")
"""Las clases que cuentan como remate antes de un gol.

`shot` está **medida**: en la línea base de T-DEED sobre `videoGP.MP4`, los 10 GOAL que
detectó llevaban un SHOT delante (PROGRESS de football-ai-training, 2026-09-20). `header`
es **extrapolación**, porque un gol de cabeza no tiene por qué llevar SHOT. Es punto
abierto del ADR 0001."""

BACKGROUND_KIND: Final = "normal_play"
"""La clase de fondo, «no pasa nada». Nunca da una acción: tiene la puntuación alta casi
todo el partido. Es `SPOTTER_BACKGROUND_CLASS` de football-ai."""

# --------------------------------------------------------------------------- #
# Ficha y modelo
# --------------------------------------------------------------------------- #

LOGITS: Final = "logits"
DISPLACEMENT: Final = "displacement"
"""Los significados de salida que entiende este repo, los de la ficha v2 de football-ai.
Las salidas se buscan por significado y no por nombre: el nombre lo elige quien exporta."""

SOFTMAX: Final = "softmax"
SIGMOID: Final = "sigmoid"
ACTIVATIONS: Final = (SOFTMAX, SIGMOID)
"""Las activaciones que puede declarar una ficha. Leer con la equivocada no da error:
da números plausibles y mal calibrados, que es lo peor que le puede pasar a un umbral."""

BATCH_AXIS: Final = "N"
TIME_AXIS: Final = "T"
CHANNEL_AXIS: Final = "C"
HEIGHT_AXIS: Final = "H"
WIDTH_AXIS: Final = "W"
"""Las letras del `layout` de la ficha, p. ej. `NTCHW` en T-DEED."""

RGB: Final = "RGB"
BGR: Final = "BGR"
"""Los órdenes de canal que puede pedir una ficha. ffmpeg entrega RGB."""

CHANNELS: Final = 3
"""Canales de color de un frame."""

HASH_CHUNK_BYTES: Final = 1 << 20
"""Bytes por lectura al calcular el sha256 de un modelo: 1 MiB, sin cargar 50 MB de golpe."""

CUDA_PROVIDER: Final = "CUDAExecutionProvider"
CPU_PROVIDER: Final = "CPUExecutionProvider"
"""Los dos proveedores de onnxruntime que se usan. TensorRT se deja fuera a propósito:
construir su motor tarda minutos y en Colab no hay nada que lo justifique."""

# --------------------------------------------------------------------------- #
# Detección de acciones
# --------------------------------------------------------------------------- #

DEFAULT_HOP_FRAMES: Final = 25
"""Frames entre el inicio de dos ventanas: 2 s a los 12,5 fps de T-DEED.

Es lo que hace la inferencia original de T-DEED (`overlap_len = clip_len // 4 * 3`, 75
de 100 frames de solape), así que cada frame se promedia sobre cuatro ventanas. Subirlo a
50 divide el tiempo entre dos y promedia sobre dos."""

ACTION_THRESHOLD: Final = 0.2
"""Puntuación mínima de un pico de acción, de 0 a 1.

Es el umbral por defecto de T-DEED, y está **medido**: a 0,2, la línea base dio 6 goles
de 6 en `videoGP.MP4`, y el más flojo salió a 0,255 (PROGRESS de football-ai-training,
2026-09-20). Lo que falta medir es el recall, que pide un partido completo anotado."""

NMS_S: Final = 2.0
"""Separación mínima entre dos picos de la misma clase, en segundos.

Es `SPOTTER_NMS_S` de football-ai, **medido** sobre T-DEED: funde las crestas del mismo
gol, que salían separadas 0,88-1,12 s, y conserva los goles distintos."""

MIN_HOP_FRAMES: Final = 1
"""El salto más corto posible: una ventana por frame nuevo."""

# --------------------------------------------------------------------------- #
# Goles
# --------------------------------------------------------------------------- #

FINISH_WINDOW_S: Final = 2.5
"""Segundos antes del gol en los que se busca su remate.

**Medido**: en la línea base, el SHOT llegó entre 0,24 y 2,16 s antes del GOAL en los
10 casos. Los 2,5 s dan margen sobre el máximo sin llegar a la jugada anterior."""

GOAL_DEDUP_S: Final = 20.0
"""Dos goles a menos de esto son el mismo, en segundos.

Entre dos goles reales hay celebración y saque de centro. Es `SPOTTER_COOLDOWN_S` de
football-ai, y queda por debajo de los 32 s que separaron los dos goles reales más
próximos de la línea base (05:28 y 06:00)."""

MATCH_TOLERANCE_S: Final = 2.0
"""Tolerancia para dar por acertado un gol frente a la anotación manual, en segundos.

Es `EVENTS_TOLERANCE_S` de football-ai (EV-03): la misma vara con la que se miden N1, N3
y N4. El error temporal de T-DEED está por debajo (línea base)."""

# --------------------------------------------------------------------------- #
# Antes y después del gol
# --------------------------------------------------------------------------- #

PRE_ROLL_S: Final = 20.0
"""Segundos que se guardan antes del gol. **Decisión** del propietario (ADR 0013 de
football-ai, `PRE_ROLL_S` de `tools/replay/clip_builder.py`): no basta con el gol, hace
falta la jugada que lo trae —el pase y el tiro— o el clip no se entiende."""

POST_ROLL_S: Final = 10.0
"""Segundos que se guardan después del gol: la celebración y la reacción. Mismo origen."""

CLIP_DURATION_TOLERANCE_S: Final = 0.5
"""Diferencia máxima entre lo que dura un clip cortado y lo que se pidió, en segundos.

Al recodificar, el corte cae en el frame; el medio segundo cubre el último paquete de
audio y el redondeo del contenedor. Más que eso es un corte que salió mal."""

X264_PRESET: Final = "veryfast"
"""Preset de libx264 al recodificar los clips. Colab da dos vCPU: así un clip de 30 s en
1080p sale en segundos, y a este CRF la calidad sobra."""

X264_CRF: Final = 20
"""Calidad constante de libx264 (0 sin pérdidas, 51 la peor). A 20 el fútbol sale casi
transparente; el clip se corta de la fuente original, no de los frames reducidos."""

AUDIO_BITRATE: Final = "128k"
"""Bitrate del AAC de los clips. Es el sonido ambiente de un partido, no música."""

SNAPSHOT_WIDTH_PX: Final = 640
"""Ancho del fotograma de cada gol en la revisión, en píxeles: se ve si hay gol sin que
el cuaderno pese."""

SNAPSHOT_QUALITY: Final = 3
"""Calidad JPEG de ffmpeg para ese fotograma (2 la mejor, 31 la peor)."""

PREVIEW_WIDTH_PX: Final = 640
PREVIEW_CRF: Final = 30
"""La copia ligera de un clip para verlo dentro del cuaderno: 640 px de ancho y CRF 30.
Un clip de 30 s ocupa así un par de MB, no los 20-30 del original en 1080p."""

# --------------------------------------------------------------------------- #
# Validación del vídeo
# --------------------------------------------------------------------------- #

ASPECT_TOLERANCE: Final = 0.10
"""Diferencia relativa máxima entre la proporción del vídeo y la de la entrada del modelo
antes de avisar. Por encima, el frame se estira al reducirlo y los jugadores salen
deformados respecto a lo que el modelo vio al entrenar. **Punto de partida.**"""

MIN_DECODED_FRACTION: Final = 0.95
"""Fracción mínima de los frames esperados (duración × `sample_fps`) que ffmpeg tiene que
entregar. Por debajo, el vídeo está dañado o cortado, y lo que falte no se analizaría sin
que nadie se enterase. **Punto de partida.**"""

QUARTER_TURN_DEG: Final = 90
HALF_TURN_DEG: Final = 180
"""Grados. Un vídeo girado 90° (con metadatos de rotación) se ve con el ancho y el alto
cambiados: es lo que pasa con los móviles en vertical."""

SECONDS_PER_MINUTE: Final = 60
SECONDS_PER_HOUR: Final = 3600
"""Para escribir y leer instantes como mm:ss y hh:mm:ss."""

FFMPEG: Final = "ffmpeg"
FFPROBE: Final = "ffprobe"
"""Los binarios externos. En Colab vienen instalados; en local tienen que estar en el PATH."""

STDERR_TAIL_CHARS: Final = 2000
"""Caracteres del final de la salida de error de ffmpeg que se enseñan cuando falla: el
motivo siempre está al final, y lo de arriba es la cabecera del binario."""
