# ADR 0001 — Highlights de goles con T-DEED, validados por el remate

- **Fecha:** 2026-10-19
- **Estado:** PROPUESTO. Lo pidió el propietario el 2026-10-19. Falta que acepte las
  decisiones marcadas como propias de este ADR.
- **Afecta a:** todo `fanalysis/` y `notebooks/highlights_goles.ipynb`
- **Hereda de football-ai:** el ADR 0013 (clips, antes y después, y que el operador
  decide) y el ADR 0021 (T-DEED fuera del producto)

## Problema

El propietario quiere un cuaderno de Colab, en un repo nuevo, con cuatro cosas:
- detectar las acciones de un vídeo de partido con los modelos ya entrenados;
- montar los highlights de los goles;
- capturar el momento de antes y de después de cada gol;
- validar lo necesario.

Hay cinco preguntas sin contestar:
1. qué modelo;
2. cómo se pasa de su salida a una lista de goles;
3. cómo se separa un gol de verdad de un falso positivo;
4. cuánto antes y cuánto después;
5. cómo se corta.

## Los modelos que hay (2026-10-19)

| Modelo | Qué es | ¿Sirve para el gol? |
|---|---|---|
| `tdeed-snb.onnx` | T-DEED sobre SoccerNet Ball: 12 acciones y fondo en ventanas de 8 s. Exportado y verificado contra torch (9,54e-06) | **Sí, y es el único.** Línea base sobre `videoGP.MP4`: 6 goles de 6 a umbral 0,2, cada GOAL con un SHOT entre 0,24 y 2,16 s antes. El recall no se ha medido |
| `rfdetr-small.onnx` | Detector de jugadores, porteros, árbitros y balón a 512×512 | No ve acciones. Para ver si el balón entra hace falta saber dónde está la portería en la imagen |
| `best.pt` | YOLOv8n de ultralytics (AGPL) con las mismas cuatro clases, sin código de cómo se hizo | No. Además, ultralytics está prohibido |
| N3 (TCN sobre posiciones) | Preentreno con SkillCorner | No: F1 de gol 0,15, y pide las posiciones del N0, que un vídeo suelto no tiene |
| N4 (spotter en streaming) | MNv4 con TSM y GRU, clases shot y goal | Todavía no: sus pesos están sembrados, esperan a ML-52 y ML-53 |

## Decisión

### 1. El modelo es `tdeed-snb`, y solo para analizar

Es el único entrenado que conoce el gol. Su licencia no admite uso comercial (ADR 0021 de
football-ai), así que lo que salga de aquí se analiza y se enseña dentro del equipo, pero
no se publica.

El código no depende de T-DEED: lee su ficha. N4 no podrá entrar solo con una ficha
nueva, porque corre paso a paso con estado. Necesitará su propio `ClipRunner`. Lo
demás, de los picos en adelante, le vale tal cual.

### 2. La inferencia es la original de T-DEED

Se replica su `inference.py`, que es con la que se midió la línea base:

| Qué | Valor | De dónde |
|---|---|---|
| Ventana | 100 frames a 12,5 fps (8 s) | ficha |
| Salto | 25 frames (75 de 100 de solape) | `overlap_len = clip_len // 4 * 3` de T-DEED |
| Por ventana | softmax por frame y después el desplazamiento, agregado por máximo | `process_double_head` de T-DEED; `apply_displacement` de football-ai |
| Entre ventanas | media por frame, contando solo las filas distintas de cero | `support` de T-DEED |

Hay dos diferencias, a propósito:
- **La última ventana se alinea con el final del vídeo.** T-DEED tira los frames que no
  llenan una, y un gol en los últimos segundos se perdería.
- **No se rellena el principio con cinco frames negros.**

Los frames los saca ffmpeg con su filtro `fps`, que remuestrea por tiempo. Así la trampa
de T1 (darle un vídeo a 30 fps a quien espera 25) no puede pasar, y los vídeos de ritmo
variable de los móviles también valen.

### 3. Las acciones: umbral 0,2 y supresión de 2 s

- **0,2** es el umbral por defecto de T-DEED, con el que se midió la línea base.
- **2 s** es `SPOTTER_NMS_S` de football-ai, medido sobre T-DEED: funde las crestas del
  mismo gol (separadas 0,88-1,12 s) y conserva los goles distintos.

### 4. Los goles: duplicados y remate (decisión de este ADR)

- **Duplicados.** Dos picos de GOAL a menos de **20 s** son el mismo gol. Es
  `SPOTTER_COOLDOWN_S` de football-ai, y queda por debajo de los 32 s entre los dos goles
  reales más próximos de la línea base.
- **Remate.** Un gol queda **confirmado** si, en los **2,5 s** anteriores, la puntuación
  por frame de SHOT o de HEADER pasa el umbral de acciones. Se mira la puntuación por
  frame y no los picos, porque un tiro más fuerte fuera de la ventana podría haber
  suprimido al de dentro. Los 2,5 s dan margen sobre el máximo medido (2,16 s).
- **Por defecto, solo los confirmados van a los highlights.** En la línea base, los 10 GOAL
  llevaban SHOT, así que exigirlo no habría costado ningún gol real. Los goles sin remate
  se enseñan igual, con su fotograma, y la revisión los puede meter por su número.
- **Revisión.** Antes de cortar, el cuaderno enseña el fotograma de cada gol y deja quitar
  y poner goles. Es el «el sistema sugiere y una persona decide» del ADR 0013 §1.

### 5. Antes y después del gol

- **20 s antes y 10 s después.** Es la decisión del propietario para los clips (ADR 0013 de
  football-ai, `PRE_ROLL_S` y `POST_ROLL_S`): sin la jugada que lo trae, el gol no se
  entiende. Se pueden cambiar en el formulario.
- **Se recorta al vídeo y se avisa.** Un gol a los 8 s del principio tiene 8 s de antes, no
  20. El clip sale igual y el informe lo marca.
- **Dos clips que se pisan se funden en uno**, para no repetir segundos en los highlights.
- **Se recodifica (decisión de este ADR).** El directo cortaba con `-c copy` porque cada
  segmento empezaba en un keyframe. Un vídeo cualquiera no, así que copiar cortaría en el
  keyframe anterior. libx264 a CRF 20 corta en el frame, y deja todos los clips con el
  mismo formato: los highlights sí se pegan copiando.
- **Cada clip se comprueba:** existe, no está vacío y dura lo pedido, con ±0,5 s.

### 6. Qué se valida antes de gastar GPU

| Es un error (no se sigue) | Es un aviso (se sigue, y se enseña) |
|---|---|
| ffprobe no sabe leer el fichero, o no hay pista de vídeo | la proporción se aleja más de un 10 % de la del modelo |
| el vídeo es vertical (con la rotación ya aplicada) | el vídeo va a menos fps que el modelo |
| el tramo dura menos que una ventana (8 s) | el vídeo tiene menos alto que la entrada del modelo |
| el tramo empieza fuera del vídeo o acaba antes de empezar | no hay audio, o hay rotación |
| la ficha no casa con el `.onnx` (sha256, entrada, salidas o clases) | |
| ffmpeg entrega menos del 95 % de los frames esperados | |
| una salida del modelo trae NaN, o una forma distinta de la de la ficha | |

### 7. Medir contra los goles reales, si se anotan

Si se apuntan a mano los instantes de los goles, el cuaderno da:
- la precisión, el recall y la mediana del error;
- con ±2 s (`EVENTS_TOLERANCE_S` de football-ai, la vara de N1, N3 y N4);
- con emparejado uno a uno por cercanía;
- dos veces: sobre todos los candidatos y sobre los que tienen remate. Así se ve si la
  regla del remate quita goles de verdad.

### 8. Las puntuaciones se guardan

`puntuaciones.npz` va en la carpeta de salida, con el nombre del vídeo, el sha256 del
modelo, el salto y el tramo. Volver a pasar un partido cuesta media hora de GPU, y
cambiar un umbral no debería costarla.

## Alternativas descartadas

- **Correr T-DEED con torch en Colab, como el cuaderno T1.** Habría que clonar código
  GPL e instalar torch. El ONNX ya está verificado contra torch y no trae nada de T-DEED a
  este repo.
- **Su soft-NMS de 12 frames y después la supresión de 2 s.** Para momentos distintos, la
  supresión de 2 s sobre la media da lo mismo con un paso menos.
- **Validar el gol con rfdetr (que el balón entre).** Hace falta saber dónde está la
  portería en la imagen, y para un vídeo cualquiera no hay calibración. Queda para cuando
  el diferido tenga el N0.
- **Un «antes» que se adapte a la jugada**, encadenando pases hacia atrás. Necesita una
  separación medida entre jugadas, y no la hay. Los 20 s fijos son decisión del
  propietario.
- **Cortar con `-c copy`.** El corte sería impreciso, y a veces el clip empezaría con
  frames congelados.

## Consecuencias

- Hay un cuarto repo. Los tres de la migración no cambian, y nada de aquí entra en el
  producto.
- El mismo cuaderno mide el recall de T-DEED en partidos propios, que nunca se ha medido.
  Esa medida le sirve a ML-54 y a EV-08 como línea base.

## Puntos abiertos

| Punto | Recomendación | Lo cierra |
|---|---|---|
| El recall de T-DEED en nuestras cámaras | Pasar el cuaderno por un partido completo con `GOLES_REALES` | el propietario |
| HEADER como remate | Es extrapolación; medirlo con partidos anotados | la primera medida |
| Los umbrales por clase | Con la curva P/R delante, como en el ADR 0013 | la primera medida |
| `ASPECT_TOLERANCE` (10 %) y `MIN_DECODED_FRACTION` (95 %) | Puntos de partida; revisarlos con los primeros vídeos que avisen | el uso |
| El cuaderno en Colab | Aquí no hay Colab: el primer uso dirá si onnxruntime-gpu casa con su CUDA | el propietario |
| La rama | `main`, porque este repo no es de la migración | el propietario |
