# PROGRESS

Estado de las tareas. Se actualiza al cerrar cada una: qué se hizo, qué quedó fuera y el
siguiente paso. **Lo que se mide va aquí**: es lo único que luego no se recuerda.

Leyenda: ✅ hecha · 🚧 en curso · ⬜ pendiente

| Tarea | Qué | Estado |
|---|---|---|
| AN-01 | El repo, sus normas y el ADR 0001 | ✅ (el ADR, PROPUESTO) |
| AN-02 | La ficha del modelo y el runner de onnxruntime | ✅ |
| AN-03 | Las acciones: la ventana deslizante de T-DEED y sus picos | ✅ |
| AN-04 | El vídeo: sondeo, validaciones y frames con ffmpeg | ✅ |
| AN-05 | Los goles: remate, duplicados y medida contra lo anotado | ✅ |
| AN-06 | Antes y después del gol: clips, highlights e informe | ✅ |
| AN-07 | El cuaderno de Colab: GPU por driver y prueba de si el modelo ve | ✅ en local · ⬜ en Colab |
| AN-08 | El recall de T-DEED en un partido propio completo | ⬜ |

---

## 2026-10-19 · AN-01 a AN-07 — los highlights de goles

Lo pidió el propietario: un cuaderno de Colab, en un repo nuevo, que detecte las acciones
de un vídeo con los modelos ya entrenados y monte los highlights de los goles, con el
momento de antes y de después y con las validaciones necesarias.

### Los modelos, mirados uno a uno

- **`tdeed-snb.onnx`** es el único entrenado que conoce el gol: 12 acciones de balón y el
  fondo, en ventanas de 8 s.
- **`rfdetr-small.onnx`** detecta jugadores, porteros, árbitros y balón.
- **`best.pt`** resultó ser un YOLOv8n de ultralytics con esas mismas cuatro clases
  (100 épocas, 640 px). Se leyó del pickle como texto, sin cargarlo.
- **N3 y N4** todavía no tienen pesos que sirvan para el gol.

El detalle está en el ADR 0001.

### Hecho

- `fanalysis/`, con cada constante en `constants.py` y su origen:
  - `ficha`: lee la ficha de `models/registry.yaml`, copia literal de la de football-ai. Comprueba el sha256 y que la sesión case con la ficha;
  - `runner`: el `.onnx` detrás de `ClipRunner`, con el tensor reservado una vez;
  - `video`: ffprobe, las validaciones (errores y avisos) y los frames de ffmpeg a 12,5 fps;
  - `spotting`: la inferencia original de T-DEED (ventanas con salto de 25, softmax,
    desplazamiento por máximo y media con soporte), con la última ventana alineada al
    final. Además, `Timeline`, que se guarda en `.npz`;
  - `goals`: duplicados a 20 s, remate (SHOT o HEADER) en los 2,5 s anteriores, la revisión
    por números y la medida contra lo anotado, con ±2 s;
  - `clips`: 20 s antes y 10 s después, recortado al vídeo, fundiendo los clips que se
    pisan, recodificado al frame y comprobado; y los highlights;
  - `report` y `plots`: tablas, el informe JSON y la línea de tiempo de gol y tiro.
- `notebooks/highlights_goles.ipynb`, con doce pasos. Se genera sin salidas.
- Docs: README, CLAUDE.md, ADR 0001 y DEPENDENCIES.

### Comprobado contra la referencia

- `apply_displacement` y `peaks` dan **exactamente** lo mismo que los de
  `libs/vision/spotter.py` de football-ai. En 500 ventanas aleatorias de 100×13, con
  desplazamientos de ±6 frames y tres pares de umbral y separación:
  - diferencia máxima en el desplazamiento: **0**;
  - listas de picos idénticas: **1.500 de 1.500**.
- Un test aparte compara el desplazamiento con el doble bucle del post-proceso original de
  T-DEED, reescrito en el test.

### Validado con el modelo real (en CPU, esta máquina: i7-1165G7, 8 GB, sin GPU NVIDIA)

**`video.mp4` de football-ai** (scouting de la Bundesliga, 30 s, 1920×1080 a 25 fps), lo
que pidió el propietario:
- 375 frames de 375 y 12 ventanas, en 256,7 s (**21,4 s por ventana**);
- **23 acciones en 30 s**: 11 PASS, 8 DRIVE, 3 HIGH PASS y 1 HEADER. Ningún GOAL ni SHOT,
  así que en ese clip no hay goles y no salen clips;
- la prueba de si el modelo ve: fondo medio 0,62, 0,44 y 0,52, con la mediana en 0,52,
  así que pasa;
- el cuaderno entero, celda a celda, fuera de Colab: termina. Con dos goles **sintéticos**
  inyectados en las puntuaciones de ese vídeo (05,0 s con su tiro y 27,0 s sin remate), se
  recorren de verdad la revisión, el corte, la fusión y los highlights:
  - el fotograma PNG de cada gol;
  - un clip fundido `goles_01-02_00m05s.mp4` de 0 a 30 s, con los dos recortes marcados;
  - highlights de 30,000 s;
  - la vista previa, de 0,9 MB;
  - el informe;
  - la medida contra lo anotado: con todos los candidatos, precisión 0,5; solo con
    remate, 1,0.

**`videoGP.MP4`, tramo 04:10-06:10**: 1.500 frames de 1.500 y 57 ventanas. Sin otra carga
iba a 44-48 s por ventana; con los tests y la paridad corriendo a la vez llegó a pasar de
150 s. GOAL máximo **0,000** y 2 acciones en 2 minutos.

**La línea base de T1 no se reproduce con este `.onnx`.** En ese tramo, T1 vio tres goles
(04:24 con 0,962). El detalle de lo descartado está en el ADR 0001, «Medido al validar».
Una ventana por caso:

| Entrada | Fondo medio | Lo más alto |
|---|---|---|
| Estadio (`video.mp4`) | 0,44-0,46 | PASS 0,92 · DRIVE 0,92 · HEADER 0,55 |
| `videoGP.MP4` en el gol de 04:24 | 0,76 | PASS 0,18 · GOAL 0,0002 |
| Ídem, recorte ×3 sobre cada portería | 0,66-0,88 | PASS 0,52 · GOAL 0,0002 |
| Frames negros | 0,71 | HIGH PASS 0,28 |
| Ruido | 0,75 | PASS 0,57 |

### Lo que salió al validar, y ya está cambiado

- **La prueba de si el modelo ve** (ADR 0001 §9): tres ventanas antes del análisis largo;
  con la mediana del fondo por encima de 0,65, se para.
- **La GPU de Colab** (§10): `fanalysis/entorno.py` elige `onnxruntime-gpu[cuda,cudnn]`
  por la CUDA del driver: desde la 1.27 para CUDA 13 y de la 1.21 a la 1.26 para CUDA 12.
  Lo comprobado en PyPI el 2026-10-19:
  - de la 1.21 a la 1.26, los extras traen `nvidia-*-cu12`;
  - desde la 1.27, traen las de CUDA 13.

  Si hay GPU y la sesión abre en CPU, el cuaderno se para.
- **Los fotogramas van en PNG.** El JPEG de ffmpeg 9 falló con `salida.mp4` («Non
  full-range YUV is non-standard»).
- **`peaks` suprime con una máscara.** En 90 minutos simulados, con 28.298 frames de PASS
  por encima de 0,2, tarda 0,03 s frente a 4,15 s comparando todos contra todos, con los
  mismos 1.980 picos.
- **El nombre del clip redondea como la tabla:** un gol en 4,96 s se llama `00m05s`, no
  `00m04s`.
- **Sin `nvidia-smi`**, la primera celda ya no revienta: sigue en CPU.

### Qué quedó fuera

- **El cuaderno no se ha ejecutado en Colab ni en una GPU.** El primer uso dirá si la
  elección de onnxruntime por el driver funciona allí.
- **Ningún gol real del modelo se ha cortado.** Con `video.mp4` no hay goles, y con
  `videoGP.MP4` el modelo no ve. Los clips se probaron con goles sintéticos sobre un vídeo
  real, y con los tests.
- **Por qué no se reproduce T1.** Es un punto abierto del ADR 0001.

### Siguiente paso

1. El propietario corre el cuaderno en Colab con una T4 sobre `video.mp4` y apunta aquí
   los segundos por ventana y el proveedor.
2. Después, sobre un partido del soporte grabado del **programa**, con `GOLES_REALES`, y
   sale la primera P/R de goles en metraje propio (AN-08).
