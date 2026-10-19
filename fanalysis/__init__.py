"""Análisis de partidos con los modelos ya entrenados de football-ai.

Lo primero, los highlights de los goles (ADR 0001):

    ficha   → la ficha del modelo y su comprobación contra el .onnx
    runner  → el .onnx en onnxruntime, detrás de una interfaz
    video   → qué es el vídeo, si sirve, y sus frames al ritmo del modelo
    spotting→ la ventana deslizante de T-DEED y los picos de cada acción
    goals   → los goles, con su remate y sin duplicados; y su medida contra lo anotado
    clips   → antes y después de cada gol, recortado, y los highlights
    report  → tablas e informe
    plots   → la línea de tiempo de gol y tiro
"""

__version__ = "0.1.0"
