# Prompts mejorados

## 1. Agente UI/UX (ui_overlay.py + theme.py)

Actúa como diseñador UI/UX senior de interfaces interactivas para ferias tecnológicas (lectura a 2–3 m, usuarios jóvenes, interacción sin contacto). Usa las skills `ui-ux-pro-max` y `frontend-design`.

**Entrega:** `theme.py` y `ui_overlay.py` (OpenCV + NumPy puro) que dibujan el HUD de una Pizarra Digital por gestos sobre cada frame 1280×720, siguiendo el contrato de `CLAUDE.md`.

**Diseño:**
- Estética profesional y minimalista (NO cyberpunk/neón): superficies grafito translúcidas, bordes finos, un único acento azul, iconos de línea, mucho espacio.
- Barra superior: 5 círculos de color, selector de herramientas (Pincel, Línea, Rectángulo, Círculo, Borrador) con iconos dibujados, grosor − / +, Undo y Clear. Botones ≥ 72 px, separación clara, estado activo evidente (relleno + borde brillante + texto).
- Cuadro de estado en esquina: `MODO: …` con color/icono por modo + FPS.
- Puntero: retículo + halo que sigue la punta del índice; cambia de forma según modo.
- Dwell: anillo de carga circular (arco 0→360°) sobre el botón en hover, con pulso al confirmar.
- Accesibilidad: contraste ≥ 4.5:1, nunca solo color, texto en español.

**Criterios de aceptación:** `Overlay.draw` < 4 ms por frame; sin asignaciones grandes por frame; incluir `if __name__ == "__main__"` que muestre una demo con puntero movido por el mouse para validar visualmente.

## 2. Agente CV/Integración (gestures.py, canvas.py, main.py)

Actúa como desarrollador senior de Visión por Computadora (Python, OpenCV, MediaPipe Hands). Conecta la UI ya existente (`ui_overlay.py`, contrato en `CLAUDE.md`) con la lógica completa.

**Entrega:** `gestures.py`, `canvas.py`, `main.py`, `requirements.txt`.

**Requisitos:**
1. Webcam en tiempo real, ≥ 30 FPS, coordenadas directas de la cámara, sin espejo (la cámara no se muestra); error claro si no hay cámara.
2. Gestos: SELECCIÓN (índice+medio), DIBUJO (solo índice), BORRAR TODO (puño sostenido 0.8 s), IDLE. Histéresis de 3–4 frames para evitar parpadeo de modo; EMA en el puntero.
3. Dwell 1.0 s sobre botón (+ pellizco como alternativa) → ejecuta acción; `UIState.dwell` alimenta la animación.
4. Herramientas: pincel libre, línea, rectángulo, círculo (vista previa al arrastrar, se confirma al soltar/salir de DIBUJO), borrador, 5 colores, grosor ajustable.
5. Undo (pila de snapshots acotada) y Limpiar.
6. Mezcla lienzo+cámara con máscara, sin lag; sin bucles Python por píxel.
7. Pruebas: tests pequeños para clasificación de gestos (landmarks sintéticos) y undo/clear del canvas.

**Criterios de aceptación:** `python main.py` corre, 30+ FPS medidos, `pytest` pasa.
