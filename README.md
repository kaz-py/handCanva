# handCanva — Pizarra digital controlada con la mano

Una pizarra blanca cuadriculada que se maneja con la mano delante de la webcam, sin tocar nada.
Hecha con Python, OpenCV y MediaPipe. En pantalla no se ve la cámara: solo la pizarra y el
esqueleto de tu mano; la punta del índice es el puntero.

## Inicio rápido

Requisitos: Python 3.10 o superior, una webcam y buena luz frontal.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (en Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
python main.py
```

- La primera vez descarga el modelo de manos (~7 MB) a `models/`; hace falta internet solo ese momento.
- Al abrir, la guía en pantalla sale 6 s. `H` la vuelve a mostrar.
- La herramienta inicial es el **pincel**.
- Ponte a 50–80 cm de la cámara, con la mano a la altura del pecho y la muñeca a la vista.

### Opciones

| Opción | Efecto |
|---|---|
| `--debug` | Muestra modo estable y candidato, puntero, botón bajo el puntero, dwell, distancia índice-pulgar (`pd`), ratios y ángulos por dedo y una miniatura de la cámara |
| `--fullscreen` | Pantalla completa (también con la tecla `F`) |
| `--source N` | Cámara número N, o ruta a un vídeo |
| `--reach 0.8` | Fracción central de la cámara que cubre toda la pantalla (menor = más sensible) |
| `--no-display --max-frames N` | Ejecuta sin ventana (pruebas y medición de FPS) |

### Teclas

| Tecla | Acción |
|---|---|
| `H` o `?` | Mostrar u ocultar la guía |
| `F` | Pantalla completa |
| `q` / `Esc` | Salir |

## Cómo se usa

### Seleccionar (índice + medio)
Levanta índice y medio, con el resto de dedos cerrados. Mueve la punta del índice sobre un botón y
mantenla ~0,8 s: el anillo de progreso se completa y se confirma. También puedes confirmar al
instante pellizcando pulgar e índice sobre el botón.

### Dibujar con el pincel (solo el índice)
Solo el índice arriba: pinta con el color y el grosor activos. Si pellizcas pulgar e índice con el pincel, el lápiz se levanta (no pinta).

### Dibujar figuras (línea, rectángulo, círculo)
Elige la herramienta y mueve el índice: el puntero se mueve sin pintar. **Toca la punta del índice con
la del pulgar** para fijar el punto de partida; mientras mantengas el pellizco y muevas la mano verás
la vista previa, y al soltar la figura queda dibujada. Sobre un botón, el pellizco sigue siendo un clic. Si pierdes la mano a mitad de figura, se descarta.

### Borrar
Usa la herramienta **Borrador** (fieltro de la bandeja) y dibuja con el índice como con el pincel.
Para vaciar la pizarra usa el botón **Limpiar**. No hay gesto de mano para borrar.

### Dos manos
- Con **un solo dedo (índice) en cada mano**, cada mano dibuja por su cuenta y con **su propio color**
  (la segunda mano empieza con el siguiente color de la paleta). Elegir un color lo asigna a la mano
  que lo eligió. Las insignias «1» y «2» en la bandeja muestran qué marcador usa cada mano.
- Con la herramienta **Línea** activa, las dos manos forman una línea entre las puntas de los índices,
  con su distancia en píxeles. Se confirma tocando índice y pulgar en cualquiera de las dos manos.

### Botones de la pizarra
| Elemento | Qué hace |
|---|---|
| Bandeja de marcadores | Elige el color: rojo, verde, azul, amarillo o negro |
| Borrador de fieltro | Herramienta de borrado |
| Imanes redondos | Pincel, línea, rectángulo, círculo, y grosor − / + |
| Deshacer | Quita el último trazo (si dos manos dibujaron a la vez solapadas, se deshacen juntos) |
| Limpiar | Vacía toda la pizarra |

## Calibrar (recomendado en el stand)

```bash
python tools/calibrar.py
```

En tu posición habitual, lleva el índice a cada objetivo (cuatro esquinas y centro) y mantenlo quieto
~1 s. Se guarda `calibracion.json`, que `main.py` carga al iniciar. Sin calibración usa un rectángulo por defecto.

Las coordenadas son las de la cámara tal cual, **sin espejo**: si mueves la mano a tu derecha, el
puntero va a la izquierda de la pantalla. Si tu webcam ya entrega la imagen espejada, será al contrario.

## Solución de problemas

- **No abre la cámara:** cierra Zoom, Teams u otros programas que la usen, y prueba `--source 1`.
- **No detecta la mano:** más luz frontal, fondo sin otras manos o caras, y 50–80 cm de distancia.
- **Se confunde de modo:** muestra la mano de frente, con los dedos bien separados o bien cerrados. Con `--debug` ves el modo candidato y el estable.
- **Los botones no responden:** con `--debug`, si el modo no es `SELECCIÓN`, extiende bien índice y medio y cierra anular y meñique.
- **No llego a una esquina o a la bandeja:** usa `--reach 0.7` o recalibra con `tools/calibrar.py`.
- **El puntero no cae donde apunto:** recalibra; el esqueleto en pantalla muestra exactamente dónde está tu índice.
- **Va lento:** cierra otras aplicaciones; el seguimiento de manos usa CPU.

## Pruebas

```bash
pytest
```

Los tests usan landmarks sintéticos y una foto de prueba (`tests/data/pointing_up.jpg`, Apache-2.0, de MediaPipe).
`python tools/audit_gestures.py` repite la auditoría de clasificación de gestos.

## Estructura

| Archivo | Función |
|---|---|
| `main.py` | Bucle principal, dwell, ventana y teclas |
| `gestures.py` | Landmarks → modo de cada mano, puntero y esqueleto |
| `canvas.py` | Capa de dibujo, herramientas y deshacer |
| `ui_overlay.py`, `theme.py` | Pizarra, bandeja, imanes, guía y esqueleto de la mano |
| `tools/` | Calibración y auditoría de gestos |
| `PROMPTS.md` | Prompts que originaron el proyecto |
