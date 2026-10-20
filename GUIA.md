# Guía de uso — handCanva

Ponte a 50–80 cm de la cámara, con la mano a la vista y buena luz frontal.

En pantalla NO se ve la cámara: se ve la pizarra y el **esqueleto de tu mano** (puntos y huesos),
cuya punta del índice es el puntero. Las coordenadas son las de la cámara tal cual, sin espejo:
el puntero se mueve como se mueve tu mano en la imagen de la cámara.

## Postura y alcance
- Mano a 50–80 cm de la cámara, a la altura del pecho, centrada frente a ella y con la muñeca a la vista.
- No hace falta llegar al borde de la imagen: el **rectángulo central** de la cámara (~80 % de ancho,
  del 8 % al 85 % de alto) ya cubre toda la pantalla, así se alcanzan esquinas y la bandeja de
  marcadores sin que la mano salga del encuadre.
- `python main.py --reach 0.7` usa un rectángulo más pequeño (menos recorrido, más sensible);
  `--reach 0.9` uno mayor (más preciso, más recorrido).
- `python main.py --debug` muestra una miniatura de la cámara (abajo-izquierda) con ese rectángulo.

## Calibrar (recomendado en el stand)
`python tools/calibrar.py`: en tu posición habitual, lleva la punta del índice a cada objetivo
(arriba-izquierda, arriba-derecha, abajo-derecha, abajo-izquierda, centro) y mantenla quieta ~1 s.
Luego verifica que el puntero cae sobre cada objetivo (`R` repite) y sal con `Q`: se guarda
`calibracion.json`, que `main.py` carga al iniciar (si no existe usa el rectángulo por defecto).

## Gestos

### 1. SELECCIONAR (índice + medio)
```
   | |
   | |      Índice y medio arriba, el resto cerrado.
  _| |_     Mueve la punta del índice sobre un botón y mantenla ~0.8 s:
 |     |    el anillo se completa y se confirma.
  \___/     Alternativa: pellizca pulgar con índice sobre el botón.
```

### 2. DIBUJAR (solo el índice)
```
    |
    |       Solo el índice arriba: pinta con la herramienta,
  __|__     color y grosor activos.
 |     |
  \___/
```

### 3. BORRAR (mano completamente cerrada, pulgar incluido)
```
   ___
  |   |     Puño cerrado: borra solo bajo el puño (borrador local).
  |___|     Con el índice solo o con dos dedos NO borra.
```

### 4. DOS MANOS (un índice en cada mano)
```
    |           |
    |           |     Cada mano dibuja por su cuenta, con color propio
   /_\         /_\    (la mano 2 empieza con el color siguiente de la paleta).
```
Elegir un color (dwell o pellizco) lo asigna a la mano que lo eligió; la otra sigue dibujando.
Con la herramienta **Línea** activa, las dos manos forman una línea con su longitud (en px);
al dejar de cumplir el gesto, la línea queda dibujada.

### 5. LIMPIAR y DESHACER
Botones `Limpiar` y `Deshacer`: se activan con el gesto de SELECCIONAR (dwell o pellizco).
Limpiar todo no se hace con gestos de mano, solo con ese botón.

## Teclas
| Tecla | Acción |
|---|---|
| `q` / `Esc` | Salir |
| `F` | Pantalla completa |
| `H` o `?` | Mostrar/ocultar la ayuda (sale sola 6 s al iniciar) |

`python main.py --debug` muestra modo, puntero, botón bajo el puntero, progreso del dwell y la miniatura de cámara.

## Solución de problemas
- **No detecta la mano**: más luz frontal, fondo sin otras manos/caras cercanas, distancia 50–80 cm.
- **No abre la cámara**: cierra Zoom/Teams/otros programas que la usen; prueba `python main.py --source 1`.
- **Se confunde el modo**: muestra la mano de frente y con los dedos bien separados o bien cerrados.
- **Los botones no responden**: usa `--debug`; si el modo no es `SELECCIÓN`, extiende bien índice y medio y cierra anular y meñique.
- **Va lento**: cierra otras apps; el seguimiento de manos usa CPU.
- **No llego a una esquina o a la bandeja**: acércate al centro de la cámara, usa `--reach 0.7` o recalibra con `tools/calibrar.py`.
- **El puntero no cae donde apunto**: recalibra; el esqueleto en pantalla muestra exactamente dónde está tu índice.
