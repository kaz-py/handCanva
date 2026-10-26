"""Paleta, medidas y textos del HUD de pizarra blanca. Colores en BGR (OpenCV).

Idea: la UI son los objetos de una pizarra física (bandeja de marcadores, borrador de
fieltro, imanes, nota adhesiva). Tonos planos; un solo acento (azul marcador).
"""

# Medidas a 1280x720 (rejilla de 8 px); ui_overlay las escala con min(w/1280, h/720).
BASE_W, BASE_H = 1280, 720
TOUCH = 72            # zona táctil mínima de cualquier objeto
FRAME_W = 16          # marco de aluminio
MARGIN = 24           # separación objeto-marco
PITCH = 80            # paso de imanes (disco 64 + aire)
MARKER_PITCH = 88     # paso de marcadores en la bandeja
LINE = 3              # grosor de rotulador de los iconos
FONT_SCALE = 0.85     # Hershey simplex (~18 px de mayúscula a 1280)
SCRIPT_SCALE = 1.0    # Hershey script para etiquetas de imán

# Pizarra y aluminio
PAPER = (238, 244, 246)       # #F6F4EE
GRID = (222, 229, 232)         # cuadrícula fina de cuaderno
GRID_MAJOR = (204, 214, 221)   # línea mayor cada 5 celdas
SMUDGE = (222, 228, 230)      # residuo de borrado, apenas visible
FRAME_OUTER = (201, 196, 191) # #BFC4C9
FRAME_INNER = (228, 225, 222) # #DEE1E4
TRAY_TOP = (228, 225, 222)
TRAY_LIP = (201, 196, 191)

# Tinta
INK = (39, 33, 29)            # #1D2127
INK_DIM = (110, 100, 91)      # #5B646E  (>= 4.5:1 sobre PAPER)
PAPER_WHITE = (252, 252, 250)

# Objetos
MAGNET_FACE = (250, 250, 248)
MAGNET_RIM = (172, 164, 158)
MAGNET_EDGE = (205, 200, 195)  # canto del imán (escalón de tono, no sombra)
MARKER_BODY = (214, 210, 205)
FELT = (80, 66, 59)
ERASER_BODY = (217, 210, 203)
NOTE = (168, 232, 247)        # #F7E8A8
NOTE_FOLD = (140, 212, 230)
TIP_BG = INK

ACCENT = (235, 100, 31)       # #1F64EB: hover y dwell

COLORS = {                    # id -> BGR; tintas de marcador
    "red": (49, 48, 214),
    "green": (90, 158, 30),
    "blue": (200, 140, 0),
    "yellow": (0, 180, 240),
    "black": (46, 39, 35),
}
COLOR_LABELS = {"red": "Rojo", "green": "Verde", "blue": "Azul", "yellow": "Amarillo", "black": "Negro"}
TOOL_LABELS = {"brush": "Pincel", "line": "Línea", "rect": "Rectángulo", "circle": "Círculo", "eraser": "Borrador"}
ACTION_LABELS = {"thick:-": "Menos grosor", "thick:+": "Más grosor",
                 "action:undo": "Deshacer", "action:clear": "Limpiar"}

# modo -> icono (el modo se distingue por icono y texto, no por color)
MODES = {"DIBUJANDO": "brush", "SELECCIÓN": "cursor", "EN ESPERA": "pause",
         "DOS MANOS: LÍNEA": "line"}
DEFAULT_MODE = "EN ESPERA"

DWELL_PULSE_S = 0.45          # duración del pulso al confirmar
