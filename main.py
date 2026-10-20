"""Pizarra por gestos: cámara -> mano -> herramientas -> lienzo -> HUD."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

import theme as T
from canvas import Canvas
from gestures import PINCH_ON, DRAW, ERASE, FULL, IDLE, LINE, SELECT, ZONE, Dwell, GestureTracker
from ui_overlay import Overlay, UIState

MODE_NAMES = {DRAW: "DIBUJANDO", SELECT: "SELECCIÓN", ERASE: "BORRANDO", LINE: "DOS MANOS: LÍNEA"}
DWELL_S, GRACE_S, ZONE_PAD, ERASE_R, DET_W, HELP_S = 0.8, 0.15, 0.10, 40, 640, 6.0   # detección sobre frame reducido: landmarks son normalizados
SHAPES = ("line", "rect", "circle")
WIN = "handCanva"
CALIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibracion.json")   # lo escribe tools/calibrar.py


def open_camera(source, w=1280, h=720):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara/fuente '{source}'. Conecta una webcam, "
                           f"cierra otras apps que la usen o prueba --source 1.")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # MJPG: mucho más FPS a 720p
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    return cap


class App:
    """Estado de la sesión; se reconstruye si cambia el tamaño del frame."""

    def __init__(self, w, h, debug=False, zone=FULL):
        self.w, self.h = w, h
        self.s = min(w / T.BASE_W, h / T.BASE_H)   # escala de grosores y tolerancias
        self.overlay, self.canvas = Overlay(w, h), Canvas(w, h)
        self.tracker = GestureTracker(w, h, zone=zone)
        self.dwells = [Dwell(DWELL_S, GRACE_S), Dwell(DWELL_S, GRACE_S)]   # uno por mano
        self.prev_pinch = [False, False]
        self.shape_on = [False, False]                   # pellizco sostenido = figura en curso, por mano
        self.pair_lock = False
        self.shape_pt = [None, None]                     # último punto con pellizco firme (al abrir los dedos la punta salta)
        self.ui = UIState()
        self.debug = debug
        pal = list(T.COLORS)
        self.hcol = [self.ui.color, pal[(pal.index(self.ui.color) + 1) % len(pal)]]   # color propio por mano
        self._ui_color = self.ui.color
        self.cur = {}                                    # id de trazo -> herramienta en curso (0/1 manos, 2 pareja)

    def act(self, bid, hand=0, primary=0):
        kind, _, val = bid.partition(":")
        ui = self.ui
        if kind == "color":
            self.hcol[hand] = val                        # el color elegido es de la mano que lo eligió
            ui.color = self._ui_color = self.hcol[primary]
        elif kind == "tool":
            ui.tool = val
        elif kind == "thick":
            ui.thickness = max(2, min(40, ui.thickness + (2 if val == "+" else -2)))
        elif bid == "action:undo":
            self.canvas.undo()
        elif bid == "action:clear":
            self.canvas.clear()

    def _hover(self, ptr, dwell):
        """Botón bajo el puntero; el que ya estaba en dwell se mantiene en una zona ampliada ~10 % (anti-temblor)."""
        cur = dwell.target
        r = getattr(self.overlay, "_el", {}).get(cur, {}).get("hit") if cur else None
        if r:
            px, py = ZONE_PAD * (r[2] - r[0]), ZONE_PAD * (r[3] - r[1])
            if r[0] - px <= ptr[0] < r[2] + px and r[1] - py <= ptr[1] < r[3] + py:
                return cur
        return self.overlay.hit_test(*ptr)

    def _stroke(self, mode):
        """(herramienta, grosor px) del trazo asociado a un modo, o None."""
        ui = self.ui
        if mode == DRAW:
            return ui.tool, ui.thickness * self.s
        if mode == ERASE:                      # borrador local: radio 40 px escalado; el canvas pinta t*3 de ancho
            return "eraser", 2 * ERASE_R * self.s / 3
        return None

    def _sync_strokes(self, want, drop=()):
        """Reconcilia trazos en curso con los deseados {id: (herramienta, grosor, color, punto[, punto2])};
        los que desaparecen se confirman (end) salvo los de `drop`, que se descartan."""
        cv = self.canvas
        for sid in list(self.cur):
            if sid not in want or want[sid][0] != self.cur[sid]:
                (cv.cancel if sid in drop else cv.end)(sid)
                del self.cur[sid]
        for sid, w in want.items():
            if sid not in self.cur:
                cv.begin(w[0], T.COLORS[w[2]], max(1, w[1]), w[3], sid)
                self.cur[sid] = w[0]
            if sid == 2:
                cv.set_start(w[3], sid)
            cv.move(w[-1], sid)

    def step(self, frame, lm, now):
        """Pinta en `frame` (in-place: pizarra, sin cámara) el estado tras las manos `lm` (lista, mano o None)."""
        ui, cv, tr = self.ui, self.canvas, self.tracker
        tr.update(lm)
        act = [(i, s) for i, s in enumerate(tr.slots) if s.active]
        prim = act[0][0] if act else 0
        if ui.color != self._ui_color:                   # cambio externo del color activo -> mano primaria
            self.hcol[prim] = self._ui_color = ui.color
        ui.hands, ui.pointer = tr.hands, tr.pointer
        ui.pointers = [s.pointer for _, s in act]
        ui.hand_colors = [self.hcol[i] for i, _ in act] if len(act) >= 2 else []
        ui.pinch = bool(act and act[0][1].pinch)
        ui.pinches = [bool(s.pinch) for _, s in act]

        # --- botones: dwell en SELECCIÓN; el pellizco (flanco) es un clic inmediato y tiene prioridad sobre dibujar
        best_hover, best_prog = None, 0.0
        hover, rise = {}, {}
        for i in (0, 1):
            s = tr.slots[i]
            ptr = s.pointer if s.active else None
            hv = None
            if ptr and s.mode != ERASE and (s.mode == SELECT or s.pinch):
                hv = self._hover(ptr, self.dwells[i])
            hover[i] = hv
            rise[i] = bool(s.active and s.pinch and not self.prev_pinch[i])
            self.prev_pinch[i] = bool(s.active and s.pinch)
            prog, fired = self.dwells[i].update(hv if s.mode == SELECT else None, now)
            if rise[i] and hv:
                fired = hv
            if fired:
                self.act(fired, i, prim)
            if hv and (best_hover is None or prog > best_prog):
                best_hover, best_prog = hv, prog
        ui.hover_id, ui.dwell = best_hover, best_prog

        # --- figuras: el pellizco fija el ancla y al soltar se confirma (cada mano con su máquina)
        drop = set()
        for i, s in enumerate(tr.slots):
            if self.shape_on[i] and not (s.active and s.mode == DRAW):
                self.shape_on[i] = False
                drop.add(i)                              # mano perdida / otro modo: no se confirma
            elif self.shape_on[i] and not s.pinch:
                self.shape_on[i] = False                 # soltar el pellizco confirma
            elif rise[i] and not hover[i] and s.mode == DRAW and ui.tool in SHAPES:
                self.shape_on[i] = True
                self.shape_pt[i] = s.pointer

        # --- línea con medida SOLO con la herramienta línea y 2 manos con solo el índice; se confirma con un pellizco
        capable = ui.tool == "line" and len(act) == 2 and all(s.mode == DRAW for _, s in act)
        if not capable or not any(s.pinch for _, s in act):
            self.pair_lock = False
        pair = capable and not self.pair_lock
        confirm = pair and any(rise[i] and not hover[i] for i, _ in act)
        if confirm:
            self.pair_lock, pair = True, False
        if 2 in self.cur and not pair and not confirm:
            drop.add(2)                                  # la guía se descarta si el gesto se pierde sin pellizco
        if pair:
            drop |= {0, 1}
            for i in (0, 1):
                self.shape_on[i] = False
        tips = tuple(sorted(s.pointer for _, s in act)) if (pair or confirm) else None
        ui.measure = tips if pair else None
        ui.mode = "DOS MANOS: LÍNEA" if pair else MODE_NAMES.get(tr.mode, "EN ESPERA")

        want = {}
        if pair:
            want[2] = ("line", ui.thickness * self.s, self.hcol[prim], tips[0], tips[1])
        elif confirm:
            want[2] = ("line", ui.thickness * self.s, self.hcol[prim], tips[0], tips[1])   # último punto, luego se cierra
        else:
            for i, s in act:
                st = self._stroke(s.mode)
                if not st:
                    continue
                if s.mode == DRAW and ui.tool in SHAPES:
                    if not self.shape_on[i]:
                        continue
                    if s.pdist < PINCH_ON:
                        self.shape_pt[i] = s.pointer
                    want[i] = (st[0], st[1], self.hcol[i], self.shape_pt[i])
                    continue
                elif s.mode == DRAW and s.pinch:         # pincel/borrador: el pellizco levanta el lápiz
                    continue
                want[i] = (st[0], st[1], self.hcol[i], s.pointer)
        self._sync_strokes(want, drop)
        if confirm:                                      # fuerza el cierre (commit) de la línea de pareja
            self._sync_strokes({}, ())

        board = getattr(self.overlay, "draw_board", None)
        if board:
            board(frame)
        cv.blend(frame)
        self.overlay.draw(frame, ui)
        if self.debug and act:
            s = act[0][1]
            txt = (f"{s.mode}<-{s.cand}x{s.count} ptr={s.pointer} hover={best_hover} dwell={best_prog:.2f} "
                   f"pinch={int(s.pinch)} pd={s.pdist:.2f} ext={[round(float(x), 2) for x in s.ext]} "
                   f"r={[round(x, 2) for x in s.ratios]} ang={[int(x) for x in s.angles]} fold={s.fold:.2f}")
            cv2.putText(frame, txt, (12, self.h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)


def draw_pip(frame, cam, zone):
    """--debug: miniatura de la cámara (abajo-izquierda) con la zona activa, para calibrar el alcance."""
    h, w = frame.shape[:2]
    tw, th = w // 5, h // 5
    t = cv2.resize(cam, (tw, th), interpolation=cv2.INTER_AREA)
    x0, y0, x1, y1 = zone
    cv2.rectangle(t, (round(x0 * tw), round(y0 * th)), (round(x1 * tw) - 1, round(y1 * th) - 1), T.ACCENT, 1)
    frame[h - th - 40:h - 40, 16:16 + tw] = t


def load_zone(path=None, reach=None):
    """Zona activa (x0, y0, x1, y1) normalizada: --reach > calibracion.json > ZONE por defecto."""
    if reach:
        m = (1 - min(max(reach, 0.3), 1.0)) / 2
        return (m, m, 1 - m, 1 - m)
    try:
        with open(path or CALIB, encoding="utf-8") as f:
            z = tuple(float(v) for v in json.load(f)["zone"])
        if len(z) == 4 and 0 <= z[0] < z[2] - 0.2 and z[2] <= 1 and 0 <= z[1] < z[3] - 0.2 and z[3] <= 1:
            return z
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return ZONE


def run(cap, detector, display=True, max_frames=None, fullscreen=False, debug=False, zone=FULL):
    """Bucle principal. Devuelve el FPS medio."""
    help_until = time.perf_counter() + HELP_S
    app, n, t_start, fps, t_prev = None, 0, time.perf_counter(), 0.0, time.perf_counter()
    if display:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        if fullscreen:
            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    while max_frames is None or n < max_frames:
        ok, cam = cap.read()
        if not ok:
            break
        h, w = cam.shape[:2]   # sin espejo: la cámara solo se usa para detectar, con sus coordenadas tal cual
        if app is None or (w, h) != (app.w, app.h):
            ui = app.ui if app else None
            app = App(w, h, debug, zone)
            frame = np.empty_like(cam)   # salida reutilizable: la pinta draw_board entera (pizarra pura)
            if ui:
                app.ui.color, app.ui.tool, app.ui.thickness = ui.color, ui.tool, ui.thickness
        small = cv2.resize(cam, (DET_W, int(h * DET_W / w)), interpolation=cv2.INTER_AREA) if w > DET_W else cam
        now = time.perf_counter()
        lm = detector.detect(small, int(now * 1000))
        app.step(frame, lm, now)
        app.ui.fps = fps
        if now < help_until and hasattr(app.overlay, "draw_help"):
            app.overlay.draw_help(frame)
        if debug:
            draw_pip(frame, cam, zone)
        n += 1
        t = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(t - t_prev, 1e-6) if fps else 1 / max(t - t_prev, 1e-6)
        t_prev = t
        if display:
            cv2.imshow(WIN, frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")) or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
            if k in (ord("h"), ord("H"), ord("?")):
                help_until = 0.0 if time.perf_counter() < help_until else time.perf_counter() + 3600
            if k in (ord("f"), ord("F")):
                fullscreen = not fullscreen
                cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
    if display:
        cv2.destroyAllWindows()
    return n / max(time.perf_counter() - t_start, 1e-6)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="índice de cámara o ruta de vídeo")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--debug", action="store_true", help="muestra modo, puntero, hover y dwell")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--reach", type=float, help="fracción central de la cámara que cubre toda la pantalla "
                    "(p. ej. 0.8); por defecto calibracion.json o la zona estándar")
    a = ap.parse_args(argv)
    src = int(a.source) if a.source.isdigit() else a.source
    try:
        cap = open_camera(src)
        from gestures import HandDetector
        det = HandDetector()
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    try:
        fps = run(cap, det, not a.no_display, a.max_frames, a.fullscreen, a.debug, load_zone(reach=a.reach))
    finally:
        cap.release()
    print(f"FPS medio: {fps:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
