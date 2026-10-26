"""Calibración del alcance: toca (con la punta del índice) 4 esquinas + centro de la pantalla.

Calcula la zona activa de la cámara que cubre toda
la pantalla y la guarda en calibracion.json. Después queda en modo verificación: objetivos + puntero
ya calibrado (mismo mapeo que la app) para comprobar la alineación a ojo.

    python tools/calibrar.py [--source 0]
Teclas: R repetir | Q/Esc guardar y salir.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import theme as T  # noqa: E402
from gestures import FULL, GestureTracker  # noqa: E402
from main import CALIB, DET_W, load_zone, open_camera  # noqa: E402
from ui_overlay import Overlay, _put  # noqa: E402

M = 0.08                                          # objetivos a 8 % del borde (las esquinas exactas no se ven bien)
TARGETS = [(M, M), (1 - M, M), (1 - M, 1 - M), (M, 1 - M), (0.5, 0.5)]   # sup-izq, sup-der, inf-der, inf-izq, centro
NAMES = ["arriba-izquierda", "arriba-derecha", "abajo-derecha", "abajo-izquierda", "centro"]
HOLD, STILL = 20, 0.006                           # frames quietos (~0.7 s) y tolerancia normalizada


def zone_from_points(raw):
    """raw: 5 puntas del índice (x, y) normalizadas de la cámara para TARGETS -> (zona, error del centro en fracción).
    El eje X puede llegar invertido desde la cámara; la zona siempre se devuelve ordenada."""
    tl, tr, br, bl, c = (np.asarray(p, float) for p in raw)
    gx = ((tr[0] + br[0]) - (tl[0] + bl[0])) / 2 / (1 - 2 * M)      # x1 - x0
    gy = ((bl[1] + br[1]) - (tl[1] + tr[1])) / 2 / (1 - 2 * M)      # y1 - y0
    if gx >= 0:
        x0 = (tl[0] + bl[0]) / 2 - M * gx
        x1 = x0 + gx
        sx_c = (c[0] - x0) / gx
    else:
        width = -gx
        x1 = (tl[0] + bl[0]) / 2 + M * width
        x0 = x1 - width
        sx_c = (x1 - c[0]) / width
    y0 = (tl[1] + tr[1]) / 2 - M * gy
    z = (max(0.0, x0), max(0.0, y0), min(1.0, x1), min(1.0, y0 + gy))
    err = np.hypot(sx_c - 0.5, (c[1] - z[1]) / (z[3] - z[1]) - 0.5)
    return z, float(err)


def _valid(z):
    return z[2] - z[0] >= 0.2 and z[3] - z[1] >= 0.2


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    a = ap.parse_args(argv)
    try:
        cap = open_camera(int(a.source) if a.source.isdigit() else a.source)
        from gestures import HandDetector
        det = HandDetector()
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    win, ov, zone, raw, hist, msg = "handCanva - calibrar", None, None, [], [], ""
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    while True:
        ok, cam = cap.read()
        if not ok:
            break
        h, w = cam.shape[:2]
        if ov is None or (ov.W, ov.H) != (w, h):
            ov, out = Overlay(w, h), np.empty_like(cam)
            trk = GestureTracker(w, h, hysteresis=1, zone=zone or FULL)
        small = cv2.resize(cam, (DET_W, int(h * DET_W / w)), interpolation=cv2.INTER_AREA) if w > DET_W else cam
        hands = det.detect(small, int(time.perf_counter() * 1000))
        _, ptr, _ = trk.update(hands)
        ov.draw_board(out)
        if trk.hands and hasattr(ov, "draw_hands"):
            ov.draw_hands(out, trk.hands)
        s = min(w / 1280, h / 720)
        r = max(10, round(18 * s))
        k = len(raw)
        for i, (tx, ty) in enumerate(TARGETS):
            if zone is None and i != k:
                continue
            c = (round(tx * w), round(ty * h))
            cv2.circle(out, c, r, T.ACCENT, max(2, round(3 * s)), cv2.LINE_AA)
            cv2.drawMarker(out, c, T.ACCENT, cv2.MARKER_CROSS, 2 * r, max(1, round(2 * s)), cv2.LINE_AA)
        if zone is None and hands:
            hist.append(np.array(hands[0][8][:2]))       # punta en coords normalizadas de la cámara
            hist = hist[-HOLD:]
            if len(hist) == HOLD and np.ptp(np.array(hist), axis=0).max() < STILL:
                raw.append(np.median(hist, axis=0))
                hist = []
                if len(raw) == len(TARGETS):
                    z, err = zone_from_points(raw)
                    if _valid(z):
                        zone, msg = z, f"Zona {tuple(round(v, 3) for v in z)}  error centro {err * w:.0f} px"
                        trk = GestureTracker(w, h, hysteresis=1, zone=zone)
                    else:
                        msg, raw = "Zona demasiado pequeña: repite con movimientos más amplios", []
        elif zone is None:
            hist = []
        if zone is None and k < len(TARGETS):
            prog = len(hist) / HOLD
            tx, ty = TARGETS[k]
            if prog > 0:
                cv2.ellipse(out, (round(tx * w), round(ty * h)), (r + 8, r + 8), -90, 0, 360 * prog, T.INK, 3, cv2.LINE_AA)
            text = f"Lleva la punta del índice al punto {NAMES[k]} y mantenla quieta ({k + 1}/5)"
        else:
            text = "Verifica: el puntero debe caer sobre cada objetivo. R repetir | Q guardar y salir"
        _put(out, text, (round(140 * s), round(h / 2 - 60 * s)), 0.8 * max(s, 0.7), T.INK, max(1, round(2 * s)))
        if msg:
            _put(out, msg, (round(140 * s), round(h / 2 + 70 * s)), 0.7 * max(s, 0.7), T.INK_DIM, max(1, round(s)))
        if ptr:
            cv2.circle(out, ptr, max(4, round(7 * s)), T.INK, -1, cv2.LINE_AA)
        cv2.imshow(win, out)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("r"), ord("R")):
            zone, raw, hist, msg = None, [], [], ""
            trk = GestureTracker(w, h, hysteresis=1, zone=FULL)
        if key in (27, ord("q"), ord("Q")) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
    cap.release()
    cv2.destroyAllWindows()
    if zone:
        with open(CALIB, "w", encoding="utf-8") as f:
            json.dump({"zone": [round(v, 4) for v in zone]}, f, indent=2)
        print(f"Guardado {CALIB}: zona {zone} (main.py la carga: {load_zone() == tuple(round(v, 4) for v in zone)})")
    else:
        print("Calibración incompleta: no se guardó nada.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
