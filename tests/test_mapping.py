"""Mapeo landmark -> px de pantalla: coordenadas directas (sin espejo), zona activa, clamp, suavizado, esqueleto = puntero."""
import json
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_core import hand  # noqa: E402
import main  # noqa: E402
from gestures import FULL, ZONE, GestureTracker  # noqa: E402

SIZES = [(960, 540), (1280, 720), (1307, 746), (1920, 1080), (1024, 768)]


def at(x, y, up=(1, 0, 0, 0)):
    """Mano sintética trasladada para que la punta del índice quede en (x, y) normalizado."""
    h = hand(up)
    h[:, :2] += np.array([x, y], np.float32) - h[8, :2]
    return h


@pytest.mark.parametrize("w,h", SIZES)
def test_direct_mapping_no_mirror(w, h):
    t = GestureTracker(w, h)                              # zona FULL: (x*W, y*H), sin espejo ni inversión
    _, ptr, _ = t.update(at(0.25, 0.5))
    assert ptr == (round(0.25 * w), round(0.5 * h))
    assert t.hands[0][8] == ptr                           # el esqueleto usa el mismo mapeo


@pytest.mark.parametrize("w,h", SIZES)
def test_zone_edges_hit_screen_corners_and_clamp(w, h):
    x0, y0, x1, y1 = ZONE
    for (x, y), exp in [((x0, y0), (0, 0)), ((x1, y1), (w - 1, h - 1)),
                        ((-0.1, 1.2), (0, h - 1)), ((1.3, -0.2), (w - 1, 0)),    # extrapolados fuera de [0,1]
                        (((x0 + x1) / 2, (y0 + y1) / 2), (round(w / 2), round(h / 2)))]:
        t = GestureTracker(w, h, zone=ZONE)
        assert t.update(at(x, y))[1] == exp, (x, y)


def test_zone_mapping_continuous_and_monotonic():
    t = GestureTracker(1280, 720, zone=ZONE)
    xs = np.linspace(-0.2, 1.2, 701)
    px = t._px(np.c_[xs, np.full_like(xs, 0.5)])[:, 0]
    assert np.all(np.diff(px) >= 0)
    assert np.diff(px).max() <= (xs[1] - xs[0]) / (ZONE[2] - ZONE[0]) * 1280 + 1e-9   # sin saltos


@pytest.mark.parametrize("w,h", [(1280, 720), (1307, 746), (1920, 1080)])
def test_every_button_reachable_and_clickable_with_zone(w, h):
    """El centro de cada botón se alcanza con la punta dentro de la zona activa y el dwell lo confirma."""
    x0, y0, x1, y1 = ZONE
    app = main.App(w, h, zone=ZONE)
    frame = np.zeros((h, w, 3), np.uint8)
    for bid, e in app.overlay._el.items():
        cx, cy = e["center"]
        nx, ny = x0 + (cx + 0.25) / w * (x1 - x0), y0 + (cy + 0.25) / h * (y1 - y0)
        assert x0 <= nx <= x1 and y0 <= ny <= y1
        for i in range(4):                                # soltar la mano: reinicia puntero y dwell
            app.step(frame, None, i / 30)
        hovered = set()
        for i in range(40):
            app.step(frame, at(nx, ny, (1, 1, 0, 0)), i / 30)
            hovered.add(app.ui.hover_id)
        assert bid in hovered, (bid, app.ui.pointer)


def test_ui_hands_filled_in_every_mode():
    app = main.App(1280, 720)
    frame = np.zeros((720, 1280, 3), np.uint8)
    for up in ((1, 0, 0, 0), (1, 1, 0, 0), (1, 1, 1, 1), (0, 0, 0, 0)):     # dibujo, selección, espera, puño
        for i in range(8):
            app.step(frame, [at(0.5, 0.4, up)], i / 30)
        assert len(app.ui.hands) == 1 and len(app.ui.hands[0]) == 21, (up, app.ui.mode)
    app.step(frame, [at(0.3, 0.4), at(0.7, 0.4)], 1.0)
    assert len(app.ui.hands) == 2


def test_adaptive_filter_less_lag_and_still_smooth():
    t = GestureTracker(1280, 720)
    for i in range(30):                                   # 15 px/frame constantes (~450 px/s a 30 FPS)
        p = t.update(at(0.2 + i * 15 / 1280, 0.5))[1]
    lag = (0.2 + 29 * 15 / 1280) * 1280 - p[0]
    assert lag < 0.6 * 15                                 # el EMA fijo 0.5 de antes iba 1 frame (15 px) detrás
    rng = np.random.default_rng(0)
    t = GestureTracker(1280, 720)
    xs = [t.update(at(0.5 + rng.normal(0, 2 / 1280), 0.5))[1][0] for _ in range(200)]
    assert np.std(xs[20:]) < 0.75 * 2                     # temblor de 2 px reducido en reposo


def test_load_zone(tmp_path):
    assert main.load_zone(tmp_path / "no.json") == ZONE
    assert main.load_zone(reach=0.8) == pytest.approx((0.1, 0.1, 0.9, 0.9))
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"zone": [0.2, 0.1, 0.8, 0.9]}))
    assert main.load_zone(p) == (0.2, 0.1, 0.8, 0.9)
    p.write_text(json.dumps({"zone": [0.5, 0.1, 0.55, 0.9]}))                 # demasiado estrecha: se ignora
    assert main.load_zone(p) == ZONE


class _Cap:
    def __init__(self, frame, n):
        self.frame, self.n = frame, n

    def read(self):
        self.n -= 1
        return self.n >= 0, self.frame.copy()


def test_run_does_not_flip_image_and_shows_board_not_camera():
    cam = np.zeros((360, 640, 3), np.uint8)
    cam[:, :320] = 255                                    # mitad izquierda blanca
    seen = []

    class Det:
        def detect(self, small, ts):
            seen.append(small[:, :320].mean() > small[:, 320:].mean())
            return []

    out = {}
    real = main.App.step
    try:
        main.App.step = lambda self, f, lm, now: (out.__setitem__("f", f), real(self, f, lm, now))
        main.run(_Cap(cam, 3), Det(), display=False)
    finally:
        main.App.step = real
    assert all(seen)                                      # el detector ve la imagen tal cual (sin flip)
    assert out["f"] is not cam and not (out["f"][:, :320] == 255).all()   # en pantalla, pizarra (no la cámara)


DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "pointing_up.jpg")


@pytest.mark.skipif(not os.path.exists(DATA), reason="falta la foto de prueba")
def test_real_hand_photo_pointer_on_fingertip():
    """Extremo a extremo con MediaPipe real: mano a la izquierda de la imagen -> izquierda de la pantalla."""
    pytest.importorskip("mediapipe")
    from gestures import MODEL_PATH, HandDetector
    if not os.path.exists(MODEL_PATH):
        pytest.skip("sin modelo de manos")
    img = cv2.imread(DATA)
    k, ox, oy = 1.4, 150, 120
    img = cv2.resize(img, None, fx=k, fy=k)
    cam = np.full((720, 1280, 3), 200, np.uint8)
    cam[oy:oy + img.shape[0], ox:ox + img.shape[1]] = img
    tip = np.array([ox + 172 * k, oy + 68 * k])           # punta del índice medida en la foto
    out = {}
    real = main.App.step
    try:
        main.App.step = lambda self, f, lm, now: (out.__setitem__("app", self), real(self, f, lm, now))
        main.run(_Cap(cam, 8), HandDetector(), display=False)
    finally:
        main.App.step = real
    app = out["app"]
    assert app.ui.mode == "DIBUJANDO" and len(app.ui.hands) == 1
    assert np.hypot(*(np.array(app.ui.pointer) - tip)) < 20
    assert app.ui.hands[0][8] == app.ui.pointer


def test_calibration_recovers_zone_consistent_with_app_mapping():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
    from calibrar import TARGETS, zone_from_points
    zx0, zy0, zx1, zy1 = 0.15, 0.1, 0.8, 0.82
    raw = [(zx0 + sx * (zx1 - zx0), zy0 + sy * (zy1 - zy0)) for sx, sy in TARGETS]   # inversa del mapeo
    z, err = zone_from_points(raw)
    assert z == pytest.approx((zx0, zy0, zx1, zy1)) and err == pytest.approx(0, abs=1e-9)
    for (sx, sy), (x, y) in zip(TARGETS, raw):            # con esa zona, la app pone el puntero sobre cada objetivo
        p = GestureTracker(1307, 746, zone=z).update(at(x, y))[1]
        assert abs(p[0] - sx * 1307) <= 1 and abs(p[1] - sy * 746) <= 1
