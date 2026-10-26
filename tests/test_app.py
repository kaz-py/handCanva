import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_core import hand  # noqa: E402
import main  # noqa: E402


class FakeCap:
    def __init__(self, n):
        self.n = n

    def read(self):
        self.n -= 1
        return self.n >= 0, np.zeros((360, 640, 3), np.uint8)


class ScriptDetector:
    def __init__(self, script):
        self.script, self.i = script, 0

    def detect(self, frame, ts):
        lm = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return lm


def test_no_camera_error(monkeypatch):
    monkeypatch.setattr(main.cv2, "VideoCapture", lambda *a: type("C", (), {"isOpened": lambda s: False})())
    with pytest.raises(RuntimeError, match="No se pudo abrir"):
        main.open_camera(0)


def test_loop_draws_then_fist_does_nothing(monkeypatch):
    seen = {}
    real = main.App.step

    def spy(self, frame, lm, now):
        seen["app"] = self
        real(self, frame, lm, now)
    monkeypatch.setattr(main.App, "step", spy)

    draw = []
    for i in range(10):
        h = hand((1, 0, 0, 0)); h[8, 0] = 0.3 + 0.03 * i; draw.append(h)
    main.run(FakeCap(10), ScriptDetector(draw), display=False)
    app = seen["app"]
    assert app.canvas.mask.any() and app.ui.mode == "DIBUJANDO"
    app.canvas.end()

    fist = hand((0, 0, 0, 0))
    now = 100.0
    n0 = int(app.canvas.mask.sum())
    for _ in range(12):                                # histéresis
        app.step(np.zeros((360, 640, 3), np.uint8), fist, now)
    assert app.ui.mode == "EN ESPERA" and int(app.canvas.mask.sum()) == n0   # el puño no borra ni pinta


def two_hands(app, f, frames, tool, left=(1, 0, 0, 0), right=(1, 0, 0, 0), t0=0.0):
    from test_core import shifted
    for i in range(frames):
        d = 0.004 * i
        app.step(f, [shifted(-0.3 + d, left), shifted(0.3 - d, right)], t0 + i / 30)


def test_line_only_with_line_tool():
    f = np.zeros((360, 640, 3), np.uint8)
    app = main.App(640, 360)
    app.ui.tool = "line"
    two_hands(app, f, 14, "line")
    assert app.ui.mode == "DOS MANOS: LÍNEA" and app.ui.measure and not app.canvas.mask.any()
    (x0, _), (x1, _) = app.ui.measure
    assert x0 < x1
    for _ in range(8):
        app.step(f, None, 1.0)                         # gesto perdido SIN pellizco: la guía se descarta
    assert not app.canvas.mask.any() and app.ui.measure is None


def test_two_hand_line_confirmed_by_pinch_on_either_hand():
    from test_core import pinch_hand
    f = np.zeros((360, 640, 3), np.uint8)
    app = main.App(640, 360)
    app.ui.tool = "line"
    for i in range(14):
        app.step(f, [pinch_hand(0.25, 0.5, False), pinch_hand(0.75, 0.5, False)], i / 30)
    assert app.ui.measure and not app.canvas.mask.any()
    for i in range(5):                                 # pellizca la segunda mano
        app.step(f, [pinch_hand(0.25, 0.5, False), pinch_hand(0.75, 0.5, True)], 1 + i / 30)
    assert app.canvas.mask.any() and app.ui.measure is None


def _drive(app, f, frames, hand_fn, t0=0.0):
    for i in range(frames):
        app.step(f, hand_fn(i), t0 + i / 30)


def test_rect_from_anchor_on_pinch_release():
    from test_core import pinch_hand
    f = np.zeros((360, 640, 3), np.uint8)
    app = main.App(640, 360)
    app.ui.tool = "rect"
    _drive(app, f, 10, lambda i: pinch_hand(0.30, 0.45, False))
    assert app.ui.mode == "DIBUJANDO" and not app.canvas.mask.any()     # el índice solo mueve el puntero
    _drive(app, f, 4, lambda i: pinch_hand(0.30, 0.45, True))           # pellizco: ancla
    st = app.canvas._strokes[0]
    anchor = st.start
    assert anchor == app.tracker.slots[0].pointer or abs(anchor[0] - app.tracker.slots[0].pointer[0]) < 6
    _drive(app, f, 14, lambda i: pinch_hand(0.30 + 0.025 * i, 0.45 + 0.01 * i, True), 1.0)
    end = app.canvas._strokes[0].last
    assert not app.canvas.mask.any()                                    # solo vista previa mientras se mantiene
    _drive(app, f, 4, lambda i: pinch_hand(0.30 + 0.025 * 13, 0.45 + 0.01 * 13, False), 2.0)
    m = app.canvas.mask
    assert m[anchor[1], anchor[0]] > 0 and m[end[1], end[0]] > 0 and m[anchor[1], end[0]] > 0 and m[end[1], anchor[0]] > 0
    assert end[0] > anchor[0] + 50 and end[1] > anchor[1]


def test_pinch_on_button_clicks_and_does_not_draw():
    from test_core import pinch_hand
    f = np.zeros((360, 640, 3), np.uint8)
    app = main.App(640, 360)
    app.ui.tool = "rect"
    x0, y0, x1, y1 = app.overlay._el["color:green"]["hit"]
    cx, cy = (x0 + x1) / 2 / 640, (y0 + y1) / 2 / 360
    _drive(app, f, 10, lambda i: pinch_hand(cx, cy - 0.12, False))
    _drive(app, f, 6, lambda i: pinch_hand(cx, cy - 0.12, True), 1.0)
    # la punta del índice queda por encima de la muñeca: se coloca el puntero sobre el botón y se pellizca
    hover = app.tracker.slots[0].pointer
    assert app.ui.mode == "DIBUJANDO"
    app2 = main.App(640, 360)
    app2.ui.tool = "rect"
    ptr_dx = cx - app.tracker.slots[0].pointer[0] / 640
    ptr_dy = cy - app.tracker.slots[0].pointer[1] / 360
    _drive(app2, f, 10, lambda i: pinch_hand(cx - 0.0 + ptr_dx + 0.0, cy - 0.12 + ptr_dy, False))
    _drive(app2, f, 6, lambda i: pinch_hand(cx + ptr_dx, cy - 0.12 + ptr_dy, True), 1.0)
    assert app2.hcol[0] == "green" and not app2.canvas.mask.any() and not any(app2.shape_on) and app2.cur == {}


def test_brush_pinch_does_not_draw_and_pinch_keeps_draw_mode():
    from test_core import pinch_hand
    f = np.zeros((360, 640, 3), np.uint8)
    app = main.App(640, 360)
    app.ui.tool = "brush"
    _drive(app, f, 12, lambda i: pinch_hand(0.5, 0.5, True))
    assert app.ui.mode == "DIBUJANDO" and not app.canvas.mask.any()
    app = main.App(640, 360)
    modes = set()
    for i in range(80):                                # alterna índice / pellizco sin salir de DIBUJO
        app.step(f, pinch_hand(0.5, 0.5, (i // 5) % 2 == 1), i / 30)
        if i > 8:
            modes.add(app.tracker.slots[0].mode)
    assert modes == {"DRAW"}


def test_brush_two_hands_are_independent_strokes_with_own_colors():
    f = np.zeros((360, 640, 3), np.uint8)
    app = main.App(640, 360)
    app.ui.tool = "brush"
    two_hands(app, f, 20, "brush")
    assert app.ui.measure is None and app.ui.mode == "DIBUJANDO"
    assert app.cur == {0: "brush", 1: "brush"}
    assert app.ui.hand_colors == ["red", "green"]
    cols = {tuple(c) for c in app.canvas.layer[app.canvas.mask > 0]}
    assert len(cols) >= 2                              # dos tintas distintas
    left = app.canvas.mask[:, :320].any() and app.canvas.mask[:, 320:].any()
    assert left                                        # un trazo en cada mitad


def test_color_assigned_to_hand_that_chose_it_and_other_keeps_drawing():
    from test_core import shifted
    f = np.zeros((360, 640, 3), np.uint8)
    app = main.App(640, 360)
    # mano 1 selecciona 'color:blue' mientras la mano 0 sigue dibujando
    x0, y0, x1, y1 = app.overlay._el["color:blue"]["hit"]
    cx, cy = (x0 + x1) / 2 / 640, (y0 + y1) / 2 / 360
    for i in range(60):
        h0 = shifted(0.0 + 0.004 * i)
        h1 = hand((1, 1, 0, 0))
        h1[:, 0] += cx - h1[8, 0]; h1[:, 1] += cy - h1[8, 1]
        app.step(f, [h0, h1], i / 30)
    assert app.hcol == ["red", "blue"]
    assert app.ui.color == "red"                       # la mano 0 conserva su color
    assert app.cur.get(0) == "brush"                   # su trazo no se interrumpió
    assert app.canvas.mask.any()


def test_undo_removes_last_stroke():
    f = np.zeros((360, 640, 3), np.uint8)
    app = main.App(640, 360)
    two_hands(app, f, 14, "brush")
    for _ in range(8):
        app.step(f, None, 1.0)
    assert app.canvas.mask.any()
    app.act("action:undo")
    app.act("action:undo")
    assert not app.canvas.mask.any()


def test_draw_fist_blip_draw_keeps_stroke():
    from test_core import index_ratio
    app = main.App(640, 360)
    f = np.zeros((360, 640, 3), np.uint8)
    for i in range(20):
        h = index_ratio(1.5); h[:, 0] += 0.01 * i
        app.step(f, h, i / 30)
    n0 = int(app.canvas.mask.sum())
    assert n0 > 0
    for i in range(3):                                  # puño dudoso de 3 frames
        app.step(f, hand((0, 0, 0, 0)), 1 + i / 30)
    for i in range(10):
        h = index_ratio(1.5); h[:, 0] += 0.3
        app.step(f, h, 2 + i / 30)
    assert app.ui.mode == "DIBUJANDO" and app.canvas.mask.sum() >= n0


