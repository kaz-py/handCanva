import os
import sys

import random

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_core import hand  # noqa: E402
import main  # noqa: E402
import theme as T  # noqa: E402

W, H = 1280, 720


def select_at(x, y):
    h = hand((1, 1, 0, 0))
    h[:, 0] += x / W - h[8, 0]; h[:, 1] += y / H - h[8, 1]   # traslada la mano entera
    return h


@pytest.mark.parametrize("color", list(T.COLORS))
def test_hit_and_dwell_selects_each_color(color):
    app = main.App(W, H)
    app.ui.color = "red" if color != "red" else "black"
    x0, y0, x1, y1 = app.overlay._el[f"color:{color}"]["hit"]
    c = ((x0 + x1) // 2, (y0 + y1) // 2)
    assert app.overlay.hit_test(*c) == f"color:{color}"
    frame = np.zeros((H, W, 3), np.uint8)
    t = 0.0
    for _ in range(60):                       # ~2 s a 30 FPS
        app.step(frame, select_at(*c), t); t += 1 / 30
    assert app.ui.color == color


def test_pinch_selects_color():
    app = main.App(W, H)
    x0, y0, x1, y1 = app.overlay._el["color:green"]["hit"]
    c = ((x0 + x1) // 2, (y0 + y1) // 2)
    frame = np.zeros((H, W, 3), np.uint8)
    for i in range(8):
        app.step(frame, select_at(*c), i / 30)
    p = select_at(*c); p[4, :2] = p[8, :2]
    for k in range(4):                                  # el pellizco necesita 2 frames de confirmación
        app.step(frame, p, 0.3 + k / 30)
    assert app.ui.color == "green"


def center(app, bid):
    x0, y0, x1, y1 = app.overlay._el[bid]["hit"]
    return (x0 + x1) // 2, (y0 + y1) // 2


@pytest.mark.parametrize("jitter", [0, 6])
@pytest.mark.parametrize("bid", ["color:green", "tool:circle", "tool:eraser", "thick:+", "thick:-",
                                 "action:undo", "action:clear"])
def test_every_button_end_to_end(bid, jitter):
    app = main.App(W, H)
    calls = []
    app.canvas.undo = lambda: calls.append("undo")
    app.canvas.clear = lambda: calls.append("clear")
    th0 = app.ui.thickness
    rng = random.Random(1)
    cx, cy = center(app, bid)
    frame = np.zeros((H, W, 3), np.uint8)
    t = 0.0
    for _ in range(45):                       # 1.5 s
        app.step(frame, select_at(cx + rng.randint(-jitter, jitter), cy + rng.randint(-jitter, jitter)), t)
        t += 1 / 30
    kind, _, val = bid.partition(":")
    if kind == "color":
        assert app.ui.color == val
    elif kind == "tool":
        assert app.ui.tool == val
    elif kind == "thick":
        assert app.ui.thickness == th0 + (2 if val == "+" else -2)
    else:
        assert calls == [val]


def test_dwell_survives_one_lost_frame():
    app = main.App(W, H)
    cx, cy = center(app, "color:blue")
    frame = np.zeros((H, W, 3), np.uint8)
    t = 0.0
    for i in range(45):
        app.step(frame, None if i == 12 else select_at(cx, cy), t)
        t += 1 / 30
    assert app.ui.color == "blue"
