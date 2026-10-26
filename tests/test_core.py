import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from canvas import Canvas  # noqa: E402
from gestures import DRAW, IDLE, SELECT, Dwell, GestureTracker, classify, is_pinch  # noqa: E402


def hand(up=(0, 0, 0, 0), pinch=False, thumb=False):
    """Mano sintética: muñeca abajo, dedos hacia arriba (y decreciente) si up, plegados si no."""
    lm = np.zeros((21, 3), np.float32)
    lm[0, :2] = (0.5, 0.9)
    lm[3, :2] = (0.3, 0.7) if thumb else (0.38, 0.7)
    lm[4, :2] = (0.5, 0.5) if pinch else ((0.22, 0.62) if thumb else (0.5, 0.65))
    for i, base in enumerate((5, 9, 13, 17)):
        x = 0.42 + 0.05 * i
        lm[base, :2] = (x, 0.6)
        lm[base + 1, :2] = (x, 0.5)                              # PIP
        ext = up[i]
        lm[base + 2, :2] = (x, 0.4 if ext else 0.58)
        lm[base + 3, :2] = (x, 0.3 if ext else 0.62)             # punta
    if pinch:
        lm[8, :2] = (0.5, 0.5)
    return lm


def test_classify():
    assert classify(hand((1, 1, 0, 0))) == SELECT
    assert classify(hand((1, 0, 0, 0))) == DRAW
    assert classify(hand((0, 0, 0, 0))) == IDLE                  # puño: sin acción
    assert classify(hand((0, 0, 0, 0), thumb=True)) == IDLE
    assert classify(hand((1, 1, 1, 1))) == IDLE
    assert classify(hand((0, 1, 0, 0))) == IDLE


def test_pinch():
    assert is_pinch(hand((1, 1, 0, 0), pinch=True))
    assert not is_pinch(hand((1, 1, 0, 0)))


def test_hysteresis_and_ema():
    t = GestureTracker(1000, 1000, hysteresis=4)
    d = hand((1, 0, 0, 0))
    modes = [t.update(d)[0] for _ in range(5)]
    assert modes[:3] == [IDLE] * 3 and modes[3] == DRAW       # cambia al 4.º frame
    t.update(hand((1, 1, 0, 0)))                              # un frame suelto no cambia
    assert t.mode == DRAW
    t.update(None)
    assert t.pointer is not None and t.hands                  # 1 frame sin detección: se conserva la mano
    for _ in range(2):
        t.update(None)
    assert t.pointer is None and t.hands == []
    p0 = t.update(d)[1]
    d2 = d.copy(); d2[8, 0] += 0.2
    p1 = t.update(d2)[1]
    assert p0[0] - 200 < p1[0] < p0[0]                        # suavizado: no salta entero


def test_dwell():
    w = Dwell(1.0)
    assert w.update("a", 0.0) == (0.0, None)
    assert w.update("a", 0.5) == (0.5, None)
    assert w.update("a", 1.0) == (1.0, "a")
    assert w.update("a", 2.0)[1] is None                      # no repite
    w.update(None, 2.1)
    assert w.update("a", 2.2)[1] is None and w.update("a", 3.3)[1] == "a"
    w.update("b", 4.0)
    assert w.update("a", 4.1)[0] == 0.0                       # cambiar de objetivo reinicia


def test_canvas_undo_clear():
    c = Canvas(200, 100)
    c.begin("brush", (0, 0, 255), 4, (10, 10)); c.move((100, 50)); c.end()
    assert c.mask.any()
    c.clear()
    assert not c.mask.any()
    c.undo()
    assert c.mask.any()
    c.undo()
    assert not c.mask.any()
    c.undo()                                                  # pila vacía: no falla


def test_canvas_shape_preview_and_commit():
    c = Canvas(200, 100)
    c.begin("rect", (0, 255, 0), 2, (10, 10)); c.move((90, 60))
    assert not c.mask.any()                                   # aún solo vista previa
    f = np.zeros((100, 200, 3), np.uint8)
    c.blend(f)
    assert f.any() and not c.layer.any()
    c.end()
    assert c.mask.any()


def test_undo_bounded_and_eraser():
    c = Canvas(50, 50, max_undo=3)
    for i in range(6):
        c.begin("brush", (255, 255, 255), 2, (5, 5 + i)); c.end()
    assert len(c._undo) == 3
    c.begin("eraser", (0, 0, 0), 5, (5, 5)); c.move((5, 12)); c.end()
    assert not c.mask[5:11, 5].any()


def test_fist_is_idle_and_keeps_pointer():
    t = GestureTracker(1000, 1000)
    for _ in range(10):
        t.update(hand((0, 0, 0, 0)))
    assert t.mode == IDLE and not t.pinch and t.pointer is not None


def shifted(dx, up=(1, 0, 0, 0)):
    h = hand(up); h[:, 0] += dx
    return h


def test_two_hands_independent_modes():
    t = GestureTracker(1000, 1000)
    for _ in range(10):
        t.update([shifted(-0.3), shifted(0.3, (1, 1, 0, 0))])
    assert [sl.mode for sl in t.slots] == [DRAW, SELECT] and t.mode == SELECT     # prioridad: SELECCIÓN
    for _ in range(10):                            # identidad por proximidad aunque cambie el orden de detección
        t.update([shifted(0.3, (1, 1, 0, 0)), shifted(-0.3)])
    assert [sl.mode for sl in t.slots] == [DRAW, SELECT]


def index_ratio(r):
    """Mano con solo el índice, cuya punta/PIP vale r (muñeca en y=0.9, PIP a 0.4)."""
    h = hand((1, 0, 0, 0)); h[8, 1] = 0.9 - 0.4 * r
    return h


def test_drawing_with_foreshortened_index_stays_drawing_or_idle():
    rng = np.random.default_rng(0)
    t = GestureTracker(1000, 1000)
    for _ in range(8):
        t.update(index_ratio(1.5))
    assert t.mode == DRAW
    for i in range(400):
        r = 1.35 + 0.25 * np.sin(i / 5) + rng.uniform(-0.05, 0.05)   # oscila 1.1-1.6
        t.update(index_ratio(r))
        assert t.mode in (DRAW, IDLE)


def test_fist_with_touching_thumb_index_is_idle_and_never_pinches():
    # Puño donde pulgar e índice se tocan: no es mano de dibujo ni pellizco
    h = hand((0, 0, 0, 0), pinch=True)
    assert classify(h) == IDLE
    t = GestureTracker(1000, 1000)
    for _ in range(8):
        t.update(h)
    assert t.mode == IDLE


def pinch_hand(x=0.5, y=0.6, pinched=True, size=0.30):
    """Mano de dibujo (16:9): índice extendido, o pellizco con la punta del pulgar tocando la del índice."""
    import synth
    rng = np.random.default_rng(3)
    m = synth.hand_model(("half" if pinched else "ext", "curl", "curl", "curl"), "out", rng)
    lm = synth.place(m, rng, size, (x, y))
    if pinched:
        lm[4] = lm[8] + (0.002, 0.002, 0)
        lm[3] = (lm[2] + lm[4]) / 2
    return lm
