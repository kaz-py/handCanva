"""HUD de pizarra blanca: superficie cuadriculada, bandeja de marcadores, borrador de fieltro,
imanes, nota adhesiva de estado, puntero y anillo de dwell.

Todo lo estático (papel, bandeja, objetos en cada estado, tooltips, nota, punteros) se
renderiza UNA vez en __init__ como sprites (BGR premultiplicado + inverso de alfa); por frame
solo se componen sobre ROIs pequeñas: out = roi * inv/255 + pre.
Orden en main.py: draw_board(frame) -> lienzo -> draw(frame, state).
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

import theme as T

AA = cv2.LINE_AA
FONT = cv2.FONT_HERSHEY_SIMPLEX
SCRIPT = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
_ACC = {"Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U", "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u"}

MARKER_IDS = [f"color:{c}" for c in T.COLORS]
TOOL_IDS = [f"tool:{t}" for t in T.TOOL_LABELS]
BUTTON_IDS = MARKER_IDS + TOOL_IDS + ["thick:-", "thick:+", "action:undo", "action:clear"]
MAGNET_IDS = ["tool:brush", "tool:line", "tool:rect", "tool:circle", "thick:-", "thick:+"]
LABEL_IDS = ["action:undo", "action:clear"]
_LABEL_ICON = {"action:undo": "undo", "action:clear": "trash"}
_MAGNET_ICON = {"tool:brush": "brush", "tool:line": "line", "tool:rect": "rect", "tool:circle": "circle",
                "thick:-": "minus", "thick:+": "plus"}


@dataclass
class UIState:
    mode: str = "EN ESPERA"            # "DIBUJANDO" | "SELECCIÓN" | "EN ESPERA"
    fps: float = 0.0
    pointer: tuple[int, int] | None = None   # punta del índice en px del frame
    hover_id: str | None = None        # botón bajo el puntero
    dwell: float = 0.0                 # 0..1 progreso del dwell sobre hover_id
    color: str = "red"                 # id de color activo
    tool: str = "brush"                # "brush"|"line"|"rect"|"circle"|"eraser"
    thickness: int = 8
    measure: tuple[tuple[int, int], tuple[int, int]] | None = None   # puntas de la línea a dos manos
    hands: list[list[tuple[int, int]]] = field(default_factory=list)  # 21 landmarks por mano, en px de pantalla
    pinch: bool = False                                                # índice+pulgar juntos (mano 0)
    pinches: list[bool] = field(default_factory=list)                  # pellizco por mano (índice = mano)
    hand_colors: list[str] = field(default_factory=list)              # id de color por mano (índice = mano)
    pointers: list[tuple[int, int] | None] = field(default_factory=list)  # un puntero por mano; si está vacío rige `pointer`


# ---------------------------------------------------------------- texto
def _base(text):
    return "".join(_ACC.get(c, c) for c in text)


def _tw(text, scale, th, font=FONT):
    return cv2.getTextSize(_base(text), font, scale, th)[0][0]


def _ink(text, scale, th, font=FONT, k=None):
    """(margen izquierdo, ancho) en px de la tinta real tal como la pinta _Art.text (a K x): getTextSize a 1x
    no sirve, incluye márgenes de glifo y Hershey no escala lineal (a 3x el texto sale ~10 % más ancho)."""
    k = k or K
    scale, th = scale * k, int(th * k)
    (tw, tht), _ = cv2.getTextSize(_base(text), font, scale, th)
    pad = tht + th
    m = np.zeros((3 * tht + 2 * th, tw + 2 * pad), np.uint8)
    cv2.putText(m, _base(text), (pad, 2 * tht), font, scale, 255, th, AA)
    xs = np.nonzero(m.any(0))[0]
    return round((xs[0] - pad) / k), round((xs[-1] + 1 - xs[0]) / k)


def _cap(scale):
    return round(21 * scale)   # getTextSize da la misma altura a toda letra; Hershey: mayúscula ~21 px a escala 1


def _put(img, text, org, scale, color, th, font=FONT):
    """putText con tildes: Hershey no trae acentos, se dibuja un trazo agudo encima."""
    x, y = org
    base = _base(text)
    cv2.putText(img, base, (x, y), font, scale, color, th, AA)
    cap, xh = _cap(scale), round(14.5 * scale)
    for i, c in enumerate(text):
        if c not in _ACC:
            continue
        cx = x + _tw(base[:i], scale, th, font) + _tw(base[i], scale, th, font) / 2
        top = y - (xh if c.islower() and c != "í" else cap)
        k = max(2.0, cap * 0.13)
        cv2.line(img, (round(cx - k), round(top - 0.6 * k)), (round(cx + k), round(top - 2.8 * k)), color, th, AA)


# ---------------------------------------------------------------- arte offline
K = 3   # supersampling: todo sprite se dibuja a K x y se reduce UNA vez (INTER_AREA) a su tamaño exacto
_CYL_T = [0, .15, .35, .62, .85, 1]
_CYL_F = [.60, .92, 1.10, .95, .74, .52]     # luz desde la izquierda


class _Sprite:
    __slots__ = ("x", "y", "pre", "inv")

    def __init__(self, x0, y0, pre, inv):
        self.x, self.y, self.pre, self.inv = x0, y0, pre, inv


class _Art:
    """Lienzo RGBA premultiplicado (float) a self.k x, con origen global (x0, y0) en px."""

    def __init__(self, x0, y0, w, h, k=K):
        self.k = k
        self.ox, self.oy, self.w, self.h = int(x0), int(y0), int(w), int(h)
        self.pre = np.zeros((self.h * self.k, self.w * self.k, 3), np.float32)
        self.a = np.zeros((self.h * self.k, self.w * self.k), np.float32)

    def _p(self, pts):
        return np.round((np.asarray(pts, np.float64) - [self.ox, self.oy]) * self.k).astype(np.int32)

    def _c(self, c):
        return (int(round((c[0] - self.ox) * self.k)), int(round((c[1] - self.oy) * self.k)))

    def paint(self, color, fn, op=1.0, blur=0.0):
        """color: BGR o callable(x0,y0,x1,y1)->array (degradados). blur en px finales."""
        m = np.zeros(self.a.shape, np.uint8)
        fn(m)
        x, y, w, h = cv2.boundingRect(m)
        if w == 0:
            return
        g = int(3 * blur * self.k) + 1 if blur > 0 else 0
        x0, y0, x1, y1 = max(x - g, 0), max(y - g, 0), min(x + w + g, m.shape[1]), min(y + h + g, m.shape[0])
        f = m[y0:y1, x0:x1].astype(np.float32) * (op / 255)
        if blur > 0:
            f = cv2.GaussianBlur(f, (0, 0), blur * self.k)
        f3 = f[..., None]
        c = color(x0, y0, x1, y1) if callable(color) else np.array(color, np.float32)
        pre, a = self.pre[y0:y1, x0:x1], self.a[y0:y1, x0:x1]
        pre[:] = pre * (1 - f3) + c * f3
        a[:] = f + a * (1 - f)

    def poly(self, color, pts, th=-1, op=1.0, blur=0.0):
        p = self._p(pts)
        t = max(1, int(round(th * self.k)))
        self.paint(color, (lambda m: cv2.fillPoly(m, [p], 255, AA)) if th < 0
                   else (lambda m: cv2.polylines(m, [p], True, 255, t, AA)), op, blur)

    def stroke(self, color, pts, th, op=1.0):
        p, t = self._p(pts), max(1, int(round(th * self.k)))
        self.paint(color, lambda m: cv2.polylines(m, [p], False, 255, t, AA), op)

    def circle(self, color, c, r, th=-1, op=1.0):
        cc, rr_ = self._c(c), max(1, int(round(r * self.k)))
        t = -1 if th < 0 else max(1, int(round(th * self.k)))
        self.paint(color, lambda m: cv2.circle(m, cc, rr_, 255, t, AA), op)

    def ellipse(self, color, c, axes, op=1.0, blur=0.0):
        cc, ax = self._c(c), (max(1, int(round(axes[0] * self.k))), max(1, int(round(axes[1] * self.k))))
        self.paint(color, lambda m: cv2.ellipse(m, cc, ax, 0, 0, 360, 255, -1, AA), op, blur)

    def rrect(self, color, box, r, th=-1, op=1.0, blur=0.0):
        self.poly(color, _rr(*box, r, f=True), th, op, blur)

    def text(self, color, txt, org, scale, th, font=FONT):
        o = self._c(org)
        self.paint(color, lambda m: _put(m, txt, o, scale * self.k, 255, int(th * self.k), font))

    def icon(self, color, name, cx, cy, a, th):
        c = self._c((cx, cy))
        self.paint(color, lambda m: _icon(m, name, c[0], c[1], a * self.k, int(th * self.k)))

    def shaded(self, color, pts, xl, xr, op=1.0):
        """Relleno con sombreado cilíndrico horizontal entre xl y xr (luz a la izquierda)."""
        p = self._p(pts)
        xl, xr = (xl - self.ox) * self.k, (xr - self.ox) * self.k
        base = np.array(color, np.float32)

        def col(x0, y0, x1, y1):
            t = np.clip((np.arange(x0, x1) - xl) / max(xr - xl, 1), 0, 1)
            return np.clip(base * np.interp(t, _CYL_T, _CYL_F)[None, :, None].astype(np.float32), 0, 255)
        self.paint(col, lambda m: cv2.fillPoly(m, [p], 255, AA), op)

    def vshaded(self, pts, y0, y1, stops):
        """Relleno con degradado vertical; stops = [(t, BGR), ...]."""
        p = self._p(pts)
        y0, y1 = (y0 - self.oy) * self.k, (y1 - self.oy) * self.k
        ts = [t for t, _ in stops]
        cs = np.array([c for _, c in stops], np.float32)

        def col(xa, ya, xb, yb):
            t = np.clip((np.arange(ya, yb) - y0) / max(y1 - y0, 1), 0, 1)
            return np.stack([np.interp(t, ts, cs[:, i]) for i in range(3)], -1)[:, None, :].astype(np.float32)
        self.paint(col, lambda m: cv2.fillPoly(m, [p], 255, AA))

    def layer(self, pre, a, gx, gy):
        """Compone encima otra capa (ya a self.k x) con su esquina en px globales enteros."""
        x0, y0 = int(round((gx - self.ox) * self.k)), int(round((gy - self.oy) * self.k))
        h, w = a.shape
        sx, sy = max(0, -x0), max(0, -y0)
        ex, ey = min(w, self.a.shape[1] - x0), min(h, self.a.shape[0] - y0)
        if ex <= sx or ey <= sy:
            return
        A = a[sy:ey, sx:ex]
        d_pre = self.pre[y0 + sy:y0 + ey, x0 + sx:x0 + ex]
        d_a = self.a[y0 + sy:y0 + ey, x0 + sx:x0 + ex]
        d_pre[:] = pre[sy:ey, sx:ex] + d_pre * (1 - A[..., None])
        d_a[:] = A + d_a * (1 - A)

    def rotated(self, angle, pivot):
        """Devuelve (pre, a) rotados `angle` grados (horario<0) alrededor de pivot (px locales)."""
        M = cv2.getRotationMatrix2D((pivot[0] * self.k, pivot[1] * self.k), angle, 1.0)
        sz = (self.a.shape[1], self.a.shape[0])
        pre = cv2.warpAffine(self.pre, M, sz, flags=cv2.INTER_LINEAR)
        a = cv2.warpAffine(self.a, M, sz, flags=cv2.INTER_LINEAR)
        return pre, a

    def sprite(self):
        pre = cv2.resize(self.pre, (self.w, self.h), interpolation=cv2.INTER_AREA)
        a = cv2.resize(self.a, (self.w, self.h), interpolation=cv2.INTER_AREA)
        inv = np.repeat(np.clip(255 - a * 255, 0, 255).astype(np.uint8)[..., None], 3, 2)
        return _Sprite(self.ox, self.oy, np.clip(pre, 0, 255).astype(np.uint8), inv)


def _blit(f, sp):
    h, w = sp.pre.shape[:2]
    H, W = f.shape[:2]
    x0, y0, x1, y1 = max(sp.x, 0), max(sp.y, 0), min(sp.x + w, W), min(sp.y + h, H)
    if x0 >= x1 or y0 >= y1:
        return
    sx, sy = x0 - sp.x, y0 - sp.y
    roi = f[y0:y1, x0:x1]
    cv2.multiply(roi, sp.inv[sy:sy + y1 - y0, sx:sx + x1 - x0], dst=roi, scale=1 / 255)
    cv2.add(roi, sp.pre[sy:sy + y1 - y0, sx:sx + x1 - x0], dst=roi)


def _rr(x0, y0, x1, y1, r, n=24, f=False):
    """Rectángulo redondeado, sentido horario desde el inicio del borde superior derecho."""
    r = max(0.5, min(r, (x1 - x0) / 2, (y1 - y0) / 2))
    pts = []
    for cx, cy, a0 in ((x1 - r, y0 + r, -90), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180)):
        for k in range(n + 1):
            t = np.radians(a0 + 90 * k / n)
            pts.append((cx + r * np.cos(t), cy + r * np.sin(t)))
    a = np.array(pts)
    return a if f else np.round(a).astype(np.int32)


def _perim(shape, n=180):
    """Contorno de un objeto resampleado a n+1 puntos equiespaciados desde arriba-centro, en horario."""
    if shape[0] == "c":
        _, cx, cy, R = shape
        t = np.radians(-90 + 360 * np.arange(n + 1) / n)
        return np.round(np.c_[cx + R * np.cos(t), cy + R * np.sin(t)]).astype(np.int32)
    _, x0, y0, x1, y1, r = shape
    mid = np.array([[(x0 + x1) / 2, y0]])
    path = np.vstack([mid, _rr(x0, y0, x1, y1, r, f=True), mid])
    cum = np.r_[0, np.cumsum(np.hypot(*np.diff(path, axis=0).T))]
    u = np.linspace(0, cum[-1], n + 1)
    return np.round(np.c_[np.interp(u, cum, path[:, 0]), np.interp(u, cum, path[:, 1])]).astype(np.int32)


def _icon(m, name, cx, cy, a, th):
    """Iconos de rotulador: línea del mismo grosor en todos. a = semi-extensión."""
    L = lambda p, q: cv2.line(m, (round(p[0]), round(p[1])), (round(q[0]), round(q[1])), 255, th, AA)
    poly = lambda pts, closed=True: cv2.polylines(m, [np.round(np.array(pts)).astype(np.int32)], closed, 255, th, AA)
    if name == "brush":       # garabato a mano alzada
        xs = np.linspace(-a, a, 24)
        poly([(cx + x, cy + 0.5 * a * np.sin((x + a) / a * np.pi * 1.25)) for x in xs], False)
    elif name == "line":
        L((cx - a, cy + a), (cx + a, cy - a))
        for s in (-1, 1):
            cv2.circle(m, (round(cx + s * a), round(cy - s * a)), max(3, th + 1), 255, th, AA)
    elif name == "rect":
        cv2.rectangle(m, (round(cx - a), round(cy - .75 * a)), (round(cx + a), round(cy + .75 * a)), 255, th, AA)
    elif name == "circle":
        cv2.circle(m, (round(cx), round(cy)), round(a), 255, th, AA)
    elif name == "eraser":
        t = np.radians(-35)
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        q = lambda x, y: np.array([cx, cy]) + R @ np.array([x * a, y * a])
        poly([q(-1, -.5), q(1, -.5), q(1, .5), q(-1, .5)])
        L(q(-.2, -.5), q(-.2, .5))
    elif name in ("minus", "plus"):
        L((cx - a * .8, cy), (cx + a * .8, cy))
        if name == "plus":
            L((cx, cy - a * .8), (cx, cy + a * .8))
    elif name == "undo":
        cv2.ellipse(m, (round(cx + .1 * a), round(cy + .15 * a)), (round(.85 * a), round(.7 * a)), 0, 180, 400, 255, th, AA)
        hx, hy = cx - .75 * a, cy + .15 * a
        L((hx, hy), (hx - .45 * a, hy - .5 * a))
        L((hx, hy), (hx + .5 * a, hy - .45 * a))
    elif name == "trash":
        L((cx - .85 * a, cy - .5 * a), (cx + .85 * a, cy - .5 * a))
        poly([(cx - .3 * a, cy - .5 * a), (cx - .3 * a, cy - .85 * a), (cx + .3 * a, cy - .85 * a), (cx + .3 * a, cy - .5 * a)], False)
        poly([(cx - .65 * a, cy - .5 * a), (cx - .5 * a, cy + .95 * a), (cx + .5 * a, cy + .95 * a), (cx + .65 * a, cy - .5 * a)], False)
    elif name == "cursor":
        pts = [(-.5, -.95), (-.5, .75), (-.15, .4), (.12, .98), (.4, .85), (.14, .3), (.6, .3)]
        poly([(cx + x * a, cy + y * a) for x, y in pts])
    elif name == "pause":
        for s in (-.4, .4):
            L((cx + s * a, cy - .7 * a), (cx + s * a, cy + .7 * a))


def _wrap(text, scale, th, maxw):
    lines, cur = [], ""
    for w in text.split():
        trial = (cur + " " + w).strip()
        if cur and _tw(trial, scale, th) > maxw:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    return lines + [cur] if cur else lines


def _capsule(art, p, q, w, color):
    a_, b_, k = art._c(p), art._c(q), art.k
    t_, r_ = max(1, int(round(w * k))), max(1, int(round(w * k / 2)))

    def fn(m):
        cv2.line(m, a_, b_, 255, t_, AA)
        cv2.circle(m, a_, r_, 255, -1, AA)
        cv2.circle(m, b_, r_, 255, -1, AA)
    art.paint(color, fn)


def _hand(art, cx, ty, u, ext, et):
    """Mano de línea (tinta sobre blanco). ext = dedos extendidos {'i','m','r','p','t'}. Devuelve las puntas extendidas
    (índice, medio, ...). ty = y de la base de los dedos."""
    fill = (253, 252, 250)
    part = lambda p, q, w: (_capsule(art, p, q, w + 2 * et, T.INK), _capsule(art, p, q, w, fill))
    xs = {"i": -1.5, "m": -0.5, "r": 0.5, "p": 1.5}
    ln = {"i": 3.0, "m": 3.4, "r": 3.0, "p": 2.3}
    w = 0.92 * u
    tips = []
    for f in ("p", "r", "m", "i"):
        if f in ext:
            part((cx + xs[f] * u, ty + 0.8 * u), (cx + xs[f] * u, ty - ln[f] * u), w)
    if "t" in ext:
        part((cx - 1.8 * u, ty + 2.3 * u), (cx - 3.5 * u, ty + 0.9 * u), w)
    art.rrect(T.INK, (cx - 2.1 * u - et, ty - 0.2 * u - et, cx + 2.1 * u + et, ty + 3.1 * u + et), 1.0 * u)
    art.rrect(fill, (cx - 2.1 * u, ty - 0.2 * u, cx + 2.1 * u, ty + 3.1 * u), 0.9 * u)
    for f in ("i", "m", "r", "p"):
        if f not in ext:
            part((cx + xs[f] * u, ty + 0.35 * u), (cx + xs[f] * u, ty + 1.45 * u), w)
    if "t" not in ext:
        part((cx - 1.6 * u, ty + 2.2 * u), (cx + 0.1 * u, ty + 2.0 * u), 0.95 * u)
    for f in ("i", "m"):
        if f in ext:
            tips.append((cx + xs[f] * u, ty - ln[f] * u))
    return tips or [(cx, ty)]


_HAND_BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
               (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
_PALM = [0, 1, 5, 9, 13, 17]


# ---------------------------------------------------------------- Overlay
class Overlay:
    def __init__(self, width: int, height: int):
        self._build(width, height)

    def resize(self, width: int, height: int) -> None:
        """Reconstruye layout y sprites para otro tamaño (llamar solo al cambiar resolución)."""
        self._build(width, height)

    def _build(self, width, height):
        self.W, self.H = width, height
        sc = self.sc = min(width / T.BASE_W, height / T.BASE_H)
        S = lambda v: int(round(v * sc))
        s = self.s = max(S(T.TOUCH), 56)
        fw, mg, pitch = S(T.FRAME_W), S(T.MARGIN), s + S(8)
        disc = (s - S(8)) / 2
        ith = max(2, S(T.LINE))
        fs, fth = max(T.FONT_SCALE * sc, 0.7), max(1, round(sc))
        cap = _cap(fs)
        self._fw = fw
        self._el, self._idle, self._act, self._tip = {}, {}, {}, {}

        self._make_paper()

        # --- nota adhesiva de estado (arriba-izquierda)
        sp, ic = S(16), S(36)
        tx0 = sp + ic + S(12)
        row = cap + S(16)
        sh = 2 * sp + cap + row
        w1 = max(_tw("MODO: " + m, fs, fth + 1) for m in T.MODES) + S(6) + S(52)
        lab_fps = _tw("FPS 000", fs, fth) + S(24)
        sw = tx0 + max(w1, lab_fps + _tw("Grosor 00 px", fs, fth)) + sp
        nx, ny = fw + mg, fw + mg
        y1, y2 = ny + sp + cap, ny + sp + cap + row
        fo = S(24)
        self._note = {}
        for mode, icon in T.MODES.items():
            a = _Art(nx, ny, sw, sh)
            a.poly(T.NOTE, [(nx, ny), (nx + sw, ny), (nx + sw, ny + sh - fo), (nx + sw - fo, ny + sh), (nx, ny + sh)])
            a.poly(T.NOTE_FOLD, [(nx + sw - fo, ny + sh), (nx + sw, ny + sh - fo), (nx + sw - fo, ny + sh - fo)])
            a.icon(T.INK, icon, nx + sp + ic // 2, ny + sh // 2, ic * .36, ith)
            a.text(T.INK_DIM, "MODO:", (nx + tx0, y1), fs, fth)
            a.text(T.INK, mode, (nx + tx0 + _tw("MODO: ", fs, fth) + S(6), y1), fs, fth + 1)
            a.text(T.INK_DIM, "FPS", (nx + tx0, y2), fs, fth)
            a.text(T.INK_DIM, "Grosor", (nx + tx0 + lab_fps, y2), fs, fth)
            self._note[mode] = a.sprite()
        self._dot_r, self._dot_dx = S(7), S(22)
        self._dot_x, self._dot_y = nx + sw - sp - S(11), y1 - cap // 2
        self._fps_xy = (nx + tx0 + _tw("FPS ", fs, fth) + S(6), y2)
        self._th_xy = (nx + tx0 + lab_fps + _tw("Grosor ", fs, fth) + S(6), y2)
        self._fs, self._fth = fs, fth

        # --- bandeja: marcadores + borrador
        mp = max(S(T.MARKER_PITCH), pitch)
        pad, gap, ew = S(32), S(32), max(S(136), s + S(24))
        tw_ = 2 * pad + 5 * mp + gap + ew
        tx, floor_y = (width - tw_) // 2, height - fw
        lip_h, top_h = S(44), S(14)
        lip_top = floor_y - lip_h
        led_top = lip_top - top_h
        self._mbase = lip_top + S(24)               # la base queda oculta tras el labio
        mh = S(140)
        hit_y0 = self._mbase - mh - S(32) - S(14)
        self._build_tray(tx, tw_, led_top, lip_top, floor_y)
        self._mx, self._mtop = {}, self._mbase - mh - S(30)
        sh_y = led_top + top_h * 0.55                     # donde cae la sombra sobre la bandeja
        for i, bid in enumerate(MARKER_IDS):
            xc = tx + pad + mp // 2 + i * mp
            self._mx[bid] = xc
            hit = (xc - mp // 2, hit_y0, xc + mp // 2, floor_y)
            self._add(bid, ("r", *hit, S(16)), hit)
            col = T.COLORS[bid.split(":")[1]]
            for table, act in ((self._idle, False), (self._act, True)):
                a = _Art(hit[0] - S(8), hit[1] - S(8), mp + S(16), floor_y - hit[1] + S(8))
                a.ellipse(T.INK, (xc + (S(10) if act else S(4)), sh_y), (S(22) if act else S(20), S(4)),
                          op=0.14 if act else 0.30, blur=S(2.5))
                pre, al, bx, by = self._marker_layer(col, act)
                a.layer(pre, al, xc - bx, self._mbase - (S(32) if act else 0) - by)
                table[bid] = a.sprite()
        xe = tx + tw_ - pad - ew // 2
        eb = lip_top - S(2)
        hit = (xe - ew // 2, eb - S(60) - S(20), xe + ew // 2, floor_y)
        self._add("tool:eraser", ("r", *hit, S(16)), hit)
        pre, al, bx, by = self._eraser_layer(ew - S(16))
        for table, rz in ((self._idle, 0), (self._act, S(16))):
            a = _Art(hit[0] - S(8), hit[1] - S(8), ew + S(16), floor_y - hit[1] + S(8))
            a.ellipse(T.INK, (xe + S(6), sh_y + S(2)), (ew * 0.40, S(4)), op=0.14 if rz else 0.30, blur=S(2.5))
            a.layer(pre, al, xe - bx, eb - rz - by)
            table["tool:eraser"] = a.sprite()

        # --- imanes (columna izquierda: herramientas; derecha: grosor)
        cxl = nx + s // 2
        y0 = ny + sh + mg + S(24)
        centers = {}
        for i, bid in enumerate(["tool:brush", "tool:line", "tool:rect", "tool:circle"]):
            centers[bid] = (cxl, y0 + s // 2 + i * pitch)
        # --- etiquetas magnéticas (arriba-derecha) y grosor debajo
        lab_font, lab_sc, lab_th = SCRIPT, T.SCRIPT_SCALE * max(sc, 0.75), max(2, round(sc * 1.5))
        lw = max(_tw(T.ACTION_LABELS[b], lab_sc, lab_th, lab_font) for b in LABEL_IDS) + S(32) + S(36) + S(12)
        lx1 = width - fw - mg
        lx0 = lx1 - max(lw, s * 2)
        for i, bid in enumerate(LABEL_IDS):
            top = fw + mg + i * pitch
            box = (lx0, top, lx1, top + s)
            self._add(bid, ("r", lx0 - S(4), top - S(4), lx1 + S(4), top + s + S(4), s // 2 + S(4)), box)
            for table, act in ((self._idle, False), (self._act, True)):
                if act:
                    continue
                a = _Art(lx0 - S(8), top - S(8), lx1 - lx0 + S(16), s + S(16))
                a.rrect(T.MAGNET_EDGE, (box[0], box[1] + S(3), box[2], box[3] + S(3)), s // 2)
                a.rrect(T.MAGNET_FACE, box, s // 2)
                a.rrect(T.MAGNET_RIM, box, s // 2, th=max(1, S(2)))
                il, iw = _ink(T.ACTION_LABELS[bid], lab_sc, lab_th, lab_font)
                icx = lx0 + ((lx1 - lx0) - (S(38) + iw)) // 2 + S(14)    # grupo icono+texto centrado en la etiqueta
                a.icon(T.INK, _LABEL_ICON[bid], icx, top + s // 2, S(14), ith)
                a.text(T.INK, T.ACTION_LABELS[bid], (icx + S(24) - il, top + s // 2 + cap // 2), lab_sc, lab_th, lab_font)
                table[bid] = a.sprite()
        ty = fw + mg + 2 * pitch + S(16) + s // 2
        centers["thick:+"] = (lx1 - s // 2, ty)
        centers["thick:-"] = (lx1 - s // 2 - pitch, ty)
        for bid, (cx, cy) in centers.items():
            self._add(bid, ("c", cx, cy, s // 2 + S(4)), (cx - s // 2, cy - s // 2, cx + s // 2, cy + s // 2))
            for table, act in ((self._idle, False), (self._act, True)):
                if act and bid.startswith("thick"):
                    continue
                a = _Art(cx - s // 2 - S(8), cy - s // 2 - S(8), s + S(16), s + S(16))
                a.circle(T.INK if act else T.MAGNET_EDGE, (cx, cy + S(3)), disc)
                a.circle(T.INK if act else T.MAGNET_FACE, (cx, cy), disc)
                a.circle(T.INK if act else T.MAGNET_RIM, (cx, cy), disc, th=max(1, S(2)))
                a.icon(T.PAPER_WHITE if act else T.INK, _MAGNET_ICON[bid], cx, cy, disc * .5, ith)
                table[bid] = a.sprite()

        # --- hover (contorno de acento + tinte) y tooltips
        self._hover = {}
        for bid, e in self._el.items():
            pts = e["perim"]
            x0, y0_ = pts.min(0) - S(10)
            x1_, y1_ = pts.max(0) + S(10)
            a = _Art(x0, y0_, x1_ - x0, y1_ - y0_)
            a.poly(T.ACCENT, pts, op=0.10)
            a.poly(T.ACCENT, pts, th=max(2, S(3)), op=0.55)
            self._hover[bid] = a.sprite()
            if bid in LABEL_IDS:
                continue
            kind, name = bid.split(":")
            text = T.COLOR_LABELS[name] if kind == "color" else T.TOOL_LABELS[name] if kind == "tool" else T.ACTION_LABELS[bid]
            tpx, tpy = S(16), S(12)
            il, iw = _ink(text, fs, fth)
            w, h = iw + 2 * tpx, cap + 2 * tpy
            hx0, hy0, hx1, hy1 = e["hit"]
            if kind in ("color",) or bid == "tool:eraser":
                px, py = (hx0 + hx1) // 2 - w // 2, hy0 - h - S(4)
            elif kind == "tool":
                px, py = hx1 + S(8), (hy0 + hy1) // 2 - h // 2
            else:
                px, py = (hx0 + hx1) // 2 - w // 2, hy1 + S(8)
            px = min(max(px, fw + S(4)), width - fw - w - S(4))
            a = _Art(px, py, w, h)
            a.rrect(T.TIP_BG, (px, py, px + w, py + h), S(16), op=0.95)
            a.text(T.PAPER_WHITE, text, (px + tpx - il, py + tpy + cap), fs, fth)
            self._tip[bid] = a.sprite()

        # --- punteros
        self._ptr = {}
        for m in T.MODES:
            for col in (T.COLORS if m == "DIBUJANDO" else (None,)):
                self._ptr[(m, col)] = self._make_pointer(m, col)

        self._badge, self._br = {}, S(11)
        fsb = fs * 0.85
        for n_ in ('1', '2'):
            r_ = self._br
            a = _Art(0, 0, 2 * r_ + 4, 2 * r_ + 4)
            a.circle(T.PAPER_WHITE, (r_ + 2, r_ + 2), r_ + 1.5)
            a.circle(T.INK, (r_ + 2, r_ + 2), r_)
            a.text(T.PAPER_WHITE, n_, (r_ + 2 - _tw(n_, fsb, fth + 1) / 2, r_ + 2 + _cap(fsb) / 2), fsb, fth + 1)
            self._badge[n_] = a.sprite()
        txt = "Pellizca para dibujar"
        fsh = fs * 0.85
        tw_h, cap_h = _tw(txt, fsh, fth), _cap(fsh)
        wh, hh = S(12) + S(20) + S(8) + tw_h + S(24), cap_h + S(16)
        a = _Art(nx, ny + sh + S(6), wh, hh)
        a.rrect(T.NOTE, (nx, ny + sh + S(6), nx + wh, ny + sh + S(6) + hh), S(8))
        gx_, gy_ = nx + S(12) + S(6), ny + sh + S(6) + hh / 2
        a.circle(T.INK, (gx_, gy_ - S(2)), S(4))
        a.circle(T.INK, (gx_ + S(9), gy_ + S(2)), S(4))
        a.text(T.INK, txt, (nx + S(12) + S(28), ny + sh + S(6) + hh / 2 + cap_h / 2), fsh, fth)
        self._pinch_hint = a.sprite()
        # --- punteros de figura: retículo abierto (libre) y cerrado/relleno (pellizco) + mini guía de la figura
        self._pin, self._tool_hint = {}, {}
        for col in T.COLORS:
            for closed in (False, True):
                self._pin[(closed, col)] = self._make_shape_pointer(col, closed)
        hr = S(13)
        for tool, icon in (("line", "line"), ("rect", "rect"), ("circle", "circle")):
            a = _Art(0, 0, 2 * hr + 4, 2 * hr + 4)
            a.circle(T.PAPER_WHITE, (hr + 2, hr + 2), hr, op=0.96)
            a.circle(T.INK, (hr + 2, hr + 2), hr, th=max(1, S(1.5)))
            a.icon(T.INK, icon, hr + 2, hr + 2, hr * 0.5, max(1, S(1.5)))
            self._tool_hint[tool] = a.sprite()
        self._pprev, self._ppulse = {}, {}
        self._build_measure(fs, fth, cap)
        self._build_help()

        self._bone_th = max(2, int(round(3 * sc)))
        self._joint_r = max(3, int(round(4 * sc)))

        # --- dwell
        self._ring_th = max(3, S(4))
        self._numc = {}
        self._fired = False
        self._prev = (None, 0.0)
        self._pulse = None  # (t0, bid)
        self._hit = [(e["hit"], bid) for bid, e in self._el.items()]

    # -------------------------------------------------- construcción
    def _add(self, bid, shape, hit):
        self._el[bid] = {"hit": tuple(int(v) for v in hit), "perim": _perim(shape),
                         "center": ((hit[0] + hit[2]) // 2, (hit[1] + hit[3]) // 2)}

    def _make_paper(self):
        W, H, sc, fw = self.W, self.H, self.sc, self._fw
        rng = np.random.default_rng(7)
        p = np.empty((H, W, 3), np.float32)
        p[:] = T.PAPER
        sm = np.zeros((H, W), np.float32)
        for _ in range(7):   # estelas finas de borrado (nítidas, no nubes difusas)
            cv2.ellipse(sm, (int(rng.integers(W // 6, W * 5 // 6)), int(rng.integers(H // 8, H * 3 // 4))),
                        (int(rng.integers(W // 12, W // 5)), max(2, int(6 * sc))), float(rng.integers(-10, 10)),
                        0, 360, 1.0, -1, AA)
        sm = cv2.GaussianBlur(sm, (0, 0), max(2.0 * sc, 1.2))[..., None] * 0.35
        p = p * (1 - sm) + np.array(T.SMUDGE, np.float32) * sm
        cell, lt = max(int(round(32 * sc)), 16), max(1, int(round(sc)))
        # cuadrícula de cuaderno centrada (mismo margen a ambos lados), línea mayor cada 5 contando desde el centro
        xs = range(fw + (W - 2 * fw - lt) % cell // 2, W - fw - lt + 1, cell)
        ys = range(fw + (H - 2 * fw - lt) % cell // 2, H - fw - lt + 1, cell)
        for k, x in enumerate(xs):
            p[fw:H - fw, x:x + lt] = T.GRID_MAJOR if (k - len(xs) // 2) % 5 == 0 else T.GRID
        for k, y in enumerate(ys):
            p[y:y + lt, fw:W - fw] = T.GRID_MAJOR if (k - len(ys) // 2) % 5 == 0 else T.GRID
        p += rng.normal(0, 1.4, (H, W, 1)).astype(np.float32)    # grano de melamina
        fi = int(round(fw * 0.55))
        p[:fw], p[H - fw:], p[:, :fw], p[:, W - fw:] = T.FRAME_OUTER, T.FRAME_OUTER, T.FRAME_OUTER, T.FRAME_OUTER
        p[fi:fw, fi:W - fi] = T.FRAME_INNER
        p[H - fw:H - fi, fi:W - fi] = T.FRAME_INNER
        p[fi:H - fi, fi:fw] = T.FRAME_INNER
        p[fi:H - fi, W - fw:W - fi] = T.FRAME_INNER
        self._paper = np.clip(p, 0, 255).astype(np.uint8)

    def _build_tray(self, tx, tw_, led_top, lip_top, floor_y):
        S = lambda v: int(round(v * self.sc))
        x0, x1 = tx, tx + tw_
        ab = _Art(x0 - S(10), led_top - S(10), tw_ + S(20), lip_top - led_top + S(18))
        ab.rrect(T.INK, (x0 + S(6), led_top - S(3), x1 - S(6), led_top + S(2)), S(2), op=0.20, blur=S(3))   # sombra sobre la pared
        ab.vshaded(_rr(x0, led_top, x1, lip_top + S(8), S(6), f=True), led_top, lip_top,
                   [(0, (214, 210, 206)), (0.35, (236, 233, 230)), (1, (222, 218, 214))])   # canal -> borde
        ab.rrect((248, 247, 246), (x0 + S(2), led_top, x1 - S(2), led_top + S(2)), S(1))          # canto trasero
        ab.rrect(T.INK, (x0, led_top, x0 + S(3), lip_top), S(1), op=0.10)                        # topes laterales
        ab.rrect(T.INK, (x1 - S(3), led_top, x1, lip_top), S(1), op=0.10)
        self._tray_back = ab.sprite()
        af = _Art(x0 - S(8), lip_top, tw_ + S(16), floor_y - lip_top)
        af.vshaded(_rr(x0 - S(8), lip_top, x1 + S(8), floor_y, S(6), f=True), lip_top, floor_y,
                   [(0, (236, 233, 230)), (0.12, (214, 210, 206)), (1, (176, 171, 167))])
        af.rrect((252, 251, 250), (x0 - S(6), lip_top + S(1), x1 + S(6), lip_top + S(3)), S(1), op=0.9)  # brillo del canto
        af.rrect(T.INK, (x0 - S(6), floor_y - S(3), x1 + S(6), floor_y), S(1), op=0.25)
        self._tray_front = af.sprite()

    def _marker_layer(self, color, active):
        """Rotulador de pizarra: cilindro con luz izquierda, etiqueta con rayas, capuchón con clip y
        brillo, o punta de fieltro biselada con tinta si está destapado. Se rota 6 grados al final."""
        S = lambda v: v * self.sc
        w, Hm, capH = S(34), S(140), S(50)
        Wl, Hl = int(w + S(44)), int(Hm + S(26))
        cx, base = Wl / 2, Hl - S(4)
        top = base - Hm
        xl, xr = cx - w / 2, cx + w / 2
        a = _Art(0, 0, Wl, Hl)
        body_y0 = top + capH - S(4) if not active else top + capH + S(8)
        ink = tuple(int(c * 0.78) for c in color)
        a.shaded((238, 236, 234), [(xl, body_y0), (xr, body_y0), (xr, base), (xl, base)], xl, xr)
        ly0, ly1 = body_y0 + S(14), body_y0 + S(54)
        a.shaded(color, [(xl, ly0), (xr, ly0), (xr, ly1), (xl, ly1)], xl, xr)
        for y in (ly0 - S(4), ly1 + S(1)):
            a.shaded((250, 250, 250), [(xl, y), (xr, y), (xr, y + S(3)), (xl, y + S(3))], xl, xr)
        a.rrect((250, 250, 250), (cx - S(9), (ly0 + ly1) / 2 - S(4), cx + S(9), (ly0 + ly1) / 2 + S(4)), S(4), op=0.9)
        a.rrect(ink, (cx - S(6), (ly0 + ly1) / 2 - S(1), cx + S(6), (ly0 + ly1) / 2 + S(1)), S(1), op=0.8)
        a.rrect((255, 255, 255), (xl + S(7), ly1 + S(8), xl + S(11), base - S(6)), S(2), op=0.55, blur=S(0.8))
        a.poly(T.INK, [(xl, body_y0), (xr, body_y0), (xr, base), (xl, base)], th=1, op=0.22)
        if active:   # destapado: collar, punta de fieltro biselada y tinta
            cy0 = body_y0 - S(10)
            a.shaded((200, 198, 196), [(xl + S(5), body_y0), (xr - S(5), body_y0), (xr - S(8), cy0), (xl + S(8), cy0)], xl, xr)
            ty = cy0 - S(24)
            a.shaded(ink, [(cx - w * .24, cy0), (cx + w * .24, cy0), (cx + w * .17, ty + S(6)), (cx + w * .06, ty),
                           (cx - w * .17, ty + S(2))], cx - w * .24, cx + w * .24)
            a.circle(color, (cx - w * .04, ty + S(3)), S(2.6))
            a.circle((255, 255, 255), (cx - w * .1, ty + S(8)), S(1.6), op=0.7)
        else:
            rim = tuple(int(c * 0.62) for c in color)
            capbox = _rr(xl - S(2), top, xr + S(2), top + capH, S(9), f=True)
            a.shaded(color, capbox, xl - S(2), xr + S(2))
            a.shaded(rim, [(xl - S(2), top + capH - S(7)), (xr + S(2), top + capH - S(7)),
                           (xr + S(2), top + capH - S(2)), (xl - S(2), top + capH - S(2))], xl - S(2), xr + S(2))
            a.rrect(color, (xr + S(1), top + S(10), xr + S(7), top + S(36)), S(3))                  # clip
            a.rrect(T.INK, (xr + S(1), top + S(10), xr + S(7), top + S(36)), S(3), th=1, op=0.25)
            a.rrect((255, 255, 255), (xl + S(5), top + S(8), xl + S(9), top + capH - S(14)), S(2), op=0.55, blur=S(0.8))
            a.poly(T.INK, capbox, th=1, op=0.25)
        pre, al = a.rotated(-6, (cx, base))
        return np.clip(pre, 0, 255), np.clip(al, 0, 1), cx, base

    def _eraser_layer(self, ew):
        """Borrador de pizarra: lomo/asa de plástico con agarre, base de plástico y cuerpo de fieltro texturizado."""
        S = lambda v: v * self.sc
        w, h = ew, S(60)
        a = _Art(0, 0, int(w + S(8)), int(h + S(8)))
        x0, y0, x1, y1 = S(4), S(4), S(4) + w, S(4) + h
        a.vshaded(_rr(x0, y0, x1, y0 + S(26), S(11), f=True), y0, y0 + S(26),
                  [(0, (240, 236, 232)), (0.5, (222, 214, 207)), (1, (190, 182, 175))])
        for k in range(5):
            gx = x0 + w * (0.30 + 0.10 * k)
            a.rrect(T.INK, (gx, y0 + S(6), gx + S(2), y0 + S(16)), S(1), op=0.22)
        a.rrect((255, 255, 255), (x0 + S(10), y0 + S(3), x1 - S(10), y0 + S(5)), S(1), op=0.7, blur=S(0.5))
        a.vshaded([(x0 + S(2), y0 + S(24)), (x1 - S(2), y0 + S(24)), (x1 - S(2), y0 + S(32)), (x0 + S(2), y0 + S(32))],
                  y0 + S(24), y0 + S(32), [(0, (176, 168, 160)), (1, (128, 120, 114))])
        rng = np.random.default_rng(11)
        noise = rng.normal(0, 26, (a.a.shape[0], a.a.shape[1], 1)).astype(np.float32)
        felt = np.array(T.FELT, np.float32) * 1.15
        fp = a._p(_rr(x0 + S(1), y0 + S(31), x1 - S(1), y1, S(8), f=True))
        a.paint(lambda xa, ya, xb, yb: np.clip(felt + noise[ya:yb, xa:xb], 0, 255),
                lambda m: cv2.fillPoly(m, [fp], 255, AA))
        a.rrect(T.INK, (x0 + S(1), y1 - S(5), x1 - S(1), y1), S(4), op=0.30)
        a.rrect(T.INK, (x0, y0, x1, y1), S(11), th=1, op=0.20)
        return np.clip(a.pre, 0, 255), np.clip(a.a, 0, 1), (x0 + x1) / 2, y1

    def _make_shape_pointer(self, col, closed):
        sc = self.sc
        S = round(96 * sc) | 1
        c = S // 2
        th = max(2, round(2 * sc))
        art = _Art(0, 0, S, S)
        r = lambda v: v * sc
        P = (c, c)
        col_bgr = T.COLORS[col]

        def shape(color, g, op=1.0):
            if closed:    # se contrae y se rellena con la tinta
                art.circle(color, P, r(11), th=th + g, op=op)
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    art.stroke(color, [(c + dx * r(14), c + dy * r(14)), (c + dx * r(20), c + dy * r(20))], th + g, op)
            else:
                art.circle(color, P, r(17), th=th + g, op=op)
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    art.stroke(color, [(c + dx * r(22), c + dy * r(22)), (c + dx * r(32), c + dy * r(32))], th + g, op)

        art.circle(T.INK, P, r(26), op=0.07)
        shape(T.PAPER_WHITE, 4, 0.9)
        shape(T.INK, 0)
        if closed:
            art.circle(col_bgr, P, r(9))
        else:
            art.circle(T.PAPER_WHITE, P, r(7))
            art.circle(col_bgr, P, r(5))
        return art.sprite()

    def _make_pointer(self, mode, col):
        sc = self.sc
        S = round(96 * sc) | 1
        c = S // 2 + (K - 1) / (2 * K)   # centro del píxel S//2 en el lienzo supersampleado: puntero centrado en la coordenada exacta
        th = max(2, round(2 * sc))
        art = _Art(0, 0, S, S, K)
        r = lambda v: v * sc
        P = (c, c)

        def shape(color, g, op=1.0):
            if mode == "DIBUJANDO":
                art.circle(color, P, r(16), th=th + g, op=op)
            elif mode == "SELECCIÓN":
                art.circle(color, P, r(17), th=th + g, op=op)
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    art.stroke(color, [(c + dx * r(22), c + dy * r(22)), (c + dx * r(32), c + dy * r(32))], th + g, op)
                art.circle(color, P, r(2) + g / 2, op=op)
            else:
                art.circle(color, P, r(12), th=th + g, op=op)

        art.circle(T.INK, P, r(26), op=0.07)                                  # halo
        shape(T.PAPER_WHITE, 4, 0.9)                                          # contorno claro
        shape(T.INK_DIM if mode == "EN ESPERA" else T.INK, 0)
        if mode == "DIBUJANDO":   # la punta muestra la tinta activa
            art.circle(T.PAPER_WHITE, P, r(8))
            art.circle(T.COLORS[col], P, r(6))
        return art.sprite()

    # -------------------------------------------------- cota de dos manos
    def _build_measure(self, fs, fth, cap):
        S = lambda v: int(round(v * self.sc))
        th = fth + 1
        self._mg = {}
        for ch in "0123456789":
            adv = _tw(ch, fs, th)
            a = _Art(0, 0, adv + 4, cap + 8)
            a.text(T.INK, ch, (2, cap + 2), fs, th)
            self._mg[ch] = (a.sprite(), adv)
        pxw = _tw("px", fs, fth)
        a = _Art(0, 0, pxw + 4, cap + 8 + S(6))
        a.text(T.INK_DIM, "px", (2, cap + 2), fs, fth)
        self._mpx = (a.sprite(), pxw)
        self._mdig = max(adv for _, adv in self._mg.values()) + S(1)
        self._mpad, self._mpady, self._mcap = S(16), S(10), cap
        self._mpill = {}
        for n in range(1, 6):
            w = 2 * self._mpad + n * self._mdig + S(8) + pxw
            h = cap + 2 * self._mpady
            a = _Art(0, 0, w, h)
            a.rrect(T.PAPER_WHITE, (0, 0, w, h), S(14), op=0.97)
            a.rrect(T.INK, (0, 0, w, h), S(14), th=max(1, S(2)))
            self._mpill[n] = a.sprite()

    def _measure(self, frame, m):
        S = lambda v: int(round(v * self.sc))
        p = [(int(m[0][0]), int(m[0][1])), (int(m[1][0]), int(m[1][1]))]
        d = np.array([p[1][0] - p[0][0], p[1][1] - p[0][1]], float)
        L = float(np.hypot(*d))
        u = d / L if L > 1 else np.array([1.0, 0.0])
        n = np.array([-u[1], u[0]])
        th = max(2, S(2))
        t = S(16)
        cv2.line(frame, p[0], p[1], T.PAPER_WHITE, th + 3, AA)
        cv2.line(frame, p[0], p[1], T.ACCENT, th, AA)
        for q in p:
            a = (int(round(q[0] + n[0] * t)), int(round(q[1] + n[1] * t)))
            b = (int(round(q[0] - n[0] * t)), int(round(q[1] - n[1] * t)))
            cv2.line(frame, a, b, T.PAPER_WHITE, th + 3, AA)
            cv2.circle(frame, q, S(9), T.PAPER_WHITE, th + 3, AA)
            cv2.line(frame, a, b, T.INK, th, AA)
            cv2.circle(frame, q, S(9), T.INK, th, AA)
            cv2.circle(frame, q, S(3), T.ACCENT, -1, AA)
        txt = str(int(round(L)))[:5]
        pill = self._mpill[len(txt)]
        ph, pw = pill.pre.shape[:2]
        mx, my = (p[0][0] + p[1][0]) // 2, (p[0][1] + p[1][1]) // 2
        pill.x = min(max(mx - pw // 2, 0), self.W - pw)
        pill.y = min(max(my - ph // 2, 0), self.H - ph)
        _blit(frame, pill)
        x = pill.x + self._mpad
        for ch in txt:
            sp, adv = self._mg[ch]
            sp.x, sp.y = x + (self._mdig - adv) // 2 - 2, pill.y + self._mpady - 2
            _blit(frame, sp)
            x += self._mdig
        sp, _ = self._mpx
        sp.x, sp.y = x + S(8) - 2, pill.y + self._mpady - 2
        _blit(frame, sp)

    # -------------------------------------------------- guía de uso
    def draw_help(self, frame) -> None:
        """Cartel de ayuda sobre la pizarra (llamar tras draw_board; sustituye al HUD)."""
        for sp in self._help:
            _blit(frame, sp)

    def _build_help(self):
        sc, W, H, fw = self.sc, self.W, self.H, self._fw
        S = lambda v: int(round(v * sc))
        fs, fth = self._fs, self._fth
        cap = _cap(fs)
        x0, y0, x1, y1 = fw + S(T.MARGIN), fw + S(T.MARGIN), W - fw - S(T.MARGIN), H - fw - S(T.MARGIN)
        self._help = []
        bd = _Art(x0 - S(24), y0 - S(24), x1 - x0 + S(48), y1 - y0 + S(60), k=1)
        bd.rrect(T.INK, (x0 + S(4), y0 + S(10), x1 - S(4), y1 + S(12)), S(20), op=0.22, blur=S(8))   # sombra del cartel
        bd.rrect((253, 252, 250), (x0, y0, x1, y1), S(20), op=0.96)
        bd.rrect(T.INK_DIM, (x0, y0, x1, y1), S(20), th=max(1, S(2)), op=0.8)
        for cx_, cy_ in ((x0 + S(22), y0 + S(22)), (x1 - S(22), y0 + S(22))):                         # chinchetas = imanes
            bd.circle(T.MAGNET_EDGE, (cx_, cy_ + S(2)), S(7))
            bd.circle(T.MAGNET_FACE, (cx_, cy_), S(7))
            bd.circle(T.MAGNET_RIM, (cx_, cy_), S(7), th=max(1, S(1.5)))
        self._help.append(bd.sprite())

        pad = S(40)
        ts, tth = 1.9 * max(sc, 0.75), max(2, round(2.5 * sc))
        ta = _Art(x0 + pad, y0 + S(18), S(640), S(70))
        ta.text(T.INK, "Cómo usar la pizarra", (x0 + pad, y0 + S(18) + S(52)), ts, tth, SCRIPT)
        self._help.append(ta.sprite())
        fy, kb = y1 - S(58), S(36)
        fa = _Art(x0 + pad, fy - S(4), x1 - x0 - 2 * pad, kb + S(10))
        kx = x0 + pad
        for key, lab in (("H", "Guía"), ("F", "Pantalla completa"), ("Q", "Salir")):
            fa.rrect(T.MAGNET_EDGE, (kx, fy + S(3), kx + kb, fy + kb + S(3)), S(8))
            fa.rrect(T.MAGNET_FACE, (kx, fy, kx + kb, fy + kb), S(8))
            fa.rrect(T.MAGNET_RIM, (kx, fy, kx + kb, fy + kb), S(8), th=max(1, S(2)))
            fa.text(T.INK, key, (kx + (kb - _tw(key, fs, fth + 1)) // 2, fy + kb // 2 + cap // 2), fs, fth + 1)
            fa.text(T.INK, lab, (kx + kb + S(10), fy + kb // 2 + cap // 2), fs * 0.95, fth)
            kx += kb + S(10) + _tw(lab, fs * 0.95, fth) + S(28)
        ss, sth = 1.0 * max(sc, 0.75), max(2, round(1.5 * sc))
        msg = "Pulsa H para cerrar"
        fa.text(T.INK_DIM, msg, (x1 - pad - _tw(msg, ss, sth, SCRIPT), fy + kb // 2 + cap // 2), ss, sth, SCRIPT)
        self._help.append(fa.sprite())

        gx0, gy0 = x0 + pad, y0 + S(104)
        gx1, gy1 = x1 - pad, y1 - S(64)
        cw, ch = (gx1 - gx0) // 3, (gy1 - gy0) // 2
        items = [
            ("Seleccionar", "Índice y medio arriba. Mueve el puntero sin dibujar.", "select"),
            ("Dibujar", "Pincel: solo el índice. Figuras: mueve el índice y toca el pulgar para dibujar; suelta para terminar.", "draw"),
            ("Línea con dos manos", "Con la herramienta Línea: un dedo en cada mano mide la distancia.", "line"),
            ("Confirmar", "Mantén 1 s sobre un botón.", "dwell"),
            ("Dos manos, dos colores", "Un dedo en cada mano: cada una dibuja a la vez con su color.", "two"),
        ]
        for i, (title, cap_txt, kind) in enumerate(items):
            cx0, cy0 = gx0 + (i % 3) * cw, gy0 + (i // 3) * ch
            a = _Art(cx0, cy0, cw, ch)
            ih = min(ch - S(16), S(170))
            iw = int(ih * 0.92)
            lh = cap + S(10)
            tx = cx0 + iw + S(16)
            maxw = cw - iw - S(44)
            tl = _wrap(title, fs * 1.05, fth + 1, maxw)
            cl = _wrap(cap_txt, fs * 0.95, fth, maxw) if cap_txt else []
            if True:
                self._help_art(a, kind, cx0 + iw // 2, cy0, iw, ih if ih < ch else ch)
                total = len(tl) * lh + S(6) + len(cl) * lh
                ty = cy0 + max(0, (ch - total) // 2) + cap
                for ln in tl:
                    a.text(T.INK, ln, (tx, ty), fs * 1.05, fth + 1)
                    ty += lh
                ty += S(6)
                for ln in cl:
                    a.text(T.INK, ln, (tx, ty), fs * 0.95, fth)
                    ty += lh
            self._help.append(a.sprite())

    def _help_art(self, art, kind, cx, cy, iw, ih):
        sc = self.sc
        et = max(1.5, 2 * sc)
        u = ih / 7.4
        ty = cy + ih * 0.52
        if kind == "select":
            tips = _hand(art, cx, ty, u, {"i", "m"}, et)
            for p in tips:
                art.circle(T.ACCENT, p, 0.42 * u)
        elif kind == "draw":   # pellizco: pulgar e índice juntos, dibujando un rectángulo
            tips = _hand(art, cx + 1.0 * u, ty, u, set(), et)
            x0_, y0_ = cx + 1.0 * u, ty
            fill = (253, 252, 250)
            w_ = 0.92 * u
            ip = [(x0_ - 1.5 * u, y0_ + 0.3 * u), (x0_ - 1.5 * u, y0_ - 1.9 * u), (x0_ - 2.7 * u, y0_ - 1.2 * u)]
            tp = [(x0_ - 2.0 * u, y0_ + 2.4 * u), (x0_ - 2.8 * u, y0_ - 0.1 * u)]
            for pa, pb in ((tp[0], tp[1]), (ip[0], ip[1]), (ip[1], ip[2])):
                _capsule(art, pa, pb, w_ + 2 * et, T.INK)
            for pa, pb in ((tp[0], tp[1]), (ip[0], ip[1]), (ip[1], ip[2])):
                _capsule(art, pa, pb, w_, fill)
            art.circle(T.COLORS["red"], (x0_ - 2.75 * u, y0_ - 0.65 * u), 0.5 * u)
            rx, ry = x0_ - 1.4 * u, y0_ - 3.5 * u
            art.stroke(T.COLORS["red"], [(rx, ry + 1.4 * u), (rx, ry), (rx + 2.6 * u, ry), (rx + 2.6 * u, ry + 1.0 * u)], max(2, 3 * sc))
        elif kind == "line":
            uu = u * 0.5
            t1 = _hand(art, cx - 2.1 * u, ty + 0.9 * uu, uu, {"i"}, et)
            t2 = _hand(art, cx + 2.1 * u, ty + 0.9 * uu, uu, {"i"}, et)
            a_, b_ = t1[0], t2[0]
            art.stroke(T.ACCENT, [a_, b_], max(2, 2.5 * sc))
            for p in (a_, b_):
                art.circle(T.ACCENT, p, 0.34 * u)
                art.stroke(T.INK, [(p[0], p[1] - 0.7 * u), (p[0], p[1] + 0.7 * u)], max(1.5, 2 * sc))
            mx, my = (a_[0] + b_[0]) / 2, (a_[1] + b_[1]) / 2
            lab = "412 px"
            fsz = self._fs * 0.85
            lw_ = _tw(lab, fsz, self._fth)
            art.rrect(T.PAPER_WHITE, (mx - lw_ / 2 - 8 * sc, my - 14 * sc, mx + lw_ / 2 + 8 * sc, my + 12 * sc), 8 * sc)
            art.rrect(T.INK, (mx - lw_ / 2 - 8 * sc, my - 14 * sc, mx + lw_ / 2 + 8 * sc, my + 12 * sc), 8 * sc, th=max(1, 1.5 * sc))
            art.text(T.INK, lab, (mx - lw_ / 2, my + 7 * sc), fsz, self._fth)
        elif kind == "two":
            uu = u * 0.5
            t1 = _hand(art, cx - 2.1 * u, ty + 0.9 * uu, uu, {"i"}, et)
            t2 = _hand(art, cx + 2.1 * u, ty + 0.9 * uu, uu, {"i"}, et)
            for tp, col, sgn in ((t1[0], "red", -1), (t2[0], "blue", 1)):
                art.stroke(T.COLORS[col], [(tp[0] + 0.35 * u * np.sin(k * 1.3) * sgn, tp[1] - 0.8 * u - 0.45 * u * k) for k in range(5)],
                           max(2, 3 * sc))
                art.circle(T.COLORS[col], tp, 0.42 * u)
        elif kind == "dwell":
            u2 = 0.72 * u
            tips = _hand(art, cx + 0.4 * u2, cy + 6.6 * u2, u2, {"i"}, et)
            p = (tips[0][0], tips[0][1] - 1.4 * u2)
            R = 2.1 * u2
            art.circle(T.MAGNET_EDGE, (p[0], p[1] + 3 * sc), R * 0.8)
            art.circle(T.MAGNET_FACE, p, R * 0.8)
            art.circle(T.MAGNET_RIM, p, R * 0.8, th=max(1, 2 * sc))
            a0 = np.radians(np.linspace(-90, 180, 40))
            art.stroke(T.ACCENT, [(p[0] + R * np.cos(t_), p[1] + R * np.sin(t_)) for t_ in a0], max(3, 4 * sc))
            art.circle(T.ACCENT, tips[0], 0.42 * u2)


    # -------------------------------------------------- API
    def draw_board(self, frame) -> None:
        """Pinta la pizarra pura (papel cuadriculado + marco). Ignora el contenido del frame."""
        np.copyto(frame, self._paper)

    def draw_hands(self, frame, hands, mode=None, colors=None, pinches=None) -> None:
        """Esqueleto de cada mano: huesos de tinta con contorno claro, articulaciones, punta del índice en azul."""
        th = self._bone_th
        for hi, lm in enumerate(hands):
            if len(lm) < 21:
                continue
            P = np.asarray(lm, np.int32)
            palm = P[_PALM]
            x0, y0 = np.maximum(palm.min(0) - 2, 0)
            x1, y1 = np.minimum(palm.max(0) + 3, [self.W, self.H])
            if x1 > x0 and y1 > y0:   # halo muy leve de la palma
                roi = frame[y0:y1, x0:x1]
                ov = roi.copy()
                cv2.fillConvexPoly(ov, cv2.convexHull(palm) - [x0, y0], T.ACCENT, AA)
                cv2.addWeighted(ov, 0.10, roi, 0.90, 0, dst=roi)
            segs = [P[list(c)] for c in _HAND_BONES]
            cv2.polylines(frame, segs, False, T.PAPER_WHITE, th + 3, AA)
            cv2.polylines(frame, segs, False, T.INK, th, AA)
            pin = bool(pinches and hi < len(pinches) and pinches[hi])
            for i in range(21):
                if i == 8 or (pin and i == 4):
                    continue
                r = self._joint_r + (1 if i in (4, 12, 16, 20) else 0)
                c = (int(P[i][0]), int(P[i][1]))
                cv2.circle(frame, c, r + 2, T.PAPER_WHITE, -1, AA)
                cv2.circle(frame, c, r, T.INK, -1, AA)
            c = (int(P[8][0]), int(P[8][1]))
            tip = T.COLORS.get(colors[hi], T.ACCENT) if colors and hi < len(colors) else T.ACCENT
            if pin:   # pellizco: pulgar e índice se unen con la tinta de la mano
                c4 = (int(P[4][0]), int(P[4][1]))
                cv2.line(frame, c4, c, T.PAPER_WHITE, th + 5, AA)
                cv2.line(frame, c4, c, tip, th + 1, AA)
                cv2.circle(frame, c4, self._joint_r + 5, T.PAPER_WHITE, -1, AA)
                cv2.circle(frame, c4, self._joint_r + 4, T.INK, -1, AA)
                cv2.circle(frame, c4, self._joint_r + 2, tip, -1, AA)
            cv2.circle(frame, c, self._joint_r + 7, T.PAPER_WHITE, -1, AA)
            cv2.circle(frame, c, self._joint_r + 6, T.INK, -1, AA)
            cv2.circle(frame, c, self._joint_r + 4, tip, -1, AA)

    def hit_test(self, x: int, y: int) -> str | None:
        for (x0, y0, x1, y1), bid in self._hit:
            if x0 <= x < x1 and y0 <= y < y1:
                return bid
        return None

    def draw(self, frame, state: UIState) -> None:
        now = time.perf_counter()
        if state.hands:
            self.draw_hands(frame, state.hands, state.mode, state.hand_colors or [state.color] * len(state.hands),
                            state.pinches or [state.pinch])
        _blit(frame, self._tray_back)
        hcols = list(state.hand_colors)
        active = {f"color:{c}" for c in (hcols or [state.color])} | {f"tool:{state.tool}"}
        for bid in BUTTON_IDS:
            sp = self._act.get(bid) if bid in active and bid in self._act else self._idle.get(bid)
            if sp:
                _blit(frame, sp)
        _blit(frame, self._tray_front)
        if len(hcols) >= 2:   # insignia 1/2 sobre el marcador que usa cada mano (apiladas si comparten color)
            seen = {}
            for hi, c in enumerate(hcols[:2]):
                xc = self._mx.get(f"color:{c}")
                if xc is None:
                    continue
                k = seen.get(c, 0)
                seen[c] = k + 1
                sp = self._badge[str(hi + 1)]
                sp.x = xc + int(round(15 * self.sc)) - self._br - 2
                sp.y = self._mtop - k * (2 * self._br + int(round(3 * self.sc))) - self._br - 2
                _blit(frame, sp)
        hv = state.hover_id if state.hover_id in self._el else None
        if hv:
            _blit(frame, self._hover[hv])
            if hv in self._tip:
                _blit(frame, self._tip[hv])

        _blit(frame, self._note.get(state.mode) or self._note[T.DEFAULT_MODE])
        self._num(frame, f"{max(0, int(round(state.fps)))}", self._fps_xy)
        self._num(frame, f"{int(state.thickness)} px", self._th_xy)
        dots = (hcols or [state.color])[:2]
        for i, c in enumerate(dots):
            x = self._dot_x - (len(dots) - 1 - i) * self._dot_dx
            cv2.circle(frame, (x, self._dot_y), self._dot_r + 3, T.PAPER_WHITE, -1, AA)
            cv2.circle(frame, (x, self._dot_y), self._dot_r + 2, T.INK, -1, AA)
            cv2.circle(frame, (x, self._dot_y), self._dot_r, T.COLORS.get(c, T.INK), -1, AA)

        self._dwell(frame, state, hv, now)
        if state.measure:
            self._measure(frame, state.measure)

        ptrs = state.pointers if state.pointers else ([state.pointer] if state.pointer is not None else [])
        cols = hcols or [state.color]
        shape_tool = state.tool in ("line", "rect", "circle") and state.mode == "DIBUJANDO"
        pins = list(state.pinches) if state.pinches else [state.pinch]
        if shape_tool and not any(pins):
            _blit(frame, self._pinch_hint)
        for i, pt in enumerate(ptrs):
            if pt is None:
                continue
            col = cols[i] if i < len(cols) else cols[-1]
            if col not in T.COLORS:
                col = "red"
            h = self._pin[(False, col)].pre.shape[0] // 2
            if shape_tool:
                pin_i = bool(pins[i]) if i < len(pins) else False
                if pin_i and not self._pprev.get(i):
                    self._ppulse[i] = (now, int(pt[0]), int(pt[1]), col)
                self._pprev[i] = pin_i
                p = self._pin[(pin_i, col)]
                p.x, p.y = int(pt[0]) - h, int(pt[1]) - h
                _blit(frame, p)
                if not pin_i:
                    sp = self._tool_hint[state.tool]
                    sp.x, sp.y = int(pt[0]) + int(round(22 * self.sc)), int(pt[1]) + int(round(16 * self.sc))
                    _blit(frame, sp)
                continue
            self._pprev[i] = False
            key = (state.mode, col if state.mode == "DIBUJANDO" else None)
            p = self._ptr.get(key) or self._ptr.get((state.mode, "red")) or self._ptr[(T.DEFAULT_MODE, None)]
            p.x, p.y = int(pt[0]) - h, int(pt[1]) - h
            _blit(frame, p)
        for i in list(self._ppulse):   # anillo de pulso al activar el pellizco
            t0, px, py, col = self._ppulse[i]
            t = (now - t0) / 0.35
            if t >= 1:
                del self._ppulse[i]
                continue
            rad = int(round((12 + 24 * t) * self.sc))
            x0, y0 = max(px - rad - 4, 0), max(py - rad - 4, 0)
            x1, y1 = min(px + rad + 5, self.W), min(py + rad + 5, self.H)
            if x1 > x0 and y1 > y0:
                roi = frame[y0:y1, x0:x1]
                ov = roi.copy()
                cv2.circle(ov, (px - x0, py - y0), rad, T.COLORS[col], max(2, int(round(3 * self.sc))), AA)
                al = 0.9 * (1 - t)
                cv2.addWeighted(ov, al, roi, 1 - al, 0, dst=roi)

    def _num(self, frame, txt, xy):
        sp = self._numc.get(txt)
        cap = _cap(self._fs)
        if sp is None:
            if len(self._numc) > 256:
                self._numc.clear()
            a = _Art(0, 0, _tw(txt, self._fs, self._fth) + 4, cap + 8)
            a.text(T.INK, txt, (2, cap + 2), self._fs, self._fth)
            sp = self._numc[txt] = a.sprite()
        sp.x, sp.y = xy[0] - 2, xy[1] - cap - 2
        _blit(frame, sp)

    # -------------------------------------------------- dwell
    def _center(self, bid):
        return self._el[bid]["center"]

    def _dwell(self, frame, st, hv, now):
        d = min(max(st.dwell, 0.0), 1.0) if hv else 0.0
        pb, pd = self._prev
        # confirmación: dwell llega a 1 o main lo reinicia justo tras confirmar
        if d >= 0.999 and not self._fired:
            self._fired, self._pulse = True, (now, hv)
        elif d < 0.5:
            if not self._fired and pb and pd > 0.95 and d < 0.3:
                self._pulse = (now, pb)
            self._fired = False
        self._prev = (hv, d)

        th = self._ring_th
        if hv and d > 0:
            pts = self._el[hv]["perim"]
            k = int(d * (len(pts) - 1))
            if k >= 1:
                cv2.polylines(frame, [pts[:k + 1]], d >= 0.999, T.ACCENT, th, AA)
        if self._pulse:
            t = (now - self._pulse[0]) / T.DWELL_PULSE_S
            bid = self._pulse[1]
            if t >= 1 or bid not in self._el:
                self._pulse = None
                return
            pts = self._el[bid]["perim"]
            c = np.array(self._el[bid]["center"])
            q = np.round((pts - c) * (1 + 0.25 * t) + c).astype(np.int32)
            x0, y0 = np.maximum(q.min(0) - th, 0)
            x1, y1 = np.minimum(q.max(0) + th + 1, [self.W, self.H])
            if x1 > x0 and y1 > y0:
                roi = frame[y0:y1, x0:x1]
                ov = roi.copy()
                cv2.polylines(ov, [q - [x0, y0]], True, T.ACCENT, th, AA)
                a = 0.9 * (1 - t)
                cv2.addWeighted(ov, a, roi, 1 - a, 0, dst=roi)


# ---------------------------------------------------------------- demo / bench
def _camera_bg(w, h):
    """Fondo de cámara simulado: pared, ventana luminosa, persona, mesa, ruido y viñeta."""
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    bg = np.zeros((h, w, 3), np.float32)
    bg[:] = np.array([120, 135, 150], np.float32) * (0.75 + 0.35 * (1 - yy / h))[..., None]
    k = lambda v, a: int(v * a)
    cv2.rectangle(bg, (k(w, .69), k(h, .08)), (k(w, .94), k(h, .58)), (230, 225, 210), -1)
    cv2.rectangle(bg, (0, k(h, .72)), (w, h), (70, 90, 120), -1)
    cv2.ellipse(bg, (k(w, .33), k(h, .78)), (k(w, .2), k(h, .46)), 0, 180, 360, (60, 70, 85), -1)
    cv2.circle(bg, (k(w, .33), k(h, .42)), k(h, .15), (130, 160, 205), -1)
    cv2.ellipse(bg, (k(w, .6), k(h, .65)), (k(w, .055), k(h, .17)), 20, 0, 360, (140, 170, 215), -1)
    bg = cv2.GaussianBlur(bg, (0, 0), 9 * w / 1280)
    bg *= (1 - 0.7 * (((xx - w / 2) / w) ** 2 + ((yy - h / 2) / h) ** 2))[..., None]
    bg += rng.normal(0, 4, bg.shape)
    return np.clip(bg, 0, 255).astype(np.uint8)


def _demo_strokes(img, ov):
    """Trazos de ejemplo con marcador para ver el contraste sobre el papel."""
    h, w = img.shape[:2]
    t = max(4, int(round(8 * ov.sc)))
    cols = T.COLORS

    def pl(pts, col, closed=False):
        cv2.polylines(img, [np.int32(pts)], closed, col, t, AA)

    x = np.linspace(w * .22, w * .50, 60)
    pl(np.c_[x, h * .34 + h * .06 * np.sin((x - x[0]) / w * 24)], cols["red"])
    pl([(w * .55, h * .24), (w * .72, h * .24), (w * .72, h * .44), (w * .55, h * .44)], cols["blue"], True)
    a = np.linspace(0, 2 * np.pi, 80)
    pl(np.c_[w * .36 + h * .12 * np.cos(a), h * .58 + h * .12 * np.sin(a)], cols["green"], True)
    pl([(w * .56, h * .56), (w * .70, h * .62), (w * .62, h * .70)], cols["yellow"])
    pl([(w * .22, h * .22), (w * .26, h * .26), (w * .30, h * .21), (w * .36, h * .27)], cols["black"])


def _parse_size(s):
    w, h = s.lower().split("x")
    return int(w), int(h)


def _bench(w, h, n=400):
    ov, cam = Overlay(w, h), _camera_bg(w, h)
    f = cam.copy()
    bid = "tool:rect"
    cx, cy = ov._center(bid)
    tb, td = [], []
    for i in range(n):
        np.copyto(f, cam)
        st = UIState("SELECCIÓN", 30.0, (cx, cy), bid, (i % 40) / 40, "red", "brush", 8,
                     measure=((w // 5 + i, h // 3), (w * 3 // 4, h // 2 + i % 50)))
        t0 = time.perf_counter()
        ov.draw_board(f)
        t1 = time.perf_counter()
        ov.draw(f, st)
        t2 = time.perf_counter()
        tb.append((t1 - t0) * 1000)
        td.append((t2 - t1) * 1000)
    tb, td = np.array(tb[20:]), np.array(td[20:])
    print(f"{w}x{h}: draw {td.mean():.2f} ms (p95 {np.percentile(td, 95):.2f}) | "
          f"draw_board {tb.mean():.2f} ms (p95 {np.percentile(tb, 95):.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--save", help="renderiza un frame a PNG y sale")
    ap.add_argument("--hover", default=None)
    ap.add_argument("--dwell", type=float, default=0.0)
    ap.add_argument("--mode", default="SELECCIÓN")
    a = ap.parse_args()
    w, h = _parse_size(a.size)
    if a.bench:
        _bench(1280, 720)
        _bench(1920, 1080)
        return
    ov = Overlay(w, h)
    scene = _camera_bg(w, h)
    ov.draw_board(scene)
    _demo_strokes(scene, ov)
    st = UIState(a.mode, 30.0, None, None, 0.0, "blue", "rect", 8)
    if a.save:
        f = scene.copy()
        if a.hover:
            st.hover_id, st.dwell, st.pointer = a.hover, a.dwell, ov._center(a.hover)
        else:
            st.pointer = (w // 2, h // 2)
        ov.draw(f, st)
        cv2.imwrite(a.save, f)
        return

    mouse, win = [w // 2, h // 2], "handCanva UI demo"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, lambda e, x, y, *_: mouse.__setitem__(slice(None), [x, y]))
    modes = list(T.MODES)
    last, dwell_t, prev_id, fps = time.perf_counter(), 0.0, None, 30.0
    print("Teclas 1-4: modo | q/Esc: salir | mantén el ratón sobre un objeto 1 s para confirmar")
    while True:
        now = time.perf_counter()
        dt, last = now - last, now
        fps = 0.9 * fps + 0.1 / max(dt, 1e-3)
        bid = ov.hit_test(*mouse) if st.mode == "SELECCIÓN" else None
        dwell_t = dwell_t + dt if bid and bid == prev_id else 0.0
        prev_id = bid
        st.pointer, st.hover_id, st.fps = tuple(mouse), bid, fps
        st.dwell = min(dwell_t, 1.0)
        if bid and dwell_t >= 1.0:
            kind, name = bid.split(":")
            if kind == "color":
                st.color = name
            elif kind == "tool":
                st.tool = name
            elif bid.startswith("thick"):
                st.thickness = max(2, min(40, st.thickness + (2 if name == "+" else -2)))
            dwell_t = 0.0   # esta iteración ya se dibuja con dwell=1.0 -> dispara el pulso
        f = scene.copy()
        ov.draw(f, st)
        cv2.imshow(win, f)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            break
        if ord("1") <= k <= ord("4"):
            st.mode = modes[k - ord("1")]
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
