"""Capa de dibujo BGR + máscara de tinta, herramientas, undo y mezcla con la cámara.

Admite varios trazos simultáneos (uno por id: mano 0, mano 1, línea de pareja = 2).
"""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np

AA = cv2.LINE_AA
SHAPES = ("line", "rect", "circle")


class _Stroke:
    __slots__ = ("tool", "color", "thick", "start", "last")

    def __init__(self, tool, color, thick, pt):
        self.tool, self.color, self.thick, self.start, self.last = tool, color, int(thick), pt, pt


class Canvas:
    def __init__(self, width: int, height: int, max_undo: int = 20):
        self.w, self.h = width, height
        self.layer = np.zeros((height, width, 3), np.uint8)
        self.mask = np.zeros((height, width), np.uint8)   # 255 donde hay tinta
        self._undo = deque(maxlen=max_undo)
        self._strokes: dict[int, _Stroke] = {}

    # ---- trazo
    def begin(self, tool, color_bgr, thickness, pt, sid=0):
        st = self._strokes[sid] = _Stroke(tool, color_bgr, thickness, pt)
        if tool in ("brush", "eraser"):
            self._snapshot()
            self._seg(st, pt, pt)

    def active(self, sid=0):
        return sid in self._strokes

    def set_start(self, pt, sid=0):
        """Mueve el origen de la forma en curso (línea de dos manos: ambos extremos se mueven)."""
        st = self._strokes.get(sid)
        if st and st.tool in SHAPES:
            st.start = pt

    def move(self, pt, sid=0):
        st = self._strokes.get(sid)
        if st is None:
            return
        if st.tool in ("brush", "eraser"):
            self._seg(st, st.last, pt)
        st.last = pt

    def end(self, sid=0):
        """Confirma formas (al salir de DIBUJO)."""
        st = self._strokes.pop(sid, None)
        if st and st.tool in SHAPES and st.last != st.start:
            self._snapshot()
            self._shape(self.layer, st, st.color, AA)
            self._shape(self.mask, st, 255, cv2.LINE_8)

    def cancel(self, sid=0):
        """Descarta una forma en curso sin dibujarla."""
        self._strokes.pop(sid, None)

    def _seg(self, st, a, b):
        if st.tool == "eraser":
            t = st.thick * 3
            cv2.line(self.layer, a, b, (0, 0, 0), t, AA)
            cv2.line(self.mask, a, b, 0, t, cv2.LINE_8)
        else:
            cv2.line(self.layer, a, b, st.color, st.thick, AA)
            cv2.line(self.mask, a, b, 255, st.thick, cv2.LINE_8)

    @staticmethod
    def _shape(img, st, col, lt):
        a, b, t = st.start, st.last, st.thick
        if st.tool == "line":
            cv2.line(img, a, b, col, t, lt)
        elif st.tool == "rect":
            cv2.rectangle(img, a, b, col, t)
        else:
            cv2.circle(img, a, int(np.hypot(b[0] - a[0], b[1] - a[1])), col, t, lt)

    # ---- historial
    def _snapshot(self):
        # shortcut: dos trazos simultáneos que se solapan en el tiempo se deshacen juntos; bastaría guardar tinta por trazo
        self._undo.append((self.layer.copy(), self.mask.copy()))

    def undo(self):
        if self._undo:
            lay, msk = self._undo.pop()
            self.layer[:] = lay
            self.mask[:] = msk

    def clear(self):
        if self.mask.any():
            self._snapshot()
        self.layer[:] = 0
        self.mask[:] = 0

    # ---- mezcla
    def blend(self, frame):
        """Pega la tinta sobre el frame in-place (copyTo con máscara, sin bucles Python)."""
        cv2.copyTo(self.layer, self.mask, frame)
        for st in self._strokes.values():
            if st.tool in SHAPES and st.last != st.start:
                self._shape(frame, st, st.color, AA)  # vista previa
