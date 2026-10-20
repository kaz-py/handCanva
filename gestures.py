"""Landmarks de mano -> modo de gesto + puntero suavizado, y máquina de dwell.

classify/GestureTracker/Dwell son puros (sin cámara ni MediaPipe) para poder testearlos.
HandDetector usa la API Tasks de MediaPipe (mp.solutions ya no existe en versiones recientes).
"""
from __future__ import annotations

import os
import urllib.request

import numpy as np

DRAW, SELECT, ERASE, LINE, IDLE = "DRAW", "SELECT", "ERASE", "LINE", "IDLE"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "hand_landmarker.task")

_FINGERS = ((6, 8), (10, 12), (14, 16), (18, 20))  # (PIP, punta): índice, medio, anular, meñique
FULL = (0.0, 0.0, 1.0, 1.0)            # zona activa = imagen entera: (x*W, y*H)
ZONE = (0.10, 0.08, 0.90, 0.85)        # zona activa por defecto: la mano llega a bordes y bandeja sin salir del encuadre
HOLD_FRAMES = 2                        # frames sin detección que se toleran antes de soltar la mano


def _d(lm, a, b):
    return float(np.hypot(lm[a][0] - lm[b][0], lm[a][1] - lm[b][1]))


# ---- características por dedo (invariantes a rotación y escala) ------------------------------
# ext in 0..1 = media de tres señales: ratio punta/PIP a la muñeca, ángulo en el PIP y alcance de la
# punta relativo al tamaño de mano (muñeca-MCP medio). Extendido ~0.9, plegado ~0.1, curvado ~0.4-0.6.
EXT_UP, EXT_STAY_UP, EXT_DOWN, EXT_STAY_DOWN = 0.65, 0.42, 0.36, 0.55   # histéresis: entrar exige más que mantener
THUMB_FOLD, THUMB_OUT = 0.60, 0.45
ENTER_EXTRA = {DRAW: 0, IDLE: 0, SELECT: 2, ERASE: 3}   # frames extra: BORRAR y SELECCIÓN son los más difíciles de activar
COOLDOWN = 5        # frames (~150 ms) sin nuevos cambios tras un cambio de modo
AMBIG_MAX = 10      # frames ambiguos seguidos tras los que SELECCIÓN/BORRAR se sueltan (seguridad); DIBUJO aguanta 3x
EXT_ALPHA = 0.6     # suavizado de las características
PINCH_ON, PINCH_OFF, PINCH_FRAMES = 0.30, 0.45, 2   # dist(punta índice, punta pulgar)/tamaño de palma, con histéresis


def _clip01(v):
    return min(1.0, max(0.0, v))


def features(lm):
    """-> (ext[4], thumb_fold 0..1, ratios[4], angulos[4]). lm ya en espacio isótropo (x escalada por w/h)."""
    size = max(_d(lm, 0, 9), 1e-6)
    ext, rat, ang = [], [], []
    for mcp, pip, dip, tip in ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)):
        r = _d(lm, tip, 0) / max(_d(lm, pip, 0), 1e-6)
        u, v = np.asarray(lm[mcp][:2]) - lm[pip][:2], np.asarray(lm[dip][:2]) - lm[pip][:2]
        g = float(np.degrees(np.arccos(np.clip(u @ v / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-9), -1, 1))))
        reach = _d(lm, tip, 0) / size
        ext.append((_clip01((r - 0.8) / 0.7) + _clip01((g - 90) / 80) + _clip01((reach - 0.9) / 0.8)) / 3)
        rat.append(r)
        ang.append(g)
    # pulgar plegado sobre la palma: su punta queda más cerca del meñique (17) que su articulación IP (3)
    fold = _clip01(0.5 + (_d(lm, 3, 17) - _d(lm, 4, 17)) / (0.5 * size))
    return np.array(ext), fold, rat, ang


def _decide(ext, fold, prev, pinch=False):
    """Patrón de dedos -> DRAW/SELECT/ERASE/IDLE, o None si es ambiguo (el llamador conserva el modo)."""
    # índice+pulgar pellizcados con el resto plegado y el pulgar NO recogido en la palma (eso sería un puño) = mano de dibujo
    if pinch and fold < THUMB_FOLD and all(v < EXT_STAY_DOWN for v in ext[1:]):
        return DRAW
    up_idx = {DRAW: {0}, SELECT: {0, 1}}.get(prev, set())
    st = []
    for i, v in enumerate(ext):
        if i in up_idx:
            st.append("U" if v > EXT_STAY_UP else "D" if v < EXT_DOWN else "?")
        elif prev in (DRAW, SELECT, ERASE):
            st.append("D" if v < EXT_STAY_DOWN else "U" if v > EXT_UP + 0.1 else "?")
        else:
            st.append("U" if v > EXT_UP else "D" if v < EXT_DOWN else "?")
    pat = "".join(st)
    if pat == "UUDD":
        return SELECT
    if pat == "UDDD":
        return DRAW
    if pat == "DDDD":
        return ERASE if fold > THUMB_FOLD else IDLE if fold < THUMB_OUT else None
    return None if "?" in pat else IDLE


def classify(lm, prev=None) -> str:
    """Clasificación sin estado de una mano (lm isótropo); lo ambiguo cuenta como IDLE."""
    ext, fold, _, _ = features(lm)
    return _decide(ext, fold, prev) or IDLE


def pinch_dist(lm) -> float:
    return _d(lm, 4, 8) / max(_d(lm, 0, 9), 1e-6)


def is_pinch(lm) -> bool:
    return pinch_dist(lm) < PINCH_ON


def palm_center(lm):
    return lm[[5, 9, 13, 17], :2].mean(axis=0)


class _Slot:
    """Estado de una mano con identidad persistente (seguimiento por proximidad, nunca Left/Right)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.active, self.miss = False, 0
        self.mode, self.cand, self.count, self.cool, self.amb = IDLE, IDLE, 0, 0, 0
        self.ext = self.fold = self.sk = None
        self.wrist = self.pointer = None
        self.pinch, self.pdist, self._pcnt = False, 1.0, 0
        self.ratios = self.angles = None


class GestureTracker:
    """Hasta 2 manos independientes, cada una con su modo estable.

    Por mano: evidencia suavizada -> candidato; entrar a un modo activo exige N frames seguidos de evidencia
    fuerte (más para SELECCIÓN y BORRAR), salir exige evidencia contraria clara, lo ambiguo conserva el modo,
    y hay un cooldown tras cada cambio. `mode/pointer/pinch/ratios` son los de la mano primaria (slot activo
    más bajo); `mode` global = el de mayor prioridad. `hands` = esqueletos suavizados en px por mano activa."""

    PRIORITY = (ERASE, SELECT, DRAW, IDLE)

    def __init__(self, width: int, height: int, hysteresis: int = 4, alpha: float = 0.3, zone=FULL):
        self.w, self.h, self.n, self.alpha, self.zone = width, height, hysteresis, alpha, zone
        self.sc = min(width / 1280, height / 720)
        self.slots = [_Slot(), _Slot()]
        self.mode, self.pointer, self.pinch, self.ratios = IDLE, None, False, None
        self.hands = []

    def _px(self, p):
        # zona activa de la cámara -> pantalla entera (ganancia lineal); fuera de la zona, o landmarks
        # extrapolados fuera de [0,1] con la mano en el borde, se recortan al borde de la pantalla
        # coordenadas directas de la cámara, sin espejo ni inversión: x = u*W, y = v*H
        x0, y0, x1, y1 = self.zone
        u = (np.asarray(p, np.float64)[..., :2] - (x0, y0)) / (x1 - x0, y1 - y0)
        return np.clip(u * (self.w, self.h), 0.0, (self.w - 1.0, self.h - 1.0))   # un punto (2,) o una mano (21, 2)

    def _smooth(self, old, new):
        if old is None:
            return new
        # alfa adaptativo tipo One-Euro: quieto -> alpha (filtra temblor); rápido -> ~0.9 (casi sin retardo)
        v = float(np.abs(new - old).max()) / self.sc     # px/frame a escala 720p
        a = min(0.9, self.alpha + v / 40)
        return a * new + (1 - a) * old

    def _assign(self, hands):
        """Empareja detecciones con slots por proximidad de muñeca (voraz). -> ({slot: detección}, muñecas px)."""
        wr = [self._px(m[0]) for m in hands]
        act = [i for i, s in enumerate(self.slots) if s.active]
        pairs = sorted((float(np.hypot(*(wr[d] - self.slots[i].wrist))), d, i) for d in range(len(hands)) for i in act)
        out, used = {}, set()
        for _, d, i in pairs:
            if i not in out and d not in used:
                out[i] = d
                used.add(d)
        for d in range(len(hands)):
            if d not in used:
                free = [i for i, s in enumerate(self.slots) if not s.active and i not in out]
                if free:
                    out[free[0]] = d
        return out, wr

    def _step_mode(self, s):
        cand = _decide(s.ext, s.fold, s.mode, s.pinch)
        if s.cool > 0:
            s.cool -= 1
            s.cand, s.count = s.mode, 0
            return
        if cand is None:                               # ambiguo: conservar, pero SELECCIÓN/BORRAR caducan
            s.count, s.cand = 0, s.mode
            s.amb += 1
            if s.mode != IDLE and s.amb >= (AMBIG_MAX if s.mode in (ERASE, SELECT) else 3 * AMBIG_MAX):
                s.mode, s.amb, s.cool = IDLE, 0, COOLDOWN
            return
        s.amb = 0
        if cand == s.mode:
            s.cand, s.count = cand, 0
            return
        s.count = s.count + 1 if cand == s.cand else 1
        s.cand = cand
        if s.count >= self.n + ENTER_EXTRA[cand]:
            s.mode, s.count, s.cool = cand, 0, COOLDOWN

    def update(self, hands):
        if hands is None:
            hands = []
        elif not isinstance(hands, (list, tuple)):
            hands = [hands]
        hands = list(hands)[:2]
        asg, wr = self._assign(hands) if hands else ({}, [])
        for i, s in enumerate(self.slots):
            if not s.active and i not in asg:
                continue
            if i not in asg:                           # fallo suelto de detección: se conserva la mano
                s.miss += 1
                if s.miss > HOLD_FRAMES:
                    s.reset()
                continue
            lm, s.miss, s.active = hands[asg[i]], 0, True
            s.wrist = wr[asg[i]]
            iso = np.asarray(lm, np.float64)[:, :2] * (self.w / self.h, 1.0)   # distancias isótropas (rotación)
            ext, fold, s.ratios, s.angles = features(iso)
            s.ext = ext if s.ext is None else EXT_ALPHA * ext + (1 - EXT_ALPHA) * s.ext
            s.fold = fold if s.fold is None else EXT_ALPHA * fold + (1 - EXT_ALPHA) * s.fold
            s.pdist = pinch_dist(iso)             # pellizco con histéresis y confirmación (evita parpadeo en el umbral)
            want = s.pdist < PINCH_ON if not s.pinch else not s.pdist > PINCH_OFF
            s._pcnt = 0 if want == s.pinch else s._pcnt + 1
            if s._pcnt >= PINCH_FRAMES:
                s.pinch, s._pcnt = want, 0
            self._step_mode(s)
            # esqueleto en px con el MISMO mapeo y suavizado que el puntero: su punta del índice ES el puntero
            s.sk = self._smooth(s.sk, self._px(lm))
            tgt = s.sk[[5, 9, 13, 17]].mean(axis=0) if s.mode == ERASE else s.sk[8]
            s.pointer = (int(round(tgt[0])), int(round(tgt[1])))   # round: int() sesga -0.5 px
        act = [s for s in self.slots if s.active]
        self.hands = [[(int(round(x)), int(round(y))) for x, y in s.sk] for s in act]
        if not act:
            self.mode, self.pointer, self.pinch, self.ratios = IDLE, None, False, None
            return self.mode, None, False
        self.mode = min((s.mode for s in act), key=self.PRIORITY.index)
        self.pointer, self.pinch, self.ratios = act[0].pointer, act[0].pinch, act[0].ratios
        return self.mode, self.pointer, self.pinch


_NO = object()


class Dwell:
    """Progreso 0..1 sobre un objetivo; dispara UNA vez al llegar a `duration`.
    Para volver a disparar hay que cambiar de objetivo o soltarlo (None).
    grace: un cambio de objetivo (temblor al borde, 1 frame sin mano) debe durar `grace` s para reiniciar."""

    def __init__(self, duration: float, grace: float = 0.0):
        self.duration, self.grace = duration, grace
        self.target, self._t0, self._fired = None, 0.0, False
        self._cand, self._cand_t = _NO, 0.0

    def update(self, target, now: float):
        """-> (progreso, id disparado o None)."""
        if target == self.target:
            self._cand = _NO                      # sin candidato en espera (None también es candidato válido)
        elif self.target is not None and self.grace > 0:
            if target != self._cand:
                self._cand, self._cand_t = target, now
            if now - self._cand_t < self.grace:
                target = self.target
        if target != self.target:
            self.target, self._t0, self._fired = target, now, False
        if target is None:
            return 0.0, None
        if self._fired:
            return 1.0, None
        prog = min(1.0, (now - self._t0) / self.duration)
        if prog >= 1.0:
            self._fired = True
            return 1.0, target
        return prog, None


class HandDetector:
    """detect(frame_bgr, ts_ms) -> lista de 0..2 manos, cada una (21,3) normalizado."""

    def __init__(self, model_path: str = MODEL_PATH):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            try:
                urllib.request.urlretrieve(MODEL_URL, model_path)
            except Exception as e:
                if os.path.exists(model_path):
                    os.remove(model_path)
                raise RuntimeError(f"No se pudo descargar el modelo de manos ({e}). "
                                   f"Descárgalo de {MODEL_URL} a {model_path}") from e
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO, num_hands=2)
        self._lm = vision.HandLandmarker.create_from_options(opts)

    def detect(self, frame_bgr, ts_ms: int):
        import cv2
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        res = self._lm.detect_for_video(self._mp.Image(self._mp.ImageFormat.SRGB, rgb), ts_ms)
        return [np.array([[p.x, p.y, p.z] for p in h], dtype=np.float32) for h in res.hand_landmarks]
