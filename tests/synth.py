"""Manos sintéticas con cinemática simple (3D ortográfica) para auditar la clasificación de gestos."""
import numpy as np

ASPECT = 16 / 9
_MCP = {5: (-0.30, 0.93), 9: (0.0, 1.0), 13: (0.28, 0.94), 17: (0.52, 0.80)}
_SPREAD = {5: -4, 9: 0, 13: 5, 17: 12}                   # grados de abanico en reposo
_LEN = {5: 0.95, 9: 1.0, 13: 0.95, 17: 0.78}
_FLEX = {"ext": ((0, 12), (0, 12), (0, 10)), "curl": ((70, 90), (85, 105), (40, 65)),
         "half": ((15, 25), (30, 45), (10, 20))}


def hand_model(fingers, thumb, rng, spread=0.0):
    """fingers: 4 estados 'ext'|'curl'|'half' (índice..meñique); thumb: 'out'|'fold'. -> (21,3) en unidades de mano."""
    lm = np.zeros((21, 3))
    for base, st in zip((5, 9, 13, 17), fingers):
        p = np.array([*_MCP[base], 0.0])
        lm[base] = p
        phi = np.radians(_SPREAD[base] + (spread * (-1 if base == 5 else 1 if base == 9 else 0)) + rng.uniform(-6, 6))
        th = 0.0
        for k, (lo, hi) in enumerate(_FLEX[st]):
            th += np.radians(rng.uniform(lo, hi))
            seg = (0.42, 0.26, 0.2)[k] * _LEN[base]
            p = p + seg * np.array([np.sin(phi) * np.cos(th), np.cos(phi) * np.cos(th), np.sin(th)])
            lm[base + 1 + k] = p
    lm[1], lm[2] = (-0.25, 0.2, 0), (-0.45, 0.45, 0)
    if thumb == "out":
        d1, d2 = np.array([-0.6, 0.8, 0.1]), np.array([-0.5, 0.85, 0.1])
    else:
        d1, d2 = np.array([0.5, 0.4, 0.5]), np.array([0.9, 0.2, 0.3])
    d1 = d1 + rng.uniform(-0.1, 0.1, 3); d2 = d2 + rng.uniform(-0.1, 0.1, 3)
    lm[3] = lm[2] + 0.3 * d1 / np.linalg.norm(d1)
    lm[4] = lm[3] + 0.25 * d2 / np.linalg.norm(d2)
    return lm


def place(model, rng, size, center, roll=0.0, pitch=0.0, yaw=0.0):
    """Rota (roll/pitch/yaw en grados), proyecta y devuelve landmarks normalizados (21,3) con z=0."""
    a, b, c = np.radians([pitch, yaw, roll])
    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    q = model @ (rz @ ry @ rx).T
    out = np.zeros((21, 3), np.float32)
    out[:, 0] = center[0] + size * q[:, 0] / ASPECT   # size = fracción de la altura de imagen
    out[:, 1] = center[1] - size * q[:, 1]
    return out


# nombre -> (dedos, pulgar, verdad)  verdad: DRAW/SELECT/IDLE/AMBIG
POSES = {
    "draw": (("ext", "curl", "curl", "curl"), None, "DRAW"),
    "select": (("ext", "ext", "curl", "curl"), None, "SELECT"),
    "fist": (("curl",) * 4, "fold", "IDLE"),
    "open": (("ext",) * 4, "out", "IDLE"),
    "three": (("ext", "ext", "ext", "curl"), None, "IDLE"),
    "pinky": (("curl", "curl", "curl", "ext"), None, "IDLE"),
    "middle": (("curl", "ext", "curl", "curl"), None, "IDLE"),
    "thumbfist": (("curl",) * 4, "out", "IDLE"),
    "pinch": (("half", "curl", "curl", "curl"), "out", "DRAW"),   # índice y pulgar tocándose
    "half_index": (("half", "curl", "curl", "curl"), None, "AMBIG"),
    "half_all": (("half",) * 4, None, "AMBIG"),
}


def sequence(name, rng, frames=30, noise=0.003):
    """Lista de landmarks para una pose mantenida, con orientación/escala/posición/ruido aleatorios."""
    fingers, thumb, truth = POSES[name]
    thumb = thumb or rng.choice(["out", "fold"])
    m = hand_model(fingers, thumb, rng, spread=rng.uniform(0, 8) if name == "select" else 0)
    if name == "pinch":
        m[4] = m[8] + rng.normal(0, 0.01, 3)
        m[3] = (m[2] + m[4]) / 2
    size = rng.uniform(0.18, 0.40)
    edge = rng.integers(0, 5)   # 0 centro; 1-4 bordes
    cx, cy = rng.uniform(0.3, 0.7), rng.uniform(0.45, 0.8)
    if edge == 1: cx = rng.uniform(0.02, 0.1)
    if edge == 2: cx = rng.uniform(0.9, 0.98)
    if edge == 3: cy = rng.uniform(0.02, 0.15) + size * 0.6
    if edge == 4: cy = rng.uniform(0.9, 1.0)
    base = place(m, rng, size, (cx, cy), roll=rng.uniform(-60, 60), pitch=rng.uniform(-35, 35), yaw=rng.uniform(-30, 30))
    return [base + np.pad(rng.normal(0, noise, (21, 2)), ((0, 0), (0, 1))).astype(np.float32) for _ in range(frames)], truth
