"""Auditoría de clasificación: matriz de confusión modo-real vs modo-estable sobre una batería sintética."""
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tests")]
import synth  # noqa: E402
from gestures import GestureTracker  # noqa: E402

MODES = ["DRAW", "SELECT", "ERASE", "IDLE"]


def run(n=150, noise=0.003, seed=0):
    rng = np.random.default_rng(seed)
    conf, flick = {}, Counter()
    for name in synth.POSES:
        c = Counter()
        for _ in range(n):
            seq, truth = synth.sequence(name, rng, noise=noise)
            t = GestureTracker(1280, 720)
            seen = set()
            for lm in seq:
                t.update(lm)
                seen.add(t.mode)
            c[t.mode] += 1
            bad = {"ERASE", "SELECT"} - {truth}
            if (seen & bad) and truth != "ERASE" and truth != "SELECT":
                flick[name] += 1
        conf[name] = c
    return conf, flick


def errors(conf):
    e = tot = 0
    for name, c in conf.items():
        truth = synth.POSES[name][2]
        for m, k in c.items():
            tot += k
            if truth == "AMBIG":
                e += k if m in ("ERASE", "SELECT") else 0
            elif m != truth:
                e += k
    return e, tot


if __name__ == "__main__":
    conf, flick = run(noise=float(sys.argv[1]) if len(sys.argv) > 1 else 0.003)
    print(f"{'pose':12s}{'verdad':8s}" + "".join(f"{m:>8s}" for m in MODES))
    for name, c in conf.items():
        print(f"{name:12s}{synth.POSES[name][2]:8s}" + "".join(f"{c[m]:8d}" for m in MODES))
    e, tot = errors(conf)
    print(f"errores finales: {e}/{tot} ({100*e/tot:.1f}%)  | secuencias con ERASE/SELECT espurio (incl. transitorio): {dict(flick)}")
