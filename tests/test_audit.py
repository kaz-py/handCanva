import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "tests")]
import audit_gestures as A  # noqa: E402


def test_confusion_battery_error_budget():
    """Batería sintética (rotación ±60°, escorzo, escala, bordes, ruido): <=2 % de errores y casi nunca BORRAR/SELECCIÓN espurios."""
    conf, flick = A.run(n=40, noise=0.005, seed=7)
    e, tot = A.errors(conf)
    assert e / tot <= 0.02, (e, tot, conf)
    assert sum(flick.values()) <= 0.01 * tot
    for name in ("draw", "select", "open", "three", "pinky", "middle", "thumbfist"):
        assert conf[name]["ERASE"] == 0
