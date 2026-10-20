"""Al iniciar la app la herramienta activa es el pincel (decisión del usuario)."""
from main import App


def test_starts_with_brush():
    app = App(1280, 720)
    assert app.ui.tool == "brush"
    assert app.canvas is not None
