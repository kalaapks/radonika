"""Pretty printing: LaTeX in Jupyter, readable unicode in a terminal."""
from __future__ import annotations

import numpy as np


def in_notebook() -> bool:
    try:
        from IPython import get_ipython

        ip = get_ipython()
        return ip is not None and ip.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def fmt(v) -> str:
    """Format a scalar or length-2 parameter ('2' if both coordinates agree, else '[a, b]')."""
    a = np.atleast_1d(np.asarray(v, float))
    if a.size == 1 or np.allclose(a, a[0]):
        return f"{a[0]:g}"
    return "[" + ", ".join(f"{x:g}" for x in a) + "]"


def show_math(title: str, lines) -> None:
    """lines: list of (latex, plain_text) pairs."""
    if in_notebook():
        from IPython.display import Markdown, Math, display

        display(Markdown(f"**{title}**"))
        for tex, _ in lines:
            display(Math(tex))
    else:
        print(title)
        for _, plain in lines:
            print("    " + plain)
