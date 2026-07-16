"""Shared matplotlib style and helpers (non-interactive Agg backend)."""

from __future__ import annotations

import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "lines.linewidth": 2.0,
    "font.family": "sans-serif",
})


_PANEL_RE = re.compile(r"^([A-Za-z])( {2,})(.*)$", flags=re.DOTALL)


def panel_label(ax, letter):
    """Place a bold panel label at the top-left of an axes."""
    ax.set_title(letter, loc="left", fontweight="bold", fontsize=13)


def finalize(fig):
    """Prepare a figure for saving: drop the figure-level title, and move the
    leading panel label (the 'A  ', 'B  ', ... convention) to a bold top-left
    label, leaving the (short) description as the centered title."""
    st = getattr(fig, "_suptitle", None)
    if st is not None:
        st.set_visible(False)
    for ax in fig.axes:
        m = _PANEL_RE.match(ax.get_title())
        if m:
            ax.set_title(m.group(3), loc="center")
            panel_label(ax, m.group(1))


def save(fig, name: str):
    """Finalize and save the figure as a PDF."""
    finalize(fig)
    os.makedirs(FIG_DIR, exist_ok=True)
    base = os.path.splitext(name)[0]
    path = os.path.join(FIG_DIR, f"{base}.pdf")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print(f"  saved {path}")
    return path
