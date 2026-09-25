"""Small geometry builders and color handling (pure Python, no Blender needed)."""

from __future__ import annotations

import math
import re
from typing import Sequence

import numpy as np

# --------------------------------------------------------------------------- colors

NAMED_COLORS = {
    "black": "#000000", "white": "#ffffff", "gray": "#808080", "grey": "#808080",
    "lightgray": "#d3d3d3", "lightgrey": "#d3d3d3", "darkgray": "#404040", "darkgrey": "#404040",
    "red": "#d62728", "green": "#2ca02c", "blue": "#1f77b4", "yellow": "#f5c518",
    "orange": "#ff7f0e", "purple": "#9467bd", "pink": "#e377c2", "brown": "#8c564b",
    "cyan": "#17becf", "teal": "#008080", "magenta": "#d62ad6", "navy": "#1a2a6c",
    "gold": "#ffd700", "olive": "#bcbd22",
    # Palette tuned for dark backgrounds (in the style of popular math videos).
    "math_blue": "#58c4dd", "math_teal": "#5cd0b3", "math_green": "#83c167",
    "math_yellow": "#ffff00", "math_gold": "#f0ac5f", "math_red": "#fc6255",
    "math_maroon": "#c55f73", "math_purple": "#9a72ac", "math_pink": "#d147bd",
    "math_grey": "#888888",
}
# matplotlib-style cycle: "C0".."C9"
_CYCLE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def parse_color(color: str | Sequence[float] | None, default: str = "white") -> list[float]:
    """Color -> *linear* RGBA for Blender.

    Accepts ``'#rrggbb'``, ``'#rrggbbaa'``, a name (see ``NAMED_COLORS``), ``'C0'``-``'C9'``,
    or a list of 3/4 sRGB floats in 0..1.  sRGB is converted to linear so that,
    with the 'Standard' view transform, the on-screen color is exactly the
    requested one.
    """
    if color is None:
        color = default
    if isinstance(color, str):
        c = color.strip().lower().replace(" ", "_")
        m = re.fullmatch(r"c(\d)", c)
        if m:
            c = _CYCLE[int(m.group(1))]
        c = NAMED_COLORS.get(c, c)
        if not re.fullmatch(r"#?[0-9a-f]{6}([0-9a-f]{2})?", c):
            raise ValueError(f"Unknown color {color!r}. Use '#rrggbb', a name like 'blue', or [r, g, b].")
        c = c.lstrip("#")
        vals = [int(c[i:i + 2], 16) / 255 for i in range(0, len(c), 2)]
    else:
        vals = [float(v) for v in color]
        if len(vals) not in (3, 4) or not all(0 <= v <= 1 for v in vals):
            raise ValueError("Color lists must have 3 or 4 components in 0..1")
    rgb = [srgb_to_linear(v) for v in vals[:3]]
    alpha = vals[3] if len(vals) == 4 else 1.0
    return [round(v, 6) for v in rgb] + [alpha]


def colormap(values: np.ndarray, name: str = "viridis") -> np.ndarray:
    """Values -> linear RGBA (N, 4) using a matplotlib colormap."""
    import matplotlib

    cmap = matplotlib.colormaps[name]
    v = np.asarray(values, float)
    lo, hi = np.nanmin(v), np.nanmax(v)
    norm = (v - lo) / (hi - lo) if hi > lo else np.full_like(v, 0.5)
    rgba = cmap(norm)
    lin = np.where(rgba[:, :3] <= 0.04045, rgba[:, :3] / 12.92, ((rgba[:, :3] + 0.055) / 1.055) ** 2.4)
    return np.concatenate([lin, rgba[:, 3:]], axis=1)


# --------------------------------------------------------------------------- meshes


def _frame(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors orthogonal to ``direction`` (and each other)."""
    d = direction / np.linalg.norm(direction)
    helper = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(d, helper)
    u /= np.linalg.norm(u)
    return u, np.cross(d, u)


def arrow_mesh(
    start: Sequence[float],
    end: Sequence[float],
    thickness: float = 0.04,
    head_length: float | None = None,
    head_width: float | None = None,
    style: str = "3d",
    segments: int = 24,
    plane_normal: Sequence[float] = (0, 0, 1),
) -> tuple[list, list]:
    """Arrow from ``start`` to ``end`` whose tip lands *exactly* on ``end``.

    ``style='3d'``: cylinder shaft + cone head.  ``style='flat'``: a flat
    polygon lying in the plane with normal ``plane_normal`` (crisp textbook look).
    The head never exceeds half the arrow's length.
    """
    a, b = np.asarray(start, float), np.asarray(end, float)
    vec = b - a
    L = float(np.linalg.norm(vec))
    if L < 1e-12:
        raise ValueError("Arrow start and end coincide.")
    d = vec / L
    hl = head_length if head_length is not None else 4.0 * thickness
    hw = head_width if head_width is not None else 3.0 * thickness
    if hl > 0.5 * L:  # keep proportions for short arrows
        k = 0.5 * L / hl
        hl, hw = hl * k, hw * k
    shaft_end = b - d * hl
    r = thickness / 2

    if style == "flat":
        n = np.asarray(plane_normal, float)
        side = np.cross(n, d)
        if np.linalg.norm(side) < 1e-9:
            raise ValueError("Flat arrow direction is parallel to plane_normal.")
        side /= np.linalg.norm(side)
        verts = [a - side * r, shaft_end - side * r, shaft_end - side * hw, b,
                 shaft_end + side * hw, shaft_end + side * r, a + side * r]
        return [list(map(float, v)) for v in verts], [list(range(7))]

    u, w = _frame(d)
    ang = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    ring = [np.cos(t) * u + np.sin(t) * w for t in ang]
    verts, faces = [], []
    # shaft: bottom ring (0..n-1), top ring (n..2n-1)
    for c in (a, shaft_end):
        verts += [c + r * q for q in ring]
    n = segments
    faces += [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
    faces.append(list(range(n - 1, -1, -1)))  # bottom cap
    # head: base ring (2n..3n-1), tip (3n)
    verts += [shaft_end + hw * q for q in ring]
    verts.append(b)
    tip = 3 * n
    faces += [[2 * n + i, 2 * n + (i + 1) % n, tip] for i in range(n)]
    faces.append(list(range(3 * n - 1, 2 * n - 1, -1)))  # head base
    return [list(map(float, v)) for v in verts], faces


def uv_sphere(center: Sequence[float], radius: float, rings: int = 16, segments: int = 32) -> tuple[list, list]:
    c = np.asarray(center, float)
    verts = [c + [0, 0, radius]]
    for i in range(1, rings):
        phi = math.pi * i / rings
        for j in range(segments):
            th = 2 * math.pi * j / segments
            verts.append(c + radius * np.array([math.sin(phi) * math.cos(th), math.sin(phi) * math.sin(th), math.cos(phi)]))
    verts.append(c - [0, 0, radius])
    faces = []
    last = len(verts) - 1
    for j in range(segments):
        faces.append([0, 1 + j, 1 + (j + 1) % segments])
    for i in range(rings - 2):
        o1, o2 = 1 + i * segments, 1 + (i + 1) * segments
        for j in range(segments):
            faces.append([o1 + j, o2 + j, o2 + (j + 1) % segments, o1 + (j + 1) % segments])
    o = 1 + (rings - 2) * segments
    for j in range(segments):
        faces.append([o + (j + 1) % segments, o + j, last])
    return [list(map(float, v)) for v in verts], faces


KAPPA = 4 * (math.sqrt(2) - 1) / 3  # cubic Bezier circle constant


def bezier_circle(center: Sequence[float], radius: float) -> dict:
    """An (essentially exact) circle as a cyclic 4-point Bezier spline in the XY plane."""
    cx, cy = center[0], center[1]
    cz = center[2] if len(center) > 2 else 0.0
    k = KAPPA * radius
    pts = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    co, hl, hr = [], [], []
    for x, y in pts:
        px, py = cx + x * radius, cy + y * radius
        tx, ty = -y, x  # tangent (counter-clockwise)
        co.append([px, py, cz])
        hl.append([px - tx * k, py - ty * k, cz])
        hr.append([px + tx * k, py + ty * k, cz])
    return {"co": co, "hl": hl, "hr": hr, "cyclic": True}


def polyline_spline(points: np.ndarray, cyclic: bool = False, ndigits: int = 6) -> dict:
    P = np.asarray(points, float)
    if P.shape[1] == 2:
        P = np.concatenate([P, np.zeros((len(P), 1))], axis=1)
    return {"poly": np.round(P, ndigits).tolist(), "cyclic": cyclic}
