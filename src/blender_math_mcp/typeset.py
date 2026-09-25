"""Typesetting: turn LaTeX into exact glyph outlines.

The outlines are produced by matplotlib's ``TextPath``.  Two backends exist:

* ``latex``    - a real TeX installation (``latex`` + ``kpsewhich`` on PATH).
                 Supports all of LaTeX/amsmath: matrices, ``aligned``, ``cases``...
* ``mathtext`` - matplotlib's built-in TeX-subset parser, rendered with the
                 Computer Modern fonts.  Needs no external programs.

``auto`` uses ``latex`` when it is installed and ``mathtext`` otherwise.

The result is a list of closed cubic Bezier contours in the format the Blender
add-on consumes directly (``co`` / ``hl`` / ``hr`` per point, all cyclic), so
what appears in Blender is the font outline itself, not a raster or an
approximation.
"""

from __future__ import annotations

import contextlib
import functools
import shutil
from dataclasses import dataclass, field

import numpy as np

MOVETO, LINETO, CURVE3, CURVE4, CLOSEPOLY = 1, 2, 3, 4, 79

DEFAULT_PREAMBLE = r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}"

_ROUND = 6


class TypesetError(ValueError):
    """Raised when a formula cannot be typeset."""


@dataclass
class Typeset:
    """Outlines of a typeset formula, in em units scaled by ``size``."""

    splines: list[dict]
    bbox: tuple[float, float, float, float]  # xmin, ymin, xmax, ymax (baseline at y=0)
    backend: str
    source: str
    notes: list[str] = field(default_factory=list)

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    def transformed(self, dx: float = 0.0, dy: float = 0.0, scale: float = 1.0) -> "Typeset":
        """Return a copy scaled about the origin and then translated."""

        def tx(pts):
            return [[round(p[0] * scale + dx, _ROUND), round(p[1] * scale + dy, _ROUND), 0.0] for p in pts]

        splines = [
            {"co": tx(s["co"]), "hl": tx(s["hl"]), "hr": tx(s["hr"]), "cyclic": s["cyclic"]}
            for s in self.splines
        ]
        x0, y0, x1, y1 = self.bbox
        bbox = (x0 * scale + dx, y0 * scale + dy, x1 * scale + dx, y1 * scale + dy)
        return Typeset(splines, bbox, self.backend, self.source, list(self.notes))


@functools.lru_cache(maxsize=1)
def latex_available() -> bool:
    return bool(shutil.which("latex") and shutil.which("kpsewhich"))


def resolve_backend(backend: str) -> str:
    backend = (backend or "auto").lower()
    if backend not in ("auto", "latex", "mathtext"):
        raise TypesetError(f"Unknown backend {backend!r}; use 'auto', 'latex' or 'mathtext'.")
    if backend == "auto":
        return "latex" if latex_available() else "mathtext"
    if backend == "latex" and not latex_available():
        raise TypesetError(
            "backend='latex' requested but no LaTeX installation was found "
            "(need `latex` and `kpsewhich` on PATH). Install TeX Live / MiKTeX / MacTeX "
            "or use backend='mathtext'."
        )
    return backend


def _strip_math_delimiters(tex: str) -> str:
    t = tex.strip()
    for open_, close in (("$$", "$$"), (r"\[", r"\]"), (r"\(", r"\)"), ("$", "$")):
        if t.startswith(open_) and t.endswith(close) and len(t) >= len(open_) + len(close):
            return t[len(open_): len(t) - len(close)].strip()
    return t


def _prepare_source(tex: str, mode: str, backend: str) -> str:
    if mode not in ("math", "text"):
        raise TypesetError("mode must be 'math' or 'text'")
    if mode == "text":
        # Text with inline $...$ math.  mathtext needs an even number of '$'.
        if (tex.count("$") - tex.count(r"\$")) % 2:
            raise TypesetError("Unbalanced '$' in text-mode input.")
        return tex
    body = _strip_math_delimiters(tex)
    if not body:
        raise TypesetError("Empty formula.")
    if backend == "latex":
        return r"$\displaystyle " + body + "$"
    return "$" + body + "$"


_TEX_PT = 10  # typeset at TeX's design size so fixed lengths (\arraycolsep, \jot...) are right


@contextlib.contextmanager
def _tex_at_design_size():
    """Make matplotlib run LaTeX at 10pt instead of its internal 100pt.

    matplotlib typesets usetex strings at 100pt and scales down.  Font-relative
    spacing survives that, but fixed dimensions such as ``\arraycolsep`` do not,
    which squeezes matrix columns together.  Here the DVI is produced at 10pt
    while glyph outlines are still loaded at high resolution, so positions and
    shapes are both exact.
    """
    from matplotlib import textpath
    from matplotlib.texmanager import TexManager

    class _DesignSizeTexManager:
        def make_dvi(self, tex, fontsize):
            return TexManager().make_dvi(tex, _TEX_PT)

    original = textpath.TexManager
    textpath.TexManager = _DesignSizeTexManager
    try:
        yield
    finally:
        textpath.TexManager = original


def _glyph_path(source: str, backend: str, preamble: str) -> tuple[np.ndarray, np.ndarray]:
    """Outline vertices and path codes in em units (baseline at y = 0)."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath, text_to_path

    try:
        if backend == "latex":
            rc = {"text.usetex": True, "text.latex.preamble": preamble}
            with matplotlib.rc_context(rc), _tex_at_design_size():
                verts, codes = text_to_path.get_text_path(FontProperties(family="serif"), source, ismath="TeX")
            # DVI positions are in pt for a 10pt font; the glyphs are scaled to
            # match (text.font_size / FONT_SCALE), so dividing by 10 gives ems.
            return np.asarray(verts, float) / _TEX_PT, np.asarray(codes)
        rc = {"mathtext.fontset": "cm", "mathtext.rm": "serif"}
        with matplotlib.rc_context(rc):
            prop = FontProperties(family="serif", math_fontfamily="cm")
            path = TextPath((0, 0), source, size=1.0, prop=prop)  # size=1 -> em units
        return np.asarray(path.vertices, float), np.asarray(path.codes)
    except Exception as exc:  # parse errors, LaTeX errors, missing fonts
        msg = str(exc).strip()
        if backend == "mathtext":
            msg += (
                "\n(mathtext supports a large TeX subset but not environments such as "
                "matrix/aligned/cases or \\text; install LaTeX to enable the 'latex' backend.)"
            )
        raise TypesetError(f"Could not typeset {source!r} with {backend}: {msg}") from exc


def _path_to_contours(vertices: np.ndarray, codes: np.ndarray) -> list[list[tuple]]:
    """Split a matplotlib path into contours of cubic segments (p0, c1, c2, p3)."""
    contours: list[list[tuple]] = []
    segs: list[tuple] = []
    start = cur = None
    i, n = 0, len(codes)

    def flush():
        nonlocal segs
        if segs:
            if start is not None and np.hypot(*(segs[-1][3] - start)) > 1e-9:
                # Close the contour explicitly with a straight edge.
                a, b = segs[-1][3], start
                segs.append((a, a + (b - a) / 3, a + 2 * (b - a) / 3, b))
            contours.append(segs)
        segs = []

    while i < n:
        code = codes[i]
        if code == MOVETO:
            flush()
            start = cur = vertices[i].astype(float)
            i += 1
        elif code == LINETO:
            p = vertices[i].astype(float)
            if np.hypot(*(p - cur)) > 1e-9:
                segs.append((cur, cur + (p - cur) / 3, cur + 2 * (p - cur) / 3, p))
            cur = p
            i += 1
        elif code == CURVE3:
            q, p = vertices[i].astype(float), vertices[i + 1].astype(float)
            segs.append((cur, cur + 2 / 3 * (q - cur), p + 2 / 3 * (q - p), p))
            cur = p
            i += 2
        elif code == CURVE4:
            c1, c2, p = (vertices[i + k].astype(float) for k in range(3))
            segs.append((cur, c1, c2, p))
            cur = p
            i += 3
        elif code == CLOSEPOLY:
            flush()
            cur = start
            i += 1
        else:  # STOP or unknown
            i += 1
    flush()
    return contours


def _contour_to_spline(segs: list[tuple]) -> dict:
    """Cubic segments of a closed contour -> Blender cyclic bezier spline."""
    n = len(segs)
    co, hl, hr = [], [], []
    for k, (p0, c1, _c2, _p3) in enumerate(segs):
        prev_c2 = segs[k - 1][2]  # handle coming into p0 (cyclic)
        co.append(p0)
        hl.append(prev_c2)
        hr.append(c1)
    r = lambda p: [round(float(p[0]), _ROUND), round(float(p[1]), _ROUND), 0.0]  # noqa: E731
    return {"co": [r(p) for p in co], "hl": [r(p) for p in hl], "hr": [r(p) for p in hr], "cyclic": True} if n else {}


def typeset(
    tex: str,
    mode: str = "math",
    backend: str = "auto",
    preamble: str | None = None,
) -> Typeset:
    """Typeset ``tex`` and return its outlines at 1 em = 1 unit, baseline at y=0."""
    if not isinstance(tex, str) or not tex.strip():
        raise TypesetError("Empty formula.")
    used = resolve_backend(backend)
    source = _prepare_source(tex, mode, used)
    verts, codes = _glyph_path(source, used, preamble or DEFAULT_PREAMBLE)
    if len(verts) == 0:
        raise TypesetError(f"{tex!r} produced no visible glyphs.")
    contours = _path_to_contours(verts, codes)
    splines = [s for s in (_contour_to_spline(c) for c in contours) if s and len(s["co"]) >= 2]
    real = verts[codes != CLOSEPOLY]
    xmin, ymin = real.min(axis=0)
    xmax, ymax = real.max(axis=0)
    return Typeset(splines, (float(xmin), float(ymin), float(xmax), float(ymax)), used, tex)


def merge(parts: list[Typeset]) -> Typeset:
    """Combine already-positioned typeset pieces into one."""
    if not parts:
        raise TypesetError("Nothing to merge.")
    splines = [s for p in parts for s in p.splines]
    bbox = (
        min(p.bbox[0] for p in parts),
        min(p.bbox[1] for p in parts),
        max(p.bbox[2] for p in parts),
        max(p.bbox[3] for p in parts),
    )
    return Typeset(splines, bbox, parts[0].backend, "\n".join(p.source for p in parts))


ANCHORS_X = {"left": 0.0, "center": 0.5, "right": 1.0}
ANCHORS_Y = {"bottom": 0.0, "center": 0.5, "top": 1.0}


def anchored(ts: Typeset, size: float, align_x: str = "center", align_y: str = "center") -> Typeset:
    """Scale to ``size`` (em height in scene units) and move the anchor point to the origin.

    ``align_y`` may also be ``'baseline'`` to keep the TeX baseline at y=0.
    """
    if align_x not in ANCHORS_X:
        raise TypesetError(f"align_x must be one of {sorted(ANCHORS_X)}")
    if align_y not in (*ANCHORS_Y, "baseline"):
        raise TypesetError(f"align_y must be one of {sorted([*ANCHORS_Y, 'baseline'])}")
    x0, y0, x1, y1 = ts.bbox
    ax = x0 + (x1 - x0) * ANCHORS_X[align_x]
    ay = 0.0 if align_y == "baseline" else y0 + (y1 - y0) * ANCHORS_Y[align_y]
    return ts.transformed(dx=-ax * size, dy=-ay * size, scale=size)


def stack_derivation(lines: list[str], backend: str = "auto", line_gap: float = 0.6) -> Typeset:
    """Lay out ``lines[0] = lines[1] = lines[2] ...`` with the '=' signs aligned.

    Output (like an ``aligned`` environment)::

        lhs = step1
            = step2
            = step3
    """
    if len(lines) < 2:
        raise TypesetError("A derivation needs at least two expressions.")
    lhs = typeset(lines[0], backend=backend)
    rights = [typeset("= " + ln, backend=backend) for ln in lines[1:]]
    gap = 0.28  # em, roughly a TeX thick space
    x_eq = lhs.bbox[2] + gap
    parts = [lhs]
    y = 0.0
    for k, r in enumerate(rights):
        if k > 0:
            # Baseline-to-baseline distance: previous line's depth + this line's
            # height + a gap, so tall fractions never collide.
            prev_depth = -min(rights[k - 1].bbox[1], 0.0)
            y -= prev_depth + max(r.bbox[3], 0.0) + line_gap * 0.5
        parts.append(r.transformed(dx=x_eq - r.bbox[0], dy=y))
    out = merge(parts)
    out.source = " = ".join(lines)
    return out
