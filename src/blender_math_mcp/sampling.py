"""Turn math into geometry: adaptive curve sampling, clipping, implicit curves, surfaces.

All sampling happens in *math* coordinates.  ``scale`` (math -> scene units per
axis) is only used to measure errors in on-screen units, so the tolerance
``tol`` means "the drawn polyline never deviates from the true curve by more
than ``tol`` scene units" (up to the refinement limit).

Correctness rules implemented here:

* Curves are refined adaptively where they bend, so they are smooth at any zoom
  level that the tolerance was chosen for.
* Poles and jump discontinuities (``tan``, ``1/x``, ``floor``) are *detected*
  and the curve is broken there instead of drawing a false vertical line.
* Undefined points (``sqrt(x)`` for ``x < 0``, non-real results) leave gaps.
* Clipping to the viewing box happens at the exact crossing parameter, found by
  bisection on the real function (not by linear interpolation of samples).
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

CurveFn = Callable[[np.ndarray], np.ndarray]  # t (N,) -> points (N, d); NaN where undefined


def adaptive_sample(
    F: CurveFn,
    t0: float,
    t1: float,
    scale: Sequence[float],
    tol: float = 1e-3,
    lo: Sequence[float] | None = None,
    hi: Sequence[float] | None = None,
    initial: int = 256,
    max_depth: int = 16,
    max_points: int = 60_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Adaptively sample a parametric curve.

    Returns ``(t, P, brk)`` where ``brk[i]`` is True when the curve must be broken
    between sample ``i`` and ``i + 1`` (a detected discontinuity).
    """
    if not (np.isfinite(t0) and np.isfinite(t1)) or t1 <= t0:
        raise ValueError(f"Invalid parameter range [{t0}, {t1}]")
    s = np.asarray(scale, dtype=float)
    lo_ = None if lo is None else np.asarray(lo, dtype=float)
    hi_ = None if hi is None else np.asarray(hi, dtype=float)

    t = np.linspace(t0, t1, initial + 1)
    P = F(t)
    cand = np.ones(len(t) - 1, dtype=bool)

    for _ in range(max_depth):
        idx = np.nonzero(cand)[0]
        if idx.size == 0 or len(t) + idx.size > max_points:
            break
        ta, tb = t[idx], t[idx + 1]
        tm = 0.5 * (ta + tb)
        Pa, Pb, Pm = P[idx], P[idx + 1], F(tm)

        fa, fb, fm = (np.all(np.isfinite(X), axis=1) for X in (Pa, Pb, Pm))
        dev = np.linalg.norm((Pm - 0.5 * (Pa + Pb)) * s, axis=1)
        refine = np.where(fa & fb & fm, dev > tol, ~((fa == fb) & (fb == fm)))
        # A NaN-free interval whose midpoint is NaN (or vice versa) must be refined
        # so that the edge of the domain is located precisely.
        if lo_ is not None and hi_ is not None:
            # No need to refine what is invisible: all three points beyond the same wall.
            same_side = np.zeros_like(refine)
            for d in range(P.shape[1]):
                with np.errstate(invalid="ignore"):
                    same_side |= (Pa[:, d] > hi_[d]) & (Pb[:, d] > hi_[d]) & (Pm[:, d] > hi_[d])
                    same_side |= (Pa[:, d] < lo_[d]) & (Pb[:, d] < lo_[d]) & (Pm[:, d] < lo_[d])
            refine &= ~same_side

        # Insert all midpoints (free accuracy); only refined halves stay candidates.
        order = np.argsort(np.concatenate([t, tm]), kind="stable")
        t = np.concatenate([t, tm])[order]
        P = np.concatenate([P, Pm])[order]
        new_cand = np.zeros(len(t) - 1, dtype=bool)
        # position of each inserted midpoint in the new array
        pos = np.searchsorted(t, tm)
        new_cand[pos[refine] - 1] = True
        new_cand[pos[refine]] = True
        cand = new_cand

    # Discontinuity detection: intervals refined down to the resolution limit
    # that still show a large jump are genuine breaks.
    w_min = (t1 - t0) / initial / 2 ** max_depth
    width = np.diff(t)
    finite = np.all(np.isfinite(P), axis=1)
    chord = np.linalg.norm(np.diff(P, axis=0) * s, axis=1)
    jump_tol = max(20 * tol, 1e-9)
    with np.errstate(invalid="ignore"):
        brk = (width <= 4 * w_min) & finite[:-1] & finite[1:] & (chord > jump_tol)
    # A jump that ends in an interval which is still a refinement candidate
    # (max_points hit) is also suspicious; break conservatively.
    brk |= cand & finite[:-1] & finite[1:] & (chord > 50 * jump_tol)
    return t, P, brk


def split_polylines(t: np.ndarray, P: np.ndarray, brk: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split at NaN samples and at detected breaks into ``(t, P)`` runs."""
    finite = np.all(np.isfinite(P), axis=1)
    runs, start = [], None
    for i in range(len(t)):
        if not finite[i]:
            if start is not None and i - start >= 2:
                runs.append((t[start:i], P[start:i]))
            start = None
            continue
        if start is None:
            start = i
        if i < len(brk) and brk[i]:
            if i + 1 - start >= 2:
                runs.append((t[start:i + 1], P[start:i + 1]))
            start = None
    if start is not None and len(t) - start >= 2:
        runs.append((t[start:], P[start:]))
    return runs


def _inside(P: np.ndarray, lo: np.ndarray, hi: np.ndarray, eps: float = 0.0) -> np.ndarray:
    return np.all((P >= lo - eps) & (P <= hi + eps), axis=-1)


def _boundary_point(F: CurveFn, t_in: float, t_out: float, lo, hi, iters: int = 60) -> tuple[float, np.ndarray]:
    """``(t, point)`` where the curve leaves the box, by bisection on the parameter."""
    a, b = t_in, t_out
    for _ in range(iters):
        m = 0.5 * (a + b)
        pm = F(np.array([m]))[0]
        if np.all(np.isfinite(pm)) and _inside(pm, lo, hi):
            a = m
        else:
            b = m
    p = F(np.array([a]))[0].copy()
    # Snap the coordinate(s) that hit the wall exactly onto the wall.
    pb = F(np.array([b]))[0]
    for d in range(len(p)):
        if np.isfinite(pb[d]) and pb[d] > hi[d]:
            p[d] = hi[d]
        elif np.isfinite(pb[d]) and pb[d] < lo[d]:
            p[d] = lo[d]
    return a, p


def _beyond_same_wall(a: np.ndarray, b: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> bool:
    return bool(np.any((a > hi) & (b > hi)) or np.any((a < lo) & (b < lo)))


def _densify_crossings(t, P, F, lo, hi, max_depth: int = 40):
    """Find samples inside the box between two outside samples whose chord may cross it.

    Example: a steep line that jumps from above the box to below it between two
    samples -- neither sample is visible, but the curve crosses the box.
    """
    ins = _inside(P, lo, hi, eps=1e-12)
    extra_t: list[float] = []

    def search(a, pa, b, pb, depth):
        if depth > max_depth or _beyond_same_wall(pa, pb, lo, hi):
            return
        m = 0.5 * (a + b)
        pm = F(np.array([m]))[0]
        if not np.all(np.isfinite(pm)):
            return
        if _inside(pm, lo, hi):
            extra_t.append(m)
            return
        search(a, pa, m, pm, depth + 1)
        search(m, pm, b, pb, depth + 1)

    for i in range(len(t) - 1):
        if not ins[i] and not ins[i + 1]:
            search(t[i], P[i], t[i + 1], P[i + 1], 0)
    if not extra_t:
        return t, P
    te = np.array(extra_t)
    t2 = np.concatenate([t, te])
    P2 = np.concatenate([P, F(te)])
    order = np.argsort(t2, kind="stable")
    return t2[order], P2[order]


def clip_runs_t(
    runs: list[tuple[np.ndarray, np.ndarray]], F: CurveFn, lo: Sequence[float], hi: Sequence[float]
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Clip each ``(t, P)`` run to the box [lo, hi] with exact crossing points (keeps parameters)."""
    lo_, hi_ = np.asarray(lo, float), np.asarray(hi, float)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for t, P in runs:
        t, P = _densify_crossings(t, P, F, lo_, hi_)
        ins = _inside(P, lo_, hi_, eps=1e-12)
        ct: list[float] = []
        cp: list[np.ndarray] = []
        for i in range(len(t)):
            if ins[i]:
                if not cp and i > 0:  # entering
                    tb, pb = _boundary_point(F, t[i], t[i - 1], lo_, hi_)
                    ct.append(tb)
                    cp.append(pb)
                ct.append(t[i])
                cp.append(P[i])
            else:
                if cp:  # leaving
                    tb, pb = _boundary_point(F, t[i - 1], t[i], lo_, hi_)
                    ct.append(tb)
                    cp.append(pb)
                    out.append((np.array(ct), np.array(cp)))
                    ct, cp = [], []
        if len(cp) >= 2:
            out.append((np.array(ct), np.array(cp)))
    return [(a, b) for a, b in out if len(b) >= 2]


def clip_runs(
    runs: list[tuple[np.ndarray, np.ndarray]], F: CurveFn, lo: Sequence[float], hi: Sequence[float]
) -> list[np.ndarray]:
    """Clip each run to the axis-aligned box [lo, hi], with exact crossing points."""
    return [P for _, P in clip_runs_t(runs, F, lo, hi)]


def simplify_polyline(P: np.ndarray, s: np.ndarray, tol: float) -> np.ndarray:
    """Drop points that are (almost) collinear with their neighbours (Visvalingam-lite)."""
    if len(P) <= 2:
        return P
    keep = np.ones(len(P), dtype=bool)
    Q = P * s
    last = 0
    for i in range(1, len(P) - 1):
        a, b, c = Q[last], Q[i], Q[i + 1]
        ab, ac = b - a, c - a
        L = np.linalg.norm(ac)
        if L == 0:
            keep[i] = False
            continue
        # distance of b from line a-c (works in 2D and 3D)
        dist = np.linalg.norm(ab - np.dot(ab, ac) / (L * L) * ac)
        if dist < tol * 0.1 and np.dot(ab, ac) > 0 and np.linalg.norm(ab) < L:
            keep[i] = False
        else:
            last = i
    return P[keep]


def sample_curve_t(
    F: CurveFn,
    t0: float,
    t1: float,
    scale: Sequence[float],
    tol: float,
    lo: Sequence[float] | None = None,
    hi: Sequence[float] | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Like ``sample_curve`` but returns ``(t, P)`` runs (unsimplified) for spline fitting."""
    t, P, brk = adaptive_sample(F, t0, t1, scale, tol, lo, hi)
    runs = split_polylines(t, P, brk)
    if lo is not None and hi is not None:
        return clip_runs_t(runs, F, lo, hi)
    return runs


def sample_curve(
    F: CurveFn,
    t0: float,
    t1: float,
    scale: Sequence[float],
    tol: float,
    lo: Sequence[float] | None = None,
    hi: Sequence[float] | None = None,
) -> list[np.ndarray]:
    """Full pipeline: adaptive sampling -> discontinuity split -> exact clipping."""
    t, P, brk = adaptive_sample(F, t0, t1, scale, tol, lo, hi)
    runs = split_polylines(t, P, brk)
    if lo is not None and hi is not None:
        polys = clip_runs(runs, F, lo, hi)
    else:
        polys = [p for _, p in runs]
    s = np.asarray(scale, float)
    return [simplify_polyline(p, s, tol) for p in polys]


# --------------------------------------------------------------------------- implicit curves


def implicit_curve(
    f: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x_range: Sequence[float],
    y_range: Sequence[float],
    resolution: int = 400,
    newton_steps: int = 4,
) -> list[np.ndarray]:
    """Zero set of ``f(x, y)`` as polylines, snapped onto the true curve with Newton steps.

    Sign changes that are poles (``|f|`` large on the "contour") are removed.
    """
    import contourpy

    xs = np.linspace(*x_range, resolution)
    ys = np.linspace(*y_range, resolution)
    X, Y = np.meshgrid(xs, ys)
    Z = f(X, Y)
    if not np.any(np.isfinite(Z)):
        return []
    gen = contourpy.contour_generator(X, Y, np.ma.masked_invalid(Z), line_type="Separate")
    lines = gen.lines(0.0)
    h = max((x_range[1] - x_range[0]), (y_range[1] - y_range[0])) / resolution
    zscale = np.nanpercentile(np.abs(Z), 90) or 1.0
    out = []
    for L in lines:
        P = np.asarray(L, float)
        P = _newton_snap(f, P, h, newton_steps)
        val = np.abs(f(P[:, 0], P[:, 1]))
        good = np.isfinite(val) & (val < 1e-6 * max(zscale, 1.0) + 1e-9)
        # Split where points failed to converge (poles, spurious sign flips).
        start = None
        for i, g in enumerate(good):
            if g and start is None:
                start = i
            elif not g and start is not None:
                if i - start >= 2:
                    out.append(P[start:i])
                start = None
        if start is not None and len(P) - start >= 2:
            seg = P[start:]
            out.append(seg)
    return out


def _newton_snap(f, P, h, steps):
    P = P.copy()
    eps = h * 1e-3
    for _ in range(steps):
        x, y = P[:, 0], P[:, 1]
        v = f(x, y)
        gx = (f(x + eps, y) - f(x - eps, y)) / (2 * eps)
        gy = (f(x, y + eps) - f(x, y - eps)) / (2 * eps)
        g2 = gx * gx + gy * gy
        with np.errstate(all="ignore"):
            step = np.where(g2 > 0, v / g2, 0.0)
        dx, dy = -step * gx, -step * gy
        # Never move a point further than one grid cell (keeps branches separate).
        mag = np.hypot(dx, dy)
        ok = np.isfinite(mag) & (mag < h)
        P[ok, 0] += dx[ok]
        P[ok, 1] += dy[ok]
    return P


# --------------------------------------------------------------------------- surfaces


def grid_surface(
    F: Callable[[np.ndarray, np.ndarray], np.ndarray],
    u_range: Sequence[float],
    v_range: Sequence[float],
    nu: int,
    nv: int,
    lo: Sequence[float] | None = None,
    hi: Sequence[float] | None = None,
) -> tuple[np.ndarray, list[list[int]], np.ndarray]:
    """Sample ``F(u, v) -> (..., 3)`` on a grid.

    Returns ``(vertices, quads, uv)``; quads touching undefined or out-of-box
    vertices are dropped (never clamped), unused vertices removed.
    """
    us = np.linspace(u_range[0], u_range[1], nu + 1)
    vs = np.linspace(v_range[0], v_range[1], nv + 1)
    U, V = np.meshgrid(us, vs, indexing="ij")
    P = F(U, V).reshape(-1, 3)
    ok = np.all(np.isfinite(P), axis=1)
    if lo is not None and hi is not None:
        ok &= _inside(P, np.asarray(lo, float), np.asarray(hi, float), eps=1e-9)
    idx = np.arange((nu + 1) * (nv + 1)).reshape(nu + 1, nv + 1)
    a, b, c, d = idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]
    quads = np.stack([a, b, c, d], axis=-1).reshape(-1, 4)
    keep = ok[quads].all(axis=1)
    quads = quads[keep]
    used = np.unique(quads)
    remap = -np.ones(len(P), dtype=int)
    remap[used] = np.arange(len(used))
    uv = np.stack([U.ravel(), V.ravel()], axis=1)[used]
    return P[used], remap[quads].tolist(), uv


# --------------------------------------------------------------------------- ticks


def nice_step(span: float, target: int = 8) -> float:
    if span <= 0 or not np.isfinite(span):
        return 1.0
    raw = span / max(target, 1)
    mag = 10 ** np.floor(np.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if m * mag >= raw:
            return float(m * mag)
    return float(10 * mag)


def tick_values(lo: float, hi: float, step: float) -> list[float]:
    k0 = int(np.ceil(lo / step - 1e-9))
    k1 = int(np.floor(hi / step + 1e-9))
    return [k * step for k in range(k0, k1 + 1)]


def tick_label(v: float, step: float, style: str = "decimal") -> str:
    """LaTeX label for a tick value.  ``style='pi'`` gives multiples of pi."""
    if style == "pi":
        from fractions import Fraction

        fr = Fraction(v / np.pi).limit_denominator(24)
        if fr == 0:
            return "0"
        sign = "-" if fr < 0 else ""
        n, d = abs(fr.numerator), fr.denominator
        num = r"\pi" if n == 1 else rf"{n}\pi"
        return f"{sign}{num}" if d == 1 else rf"{sign}\frac{{{num}}}{{{d}}}"
    if abs(v) < step * 1e-9:
        return "0"
    decimals = max(0, -int(np.floor(np.log10(step) + 1e-9)))
    if step * 10 ** decimals % 1 > 1e-9:  # e.g. step 2.5 -> one decimal
        decimals += 1
    return f"{v:.{decimals}f}"  # typeset in math mode, so '-' becomes a true minus sign
