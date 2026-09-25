"""Splines: exact Bezier representation of curves, interpolation splines, Bezier/B-spline math.

Why Bezier output
-----------------
Blender curves are cubic Beziers.  A cubic Bezier with end tangents taken from
the *true derivative* (cubic Hermite interpolation) matches a smooth curve to
4th order, so far fewer points are needed than with a polyline and the result is
smooth at any zoom.  Every segment is checked against the true function
(error <= tol in scene units at many interior points) before it is accepted.

Any polynomial piece of degree <= 3 (interpolating splines, Taylor polynomials
up to degree 3, cubic B-spline spans, Bezier curves of degree <= 3) is
represented *exactly*: the Hermite form of a cubic is the cubic itself.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import sympy as sp

from .expressions import ExpressionError

# --------------------------------------------------------------------------- Hermite fitting


def _hermite_eval(P0, P1, D0, D1, dt, u):
    """Cubic Hermite (= Bezier with handles P0 + D0 dt/3, P1 - D1 dt/3) at u in [0, 1]."""
    u = u[:, None]
    h00 = 2 * u ** 3 - 3 * u ** 2 + 1
    h10 = u ** 3 - 2 * u ** 2 + u
    h01 = -2 * u ** 3 + 3 * u ** 2
    h11 = u ** 3 - u ** 2
    return h00 * P0 + h10 * dt * D0 + h01 * P1 + h11 * dt * D1


def hermite_bezier(
    F: Callable[[np.ndarray], np.ndarray],
    dF: Callable[[np.ndarray], np.ndarray] | None,
    t: np.ndarray,
    P: np.ndarray,
    scale: Sequence[float],
    tol: float,
    checks: int = 6,
) -> dict:
    """Fit one run of samples ``(t, P)`` with as few Hermite Bezier segments as possible.

    Returns a Blender bezier spline dict (``co``/``hl``/``hr``) in the same
    coordinates as ``P``.  Guarantee: at the sample points of the run and at
    ``checks`` extra points per segment, the deviation from ``F`` is <= tol
    (measured after ``scale``); segments that cannot meet this fall back to
    straight lines between the (already tolerance-refined) samples.
    """
    t = np.asarray(t, float)
    P = np.asarray(P, float)
    n = len(t)
    s = np.asarray(scale, float)[: P.shape[1]]
    D = dF(t) if dF is not None else np.full_like(P, np.nan)
    # Replace non-finite derivatives (domain edges, cusps) with one-sided differences.
    bad = ~np.all(np.isfinite(D), axis=1)
    if bad.any():
        fd = np.gradient(P, t, axis=0, edge_order=1) if n > 1 else np.zeros_like(P)
        D[bad] = fd[bad]
        D_is_exact = ~bad
    else:
        D_is_exact = np.ones(n, dtype=bool)

    def seg_ok(i, j):
        dt = t[j] - t[i]
        if dt <= 0 or not (D_is_exact[i] and D_is_exact[j]):
            return False
        # interior samples + extra check points evaluated on the true curve
        tt = np.concatenate([t[i + 1:j], t[i] + dt * (np.arange(1, checks + 1) / (checks + 1))])
        if tt.size == 0:
            return True
        true = F(tt)
        if not np.all(np.isfinite(true)):
            return False
        u = (tt - t[i]) / dt
        H = _hermite_eval(P[i], P[j], D[i], D[j], dt, u)
        return float(np.max(np.linalg.norm((H - true) * s, axis=1))) <= tol

    co, hl, hr = [P[0]], [P[0]], []
    i = 0
    while i < n - 1:
        # exponential + binary search for the furthest acceptable knot
        j, step, best = i + 1, 1, None
        while j < n and seg_ok(i, j):
            best = j
            step *= 2
            j = i + step
        if best is not None and best < n - 1:
            lo_, hi_ = best, min(j, n - 1)
            while hi_ - lo_ > 1:
                mid = (lo_ + hi_) // 2
                if seg_ok(i, mid):
                    lo_ = mid
                else:
                    hi_ = mid
            best = lo_
        if best is None:  # straight segment to the next sample (vector handles)
            j = i + 1
            a, b = P[i], P[j]
            hr.append(a + (b - a) / 3)
            hl.append(b - (b - a) / 3)
        else:
            j = best
            dt = t[j] - t[i]
            hr.append(P[i] + D[i] * dt / 3)
            hl.append(P[j] - D[j] * dt / 3)
        co.append(P[j])
        i = j
    hr.append(P[-1])
    return {"co": np.asarray(co), "hl": np.asarray(hl), "hr": np.asarray(hr)}


def spline_to_blender(sp_: dict, to_local: Callable[[np.ndarray], np.ndarray], cyclic: bool = False,
                      ndigits: int = 7, max_len: float = 0.3) -> dict:
    """Map a bezier spline to local coordinates (affine maps keep Beziers exact) and subdivide
    long segments exactly, so Blender's fixed per-segment tessellation stays smooth."""
    co, hl, hr = (to_local(np.asarray(sp_[k], float)) for k in ("co", "hl", "hr"))
    co, hl, hr = subdivide(co, hl, hr, cyclic, max_len)
    r = lambda A: np.round(A, ndigits).tolist()  # noqa: E731
    return {"co": r(co), "hl": r(hl), "hr": r(hr), "cyclic": cyclic}


def subdivide(co, hl, hr, cyclic: bool = False, max_len: float = 0.3):
    """Exact de Casteljau subdivision of every segment whose control polygon is longer than max_len."""
    n = len(co)
    segs = n if cyclic else n - 1
    out_co, out_hl, out_hr = [co[0]], [hl[0]], []
    for i in range(segs):
        j = (i + 1) % n
        pieces = _split([co[i], hr[i], hl[j], co[j]], max_len)
        for k, (p0, p1, p2, p3) in enumerate(pieces):
            out_hr.append(p1)
            if cyclic and i == segs - 1 and k == len(pieces) - 1:
                out_hl[0] = p2  # closing segment ends at the first point
            else:
                out_co.append(p3)
                out_hl.append(p2)
    if not cyclic:
        out_hr.append(hr[-1])
    return np.array(out_co), np.array(out_hl), np.array(out_hr)


def _split(P, max_len, depth=0):
    p0, p1, p2, p3 = (np.asarray(p, float) for p in P)
    L = np.linalg.norm(p1 - p0) + np.linalg.norm(p2 - p1) + np.linalg.norm(p3 - p2)
    if L <= max_len or depth > 12:
        return [(p0, p1, p2, p3)]
    a, b, c = (p0 + p1) / 2, (p1 + p2) / 2, (p2 + p3) / 2
    d, e = (a + b) / 2, (b + c) / 2
    m = (d + e) / 2
    return _split((p0, a, d, m), max_len, depth + 1) + _split((m, e, c, p3), max_len, depth + 1)


# --------------------------------------------------------------------------- exact polynomial pieces


def cubic_piece_bezier(poly: sp.Expr, x: sp.Symbol, a, b) -> tuple[list, list]:
    """Exact Bezier control points of the graph of a polynomial (deg <= 3) on [a, b]."""
    p = sp.Poly(sp.expand(poly), x)
    if p.degree() > 3:
        raise ValueError("cubic_piece_bezier needs degree <= 3")
    a, b = sp.nsimplify(a), sp.nsimplify(b)
    h = b - a
    dp = sp.diff(poly, x)
    ya, yb = poly.subs(x, a), poly.subs(x, b)
    ctrl = [(a, ya), (a + h / 3, ya + h / 3 * dp.subs(x, a)), (b - h / 3, yb - h / 3 * dp.subs(x, b)), (b, yb)]
    return [(sp.nsimplify(cx), sp.simplify(cy)) for cx, cy in ctrl], [float(c) for pt in ctrl for c in pt]


def piecewise_graph_spline(pieces: Sequence[tuple[sp.Expr, object, object]], x: sp.Symbol) -> dict:
    """One continuous bezier spline for consecutive polynomial pieces (deg <= 3), exact."""
    co, hl, hr = [], [], []
    for k, (poly, a, b) in enumerate(pieces):
        ctrl, _ = cubic_piece_bezier(poly, x, a, b)
        c = np.array([[float(px), float(py)] for px, py in ctrl])
        if k == 0:
            co.append(c[0])
            hl.append(c[0])
        hr.append(c[1])
        hl.append(c[2])
        co.append(c[3])
    hr.append(co[-1])
    return {"co": np.array(co), "hl": np.array(hl), "hr": np.array(hr)}


# --------------------------------------------------------------------------- interpolation


def interpolating_spline(points: Sequence[Sequence], kind: str = "natural", end_slopes=None):
    """Exact cubic spline through points (rational arithmetic).

    kind: 'natural' (S''=0 at ends), 'clamped' (given end_slopes), 'not-a-knot',
    'periodic' (first and last y must be equal), or 'lagrange' (single polynomial).
    Returns ``[(poly_i(x), x_i, x_{i+1})]`` with exact SymPy polynomials.
    """
    x = sp.Symbol("x", real=True)
    pts = [(sp.nsimplify(p[0]), sp.nsimplify(p[1])) for p in points]
    pts.sort(key=lambda p: p[0])
    n = len(pts) - 1
    if n < 1:
        raise ExpressionError("Need at least two points.")
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    if len(set(xs)) != len(xs):
        raise ExpressionError("Interpolation points must have distinct x values (a function graph).")
    if kind == "lagrange":
        poly = sp.expand(sp.interpolate(list(zip(xs, ys)), x))
        return [(poly, xs[0], xs[-1])]
    if n == 1 and kind != "clamped":
        return [(sp.expand(ys[0] + (ys[1] - ys[0]) / (xs[1] - xs[0]) * (x - xs[0])), xs[0], xs[1])]
    h = [xs[i + 1] - xs[i] for i in range(n)]
    M = sp.symbols(f"M0:{n + 1}")  # second derivatives at knots
    eqs = []
    for i in range(1, n):
        eqs.append(h[i - 1] * M[i - 1] + 2 * (h[i - 1] + h[i]) * M[i] + h[i] * M[i + 1]
                   - 6 * ((ys[i + 1] - ys[i]) / h[i] - (ys[i] - ys[i - 1]) / h[i - 1]))
    if kind == "natural":
        eqs += [M[0], M[n]]
    elif kind == "clamped":
        if end_slopes is None or len(end_slopes) != 2:
            raise ExpressionError("clamped spline needs end_slopes [s0, sn].")
        s0, sn = (sp.nsimplify(v) for v in end_slopes)
        eqs.append(2 * h[0] * M[0] + h[0] * M[1] - 6 * ((ys[1] - ys[0]) / h[0] - s0))
        eqs.append(h[-1] * M[n - 1] + 2 * h[-1] * M[n] - 6 * (sn - (ys[n] - ys[n - 1]) / h[-1]))
    elif kind == "not-a-knot":
        if n < 3:
            return interpolating_spline(points, "lagrange")
        eqs.append(h[1] * (M[1] - M[0]) - h[0] * (M[2] - M[1]))
        eqs.append(h[-1] * (M[n - 1] - M[n - 2]) - h[-2] * (M[n] - M[n - 1]))
    elif kind == "periodic":
        if ys[0] != ys[-1]:
            raise ExpressionError("periodic spline needs the first and last y values to be equal.")
        eqs.append(M[0] - M[n])
        eqs.append(h[0] * M[1] + 2 * (h[0] + h[-1]) * M[0] + h[-1] * M[n - 1]
                   - 6 * ((ys[1] - ys[0]) / h[0] - (ys[n] - ys[n - 1]) / h[-1]))
    else:
        raise ExpressionError(f"Unknown spline kind {kind!r}")
    sol = sp.solve(eqs, M, dict=True)
    if not sol:
        raise ExpressionError("Spline system has no solution.")
    Mv = [sol[0][m] for m in M]
    pieces = []
    for i in range(n):
        a, b = xs[i], xs[i + 1]
        S = (Mv[i] * (b - x) ** 3 / (6 * h[i]) + Mv[i + 1] * (x - a) ** 3 / (6 * h[i])
             + (ys[i] / h[i] - Mv[i] * h[i] / 6) * (b - x) + (ys[i + 1] / h[i] - Mv[i + 1] * h[i] / 6) * (x - a))
        pieces.append((sp.expand(S), a, b))
    return pieces


def piecewise_latex(pieces, x=None) -> str:
    rows = [rf"{sp.latex(p)} & {sp.latex(a)} \le x \le {sp.latex(b)}" for p, a, b in pieces]
    if len(rows) == 1:
        return sp.latex(pieces[0][0])
    return r"\begin{cases} " + r" \\ ".join(rows) + r" \end{cases}"


# --------------------------------------------------------------------------- Bezier / B-spline math


def bernstein(n: int, i: int, t):
    return sp.binomial(n, i) * t ** i * (1 - t) ** (n - i)


def bezier_polynomial(ctrl: Sequence[Sequence], t: sp.Symbol) -> list[sp.Expr]:
    """Exact component polynomials of a Bezier curve of any degree."""
    P = [[sp.nsimplify(c) for c in p] for p in ctrl]
    n = len(P) - 1
    dims = len(P[0])
    return [sp.expand(sum(bernstein(n, i, t) * P[i][d] for i in range(n + 1))) for d in range(dims)]


def de_casteljau(ctrl: Sequence[Sequence], t0) -> list[list[list[sp.Expr]]]:
    """All levels of de Casteljau's algorithm at parameter t0 (exact).  Last level = curve point."""
    t0 = sp.nsimplify(t0)
    level = [[sp.nsimplify(c) for c in p] for p in ctrl]
    levels = [level]
    while len(level) > 1:
        level = [[(1 - t0) * a + t0 * b for a, b in zip(level[i], level[i + 1])] for i in range(len(level) - 1)]
        levels.append(level)
    return levels


def elevate_to_cubic(ctrl: Sequence[Sequence]) -> list[list[float]]:
    """Exact cubic control points for a Bezier curve of degree 1..3."""
    P = [np.asarray(p, float) for p in ctrl]
    if len(P) == 2:
        a, b = P
        return [a, a + (b - a) / 3, a + 2 * (b - a) / 3, b]
    if len(P) == 3:
        a, q, b = P
        return [a, a + 2 / 3 * (q - a), b + 2 / 3 * (q - b), b]
    if len(P) == 4:
        return P
    raise ValueError("degree > 3")


def clamped_uniform_knots(n_ctrl: int, degree: int) -> list[sp.Rational]:
    inner = n_ctrl - degree - 1
    if inner < 0:
        raise ExpressionError(f"A degree-{degree} B-spline needs at least {degree + 1} control points.")
    return [sp.Integer(0)] * (degree + 1) + [sp.Rational(i, inner + 1) for i in range(1, inner + 1)] + \
        [sp.Integer(1)] * (degree + 1)


def bspline_basis(degree: int, knots: Sequence, t: sp.Symbol) -> list[sp.Expr]:
    """Exact B-spline basis functions (Piecewise polynomials) for the given knots."""
    return sp.bspline_basis_set(degree, [sp.nsimplify(k) for k in knots], t)


def bspline_pieces(ctrl: Sequence[Sequence], degree: int, knots: Sequence, t: sp.Symbol):
    """Per knot span ``[(components, t_a, t_b)]`` with exact polynomial components."""
    basis = bspline_basis(degree, knots, t)
    if len(basis) != len(ctrl):
        raise ExpressionError(f"{len(ctrl)} control points need {len(ctrl) + degree + 1} knots for degree "
                              f"{degree} (got {len(knots)}).")
    K = [sp.nsimplify(k) for k in knots]
    spans = []
    for a, b in zip(K[degree:-degree - 1], K[degree + 1:-degree]):
        if a == b:
            continue
        mid = (a + b) / 2
        comps = []
        for d in range(len(ctrl[0])):
            expr = 0
            for Ni, P in zip(basis, ctrl):
                piece = _piece_at(Ni, t, mid)
                expr += piece * sp.nsimplify(P[d])
            comps.append(sp.expand(expr))
        spans.append((comps, a, b))
    return spans


def _piece_at(pw: sp.Expr, t: sp.Symbol, t0) -> sp.Expr:
    """The polynomial branch of a Piecewise that is active at t0."""
    if not isinstance(pw, sp.Piecewise):
        return pw
    for expr, cond in pw.args:
        if cond == True or bool(cond.subs(t, t0)):  # noqa: E712 - SymPy truth
            return expr
    return sp.Integer(0)


def parametric_piece_bezier(comps: Sequence[sp.Expr], t: sp.Symbol, a, b) -> list[np.ndarray]:
    """Exact cubic Bezier control points of a polynomial parametric piece (deg <= 3)."""
    for c in comps:
        if sp.Poly(c, t).degree() > 3:
            raise ValueError("degree > 3")
    a, b = sp.nsimplify(a), sp.nsimplify(b)
    h = b - a
    P0 = [c.subs(t, a) for c in comps]
    P3 = [c.subs(t, b) for c in comps]
    D0 = [sp.diff(c, t).subs(t, a) for c in comps]
    D3 = [sp.diff(c, t).subs(t, b) for c in comps]
    P1 = [p + h / 3 * d for p, d in zip(P0, D0)]
    P2 = [p - h / 3 * d for p, d in zip(P3, D3)]
    return [np.array([float(v) for v in P]) for P in (P0, P1, P2, P3)]
