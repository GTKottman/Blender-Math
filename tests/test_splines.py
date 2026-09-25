import numpy as np
import sympy as sp

from blender_math_mcp import sampling as smp
from blender_math_mcp import splines as spl
from blender_math_mcp.expressions import compile_numeric, parse

X = sp.Symbol("x", real=True)
T = sp.Symbol("t", real=True)


def bez(p0, p1, p2, p3, u):
    u = u[:, None]
    return (1 - u) ** 3 * p0 + 3 * (1 - u) ** 2 * u * p1 + 3 * (1 - u) * u ** 2 * p2 + u ** 3 * p3


def fit(expr, a, b, tol=1e-3):
    e = parse(expr)
    f = compile_numeric(e, ["x"])
    d = compile_numeric(sp.diff(e, X), ["x"])
    F = lambda t: np.stack([t, f(t)], 1)  # noqa: E731
    dF = lambda t: np.stack([np.ones_like(t), d(t)], 1)  # noqa: E731
    (t, P), = smp.sample_curve_t(F, a, b, [1, 1], tol)
    return spl.hermite_bezier(F, dF, t, P, [1, 1], tol), f, len(P)


def max_err(bz, f):
    err = 0
    for k in range(len(bz["co"]) - 1):
        B = bez(bz["co"][k], bz["hr"][k], bz["hl"][k + 1], bz["co"][k + 1], np.linspace(0, 1, 300))
        err = max(err, np.max(np.abs(B[:, 1] - f(B[:, 0]))))
    return err


def test_hermite_bezier_within_tolerance_and_compact():
    bz, f, n = fit("sin(x)", -10, 10)
    assert max_err(bz, f) <= 1e-3
    assert len(bz["co"]) < n / 10  # far fewer knots than polyline points


def test_cubic_is_exact():
    bz, f, _ = fit("x^3 - x", -2, 2)
    assert len(bz["co"]) == 2 and max_err(bz, f) < 1e-12


def test_subdivision_is_exact():
    co = np.array([[0, 0, 0], [4, 0, 0.0]])
    hl = np.array([[0, 0, 0], [3, 1, 0.0]])
    hr = np.array([[1, 1, 0], [4, 0, 0.0]])
    c, left, right = spl.subdivide(co, hl, hr, False, 0.3)
    orig = lambda u: bez(co[0], hr[0], hl[1], co[1], u)  # noqa: E731
    # the k-th of 2^m equal pieces covers u in [k/2^m, (k+1)/2^m]
    n = len(c) - 1
    for k in range(n):
        u = np.linspace(0, 1, 7)
        piece = bez(c[k], right[k], left[k + 1], c[k + 1], u)
        assert np.allclose(piece, orig((k + u) / n), atol=1e-12)


def test_natural_spline_exact_and_smooth():
    pieces = spl.interpolating_spline([(0, 0), (1, 1), (2, 0), (3, 1)], "natural")
    assert pieces[0][0] == -2 * X ** 3 / 3 + 5 * X / 3
    for (p, _, b), (q, _, _) in zip(pieces, pieces[1:]):  # C2 continuity at knots
        for k in range(3):
            assert sp.simplify(sp.diff(p, X, k).subs(X, b) - sp.diff(q, X, k).subs(X, b)) == 0
    assert sp.diff(pieces[0][0], X, 2).subs(X, 0) == 0  # natural end condition


def test_de_casteljau_point_is_on_curve():
    ctrl = [(0, 0), (1, 2), (3, 2), (4, 0)]
    levels = spl.de_casteljau(ctrl, sp.Rational(2, 5))
    comps = spl.bezier_polynomial(ctrl, T)
    assert [c.subs(T, sp.Rational(2, 5)) for c in comps] == levels[-1][0]


def test_bspline_matches_de_boor():
    ctrl = [[3.5, -3], [4, 1], [5, 3], [6, -1], [6.5, 2.5], [5.5, -3]]
    p = 3
    kn = spl.clamped_uniform_knots(len(ctrl), p)
    spans = spl.bspline_pieces(ctrl, p, kn, T)
    K = [float(k) for k in kn]

    def de_boor(tt):
        k = max(i for i in range(len(K) - 1) if K[i] <= tt < K[i + 1])
        d = [np.array(ctrl[j + k - p], float) for j in range(p + 1)]
        for r in range(1, p + 1):
            for j in range(p, r - 1, -1):
                a = (tt - K[j + k - p]) / (K[j + 1 + k - r] - K[j + k - p])
                d[j] = (1 - a) * d[j - 1] + a * d[j]
        return d[p]

    for comps, a, b in spans:
        for tt in np.linspace(float(a), float(b), 20)[:-1]:
            v = np.array([float(c.subs(T, tt)) for c in comps])
            assert np.allclose(v, de_boor(tt), atol=1e-12)
