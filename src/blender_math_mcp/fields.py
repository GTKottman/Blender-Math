"""Differential equations and fields: exact solutions first, high-accuracy numerics otherwise."""

from __future__ import annotations

from typing import Callable

import numpy as np
import sympy as sp

from .keypoints import MathRefusal, with_timeout


def solve_ivp_exact(rhs: sp.Expr, x: sp.Symbol, y: sp.Symbol, x0, y0, timeout: float = 10.0):
    """Exact solution y(x) of y' = rhs(x, y), y(x0) = y0, verified with checkodesol.  None if not found."""
    Y = sp.Function("Y")
    ode = sp.Eq(Y(x).diff(x), rhs.subs(y, Y(x)))

    def run():
        sols = sp.dsolve(ode, Y(x), ics={Y(sp.nsimplify(x0)): sp.nsimplify(y0)})
        return sols if isinstance(sols, list) else [sols]

    sols = with_timeout(run, timeout)
    if not sols:
        return None
    for s in sols:
        if not isinstance(s, sp.Eq) or s.lhs != Y(x) or s.rhs.has(Y):
            continue
        ok = with_timeout(lambda s=s: sp.checkodesol(ode, s), 8.0)
        if ok and ok[0] is True and sp.simplify(s.rhs.subs(x, sp.nsimplify(x0)) - sp.nsimplify(y0)) == 0:
            return s.rhs
    return None


def dopri45(f: Callable[[float, np.ndarray], np.ndarray], t0: float, y0, t1: float,
            rtol: float = 1e-10, atol: float = 1e-12, max_steps: int = 200_000, bound: float = 1e8,
            max_step: float | None = None):
    """Dormand-Prince 5(4) with adaptive steps.  Integrates from t0 towards t1 (either direction).

    Returns ``(t, Y, note)``; stops early (with a note) at blow-up or where f is undefined.
    """
    c = np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1])
    A = [
        [],
        [1 / 5],
        [3 / 40, 9 / 40],
        [44 / 45, -56 / 15, 32 / 9],
        [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
        [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
        [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
    ]
    b5 = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0])
    b4 = np.array([5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40])
    y = np.atleast_1d(np.asarray(y0, float))
    t = float(t0)
    direction = 1.0 if t1 >= t0 else -1.0
    h = direction * min(abs(t1 - t0) / 100, 1e-2) if t1 != t0 else 0.0
    ts, ys = [t], [y.copy()]
    note = None
    for _ in range(max_steps):
        if direction * (t1 - t) <= 1e-14 * max(1, abs(t1)):
            break
        if max_step is not None and abs(h) > max_step:
            h = direction * max_step
        if direction * (t + h - t1) > 0:
            h = t1 - t
        K = []
        ok = True
        for i in range(7):
            yi = y + h * sum(A[i][j] * K[j] for j in range(i)) if i else y
            k = np.atleast_1d(np.asarray(f(t + c[i] * h, yi), float))
            if not np.all(np.isfinite(k)):
                ok = False
                break
            K.append(k)
        if not ok:
            h *= 0.25
            if abs(h) < 1e-12 * max(1.0, abs(t)):
                note = f"stopped at t = {t:.10g}: the equation is undefined or singular there"
                break
            continue
        K = np.array(K)
        y5 = y + h * (b5 @ K)
        y4 = y + h * (b4 @ K)
        sc = atol + rtol * np.maximum(np.abs(y), np.abs(y5))
        err = np.sqrt(np.mean(((y5 - y4) / sc) ** 2))
        if err <= 1.0:
            t += h
            y = y5
            ts.append(t)
            ys.append(y.copy())
            if np.max(np.abs(y)) > bound:
                note = f"solution blows up near t = {t:.10g}"
                break
        fac = 0.9 * (1.0 / max(err, 1e-10)) ** 0.2
        h *= min(5.0, max(0.2, fac))
        if abs(h) < 1e-14 * max(1.0, abs(t)):
            note = f"step size underflow at t = {t:.10g} (singularity)"
            break
    return np.array(ts), np.array(ys), note


def require_real_matrix(M) -> sp.Matrix:
    m = sp.Matrix(M)
    if any(v.is_real is False for v in m):
        raise MathRefusal("Matrix entries must be real.")
    return m


def eigen_real(M: sp.Matrix):
    """Exact real eigenvalues with eigenvector bases; complex ones are reported, not drawn."""
    out, complex_ = [], []
    for val, mult, vecs in M.eigenvects():
        if val.is_real is False or sp.im(sp.N(val)) != 0:
            complex_.append(val)
            continue
        out.append((sp.simplify(val), mult, [sp.simplify(v) for v in vecs]))
    return out, complex_
