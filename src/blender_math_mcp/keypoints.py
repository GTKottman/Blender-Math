"""Exact special points: roots, intersections, extrema, inflection points.

Strategy (never guess):

1. **Exact**: ``solveset`` over the interval.  Every solution is checked by
   substituting it back (symbolic simplification, or 50-digit evaluation).
2. **Certified numeric**: dense sampling finds sign changes; each is refined by a
   bracketing solver at 50 digits.  A sign change of a *continuous* function
   proves a root exists in the bracket (intermediate value theorem); brackets
   whose values blow up are poles and are rejected.  Tangential roots (no sign
   change) are found as minima of |f| and accepted only when |f| < 1e-30.
3. **Identification**: a numeric root is replaced by a closed form
   (``nsimplify``) only if substituting that closed form gives exactly 0.

Every result says how it was obtained (``method``) and whether it is exact.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

import mpmath
import numpy as np
import sympy as sp

from .expressions import ExpressionError, compile_numeric

DPS = 50


class MathRefusal(ValueError):
    """A request that would produce mathematically wrong output."""


def with_timeout(fn: Callable, seconds: float, default=None):
    """Run ``fn`` in a daemon thread; return ``default`` if it takes too long."""
    box = {}

    def run():
        try:
            box["v"] = fn()
        except Exception as exc:  # noqa: BLE001 - reported by returning default
            box["e"] = exc

    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(seconds)
    if th.is_alive() or "e" in box:
        return default
    return box.get("v", default)


@dataclass
class KeyPoint:
    x: sp.Expr
    y: sp.Expr
    kind: str
    exact: bool
    method: str

    @property
    def xf(self) -> float:
        return float(sp.N(self.x, 20))

    @property
    def yf(self) -> float:
        return float(sp.N(self.y, 20))

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "x": str(self.x), "y": str(self.y),
            "x_latex": sp.latex(self.x), "y_latex": sp.latex(self.y),
            "x_value": self.xf, "y_value": self.yf,
            "exact": self.exact, "method": self.method,
        }


def _is_zero(expr: sp.Expr, var: sp.Symbol, x0: sp.Expr) -> bool | None:
    """True if expr(x0) == 0 (proved or to 50 digits), False if not, None if unknown."""
    val = with_timeout(lambda: sp.simplify(expr.subs(var, x0)), 5.0)
    if val is not None and val == 0:
        return True
    try:
        num = sp.N(expr.subs(var, x0), DPS)
        if num.is_real is False and abs(sp.im(num)) > 1e-40:
            return False
        v = abs(complex(num))
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return False
    return v < 1e-40


def _in_interval(x0, a, b) -> bool:
    try:
        v = float(sp.N(x0, 30))
    except (TypeError, ValueError):
        return False
    return a - 1e-12 <= v <= b + 1e-12


def exact_zeros(expr: sp.Expr, var: sp.Symbol, a: float, b: float, timeout: float = 8.0):
    """Exact zeros in [a, b] or None if SymPy cannot solve it (in time)."""
    dom = sp.Interval(sp.nsimplify(a), sp.nsimplify(b))
    sol = with_timeout(lambda: sp.solveset(expr, var, domain=dom), timeout)
    if sol is None:
        return None
    if sol is sp.S.EmptySet:
        return []
    if isinstance(sol, sp.Interval) or (isinstance(sol, sp.Union) and any(isinstance(s, sp.Interval)
                                                                          for s in sol.args)):
        raise MathRefusal(f"{expr} is zero on a whole interval ({sol}); there are no isolated points.")
    if not isinstance(sol, sp.FiniteSet):
        return None
    out = []
    for r in sol:
        if r.is_real is False or not _in_interval(r, a, b):
            continue
        ok = _is_zero(expr, var, r)
        if ok:
            out.append(r)
    return sorted(out, key=lambda r: float(sp.N(r, 30)))


def numeric_zeros(expr: sp.Expr, var: sp.Symbol, a: float, b: float, samples: int = 4001) -> list[tuple]:
    """Certified numeric zeros: list of ``(mpf root, method)``."""
    f = compile_numeric(expr, [var.name])
    fm = sp.lambdify(var, expr, modules="mpmath")
    xs = np.linspace(a, b, samples)
    ys = f(xs)
    out = []

    def mp_eval(x):
        with mpmath.workdps(DPS):
            try:
                v = fm(mpmath.mpf(x))
                return v if mpmath.im(v) == 0 else mpmath.nan
            except Exception:  # noqa: BLE001 - undefined here
                return mpmath.nan

    with mpmath.workdps(DPS):
        # exact zeros at sample points
        for i in np.nonzero(ys == 0)[0]:
            v = mp_eval(xs[i])
            if v == 0:
                out.append((mpmath.mpf(xs[i]), "exact sample"))
        # sign changes
        for i in range(len(xs) - 1):
            y0, y1 = ys[i], ys[i + 1]
            if not (np.isfinite(y0) and np.isfinite(y1)) or y0 == 0 or y1 == 0 or np.sign(y0) == np.sign(y1):
                continue
            lo, hi = mpmath.mpf(xs[i]), mpmath.mpf(xs[i + 1])
            flo = mp_eval(lo)
            for _ in range(200):  # bisection to 50 digits (robust, no derivative needed)
                mid = (lo + hi) / 2
                fm_ = mp_eval(mid)
                if not mpmath.isfinite(fm_):
                    break
                if fm_ == 0:
                    lo = hi = mid
                    break
                if mpmath.sign(fm_) == mpmath.sign(flo):
                    lo, flo = mid, fm_
                else:
                    hi = mid
                if hi - lo < mpmath.mpf(10) ** (-(DPS - 5)) * max(1, abs(lo)):
                    break
            root = (lo + hi) / 2
            val = mp_eval(root)
            # Reject poles / jumps: at a real root |f| shrinks to ~0.
            if mpmath.isfinite(val) and abs(val) < mpmath.mpf(10) ** -20:
                out.append((root, "sign change (intermediate value theorem), bisection to 50 digits"))
        # tangential zeros: local minima of |f| that are not sign changes
        ay = np.abs(ys)
        try:
            fd = sp.lambdify(var, sp.diff(expr, var), modules="mpmath")
        except Exception:  # noqa: BLE001 - derivative not numerically evaluable
            fd = None
        for i in range(1, len(xs) - 1):
            if fd is None:
                break
            if not np.all(np.isfinite(ay[i - 1:i + 2])):
                continue
            if ay[i] <= ay[i - 1] and ay[i] <= ay[i + 1] and ay[i] > 0 and np.sign(ys[i - 1]) == np.sign(ys[i + 1]):
                try:
                    r = mpmath.findroot(fd, mpmath.mpf(xs[i]))
                except Exception:  # noqa: BLE001 - no convergence / not evaluable
                    continue
                if a <= r <= b:
                    v = mp_eval(r)
                    if mpmath.isfinite(v) and abs(v) < mpmath.mpf(10) ** -30:
                        out.append((r, "tangential zero (|f| < 1e-30 at a critical point), 50-digit Newton"))
    # dedupe
    out.sort(key=lambda t: t[0])
    dedup = []
    for r, m in out:
        if not dedup or abs(r - dedup[-1][0]) > mpmath.mpf(10) ** -12 * max(1, abs(r)):
            dedup.append((r, m))
    return dedup


def _identify(expr, var, r) -> sp.Expr | None:
    """Closed form for a numeric root, accepted only if it is *proved* to be a root."""
    with mpmath.workdps(DPS):
        f = sp.Float(mpmath.nstr(r, DPS), DPS)
    cand = with_timeout(lambda: sp.nsimplify(f, [sp.pi, sp.E], tolerance=sp.Float("1e-35")), 3.0)
    if cand is None or cand.count_ops() > 12:
        return None
    if abs(sp.N(cand - f, DPS)) > 1e-35:
        return None
    val = with_timeout(lambda: sp.simplify(expr.subs(var, cand)), 4.0)
    return cand if val is not None and val == 0 else None


def zeros(expr: sp.Expr, var: sp.Symbol, a: float, b: float, kind: str = "root") -> list[tuple[sp.Expr, bool, str]]:
    """All zeros of ``expr`` in [a, b] as ``(value, exact, method)``."""
    exact = exact_zeros(expr, var, a, b)
    numeric = numeric_zeros(expr, var, a, b)
    results: list[tuple[sp.Expr, bool, str]] = []
    if exact is not None:
        results = [(r, True, "exact (solveset, verified by substitution)") for r in exact]
    # Numeric roots the exact solver did not report (or everything, if it failed).
    for r, m in numeric:
        if any(abs(float(sp.N(e[0], 30)) - float(r)) < 1e-9 * max(1.0, abs(float(r))) for e in results):
            continue
        ident = _identify(expr, var, r)
        if ident is not None:
            results.append((ident, True, "exact (identified closed form, proved by substitution)"))
        else:
            results.append((sp.Float(mpmath.nstr(r, 30), 30), False, m))
    results.sort(key=lambda t: float(sp.N(t[0], 30)))
    return results


def _value(expr, var, x0, exact: bool) -> sp.Expr:
    y = expr.subs(var, x0)
    if exact:
        s = with_timeout(lambda: sp.nsimplify(sp.simplify(y)) if y.is_number and not y.free_symbols
                         else sp.simplify(y), 5.0)
        if s is not None and s.is_real is not False:
            # nsimplify may round; only accept if equal
            if abs(sp.N(s - y, 40)) < 1e-35:
                return sp.radsimp(s) if s.is_Pow or s.is_Mul else s
        y = sp.simplify(y) if with_timeout(lambda: sp.simplify(y), 5.0) is not None else y
        return y
    return sp.Float(sp.N(y, 30), 30)


def roots(expr, var, a, b) -> list[KeyPoint]:
    return [KeyPoint(r, sp.Integer(0), "root", ex, m) for r, ex, m in zeros(expr, var, a, b)]


def intersections(f, g, var, a, b) -> list[KeyPoint]:
    out = []
    for r, ex, m in zeros(f - g, var, a, b):
        out.append(KeyPoint(r, _value(f, var, r, ex), "intersection", ex, m))
    return out


def _sign_near(expr, var, x0, side: int) -> int:
    fm = sp.lambdify(var, expr, modules="mpmath")
    with mpmath.workdps(DPS):
        x = mpmath.mpf(str(sp.N(x0, DPS)))
        for k in (12, 9, 6):
            h = mpmath.mpf(10) ** (-k) * max(1, abs(x))
            try:
                v = fm(x + side * h)
            except Exception:  # noqa: BLE001 - undefined at this offset
                continue
            if mpmath.isfinite(v) and mpmath.im(v) == 0 and v != 0:
                return int(mpmath.sign(v))
    return 0


def extrema(expr, var, a, b, include_endpoints: bool = False) -> list[KeyPoint]:
    """Local maxima/minima (first-derivative test at 50 digits)."""
    d = sp.diff(expr, var)
    out = []
    for r, ex, m in zeros(d, var, a, b):
        left, right = _sign_near(d, var, r, -1), _sign_near(d, var, r, +1)
        if left > 0 and right < 0:
            kind = "local_max"
        elif left < 0 and right > 0:
            kind = "local_min"
        else:
            continue  # stationary but not an extremum (e.g. x^3 at 0)
        out.append(KeyPoint(r, _value(expr, var, r, ex), kind, ex, m + "; first-derivative test"))
    # corners (f continuous, f' jumps sign) e.g. |x| at 0
    for x0 in _derivative_sign_jumps(d, var, a, b):
        if any(abs(k.xf - float(x0)) < 1e-9 for k in out):
            continue
        fv = expr.subs(var, x0)
        if not sp.N(fv).is_finite:
            continue
        left, right = _sign_near(d, var, x0, -1), _sign_near(d, var, x0, +1)
        kind = "local_max" if (left > 0 and right < 0) else "local_min" if (left < 0 and right > 0) else None
        if kind:
            out.append(KeyPoint(x0, _value(expr, var, x0, True), kind, True,
                                "corner: f' changes sign across a point where f is continuous"))
    if include_endpoints:
        for e in (a, b):
            ev = sp.nsimplify(e)
            out.append(KeyPoint(ev, _value(expr, var, ev, True), "endpoint", True, "interval endpoint"))
    out.sort(key=lambda k: k.xf)
    return out


def _derivative_sign_jumps(d, var, a, b):
    """Points where f' jumps (not through 0), found exactly via singularities when possible."""
    sing = with_timeout(lambda: sp.calculus.singularities(d, var, sp.Interval(sp.nsimplify(a), sp.nsimplify(b))), 5.0)
    pts = []
    if isinstance(sing, sp.FiniteSet):
        pts += list(sing)
    # sign(x) / Abs derivatives: SymPy reports no singularity -> look for sign()/Heaviside arguments
    for s in d.atoms(sp.sign, sp.Heaviside):
        sol = with_timeout(lambda s=s: sp.solveset(s.args[0], var, sp.Interval(sp.nsimplify(a), sp.nsimplify(b))), 5.0)
        if isinstance(sol, sp.FiniteSet):
            pts += list(sol)
    return [p for p in pts if _in_interval(p, a, b)]


def inflections(expr, var, a, b) -> list[KeyPoint]:
    d2 = sp.diff(expr, var, 2)
    out = []
    for r, ex, m in zeros(d2, var, a, b):
        if _sign_near(d2, var, r, -1) * _sign_near(d2, var, r, +1) < 0:
            out.append(KeyPoint(r, _value(expr, var, r, ex), "inflection", ex, m + "; f'' changes sign"))
    return out


# --------------------------------------------------------------------------- 2D systems


def solve_system_box(P: sp.Expr, Q: sp.Expr, x: sp.Symbol, y: sp.Symbol, box, grid=(28, 16)) -> list[tuple]:
    """All solutions of P = Q = 0 inside box = (x0, x1, y0, y1).

    SymPy's ``solve`` only returns principal solutions of periodic systems (e.g. sin x = 0
    gives 0 and pi), so every candidate is also searched numerically: Newton's method at
    50 digits from a grid of starting points.  Each solution is identified in closed form
    when substituting the closed form gives exactly zero.  Returns ``(point, exact, method)``.
    """
    x0, x1, y0, y1 = box
    found: list[tuple] = []

    def add(px, py, exact, method):
        fx, fy = float(sp.N(px, 20)), float(sp.N(py, 20))
        if not (x0 - 1e-9 <= fx <= x1 + 1e-9 and y0 - 1e-9 <= fy <= y1 + 1e-9):
            return
        for q in found:
            if abs(float(sp.N(q[0][0])) - fx) < 1e-8 and abs(float(sp.N(q[0][1])) - fy) < 1e-8:
                return
        found.append(([px, py], exact, method))

    sols = with_timeout(lambda: sp.solve([P, Q], [x, y], dict=True), 3.0) or []
    for s_ in sols:
        if x in s_ and y in s_ and s_[x].is_real and s_[y].is_real:
            add(s_[x], s_[y], True, "exact (solve)")
    # 1) vectorised float Newton from a grid of starts
    nx, ny = grid
    try:
        fnp = sp.lambdify((x, y), [P, Q, sp.diff(P, x), sp.diff(P, y), sp.diff(Q, x), sp.diff(Q, y)], modules="numpy")
    except Exception:  # noqa: BLE001
        fnp = None
    cands = []
    if fnp is not None:
        GX, GY = np.meshgrid(np.linspace(x0, x1, nx + 1), np.linspace(y0, y1, ny + 1))
        X_, Y_ = GX.ravel().astype(float), GY.ravel().astype(float)
        with np.errstate(all="ignore"):
            for _ in range(60):
                p_, q_, a, b, c, d = (np.broadcast_to(np.asarray(v, float), X_.shape) for v in fnp(X_, Y_))
                det = a * d - b * c
                dx = (d * p_ - b * q_) / det
                dy = (-c * p_ + a * q_) / det
                X_, Y_ = X_ - dx, Y_ - dy
            p_, q_ = (np.broadcast_to(np.asarray(v, float), X_.shape) for v in fnp(X_, Y_)[:2])
        ok = np.isfinite(X_) & np.isfinite(Y_) & (np.abs(p_) < 1e-9) & (np.abs(q_) < 1e-9) & \
            (X_ >= x0 - 1e-6) & (X_ <= x1 + 1e-6) & (Y_ >= y0 - 1e-6) & (Y_ <= y1 + 1e-6)
        for vx, vy in zip(X_[ok], Y_[ok]):
            if not any(abs(vx - a_) < 1e-7 and abs(vy - b_) < 1e-7 for a_, b_ in cands):
                cands.append((vx, vy))
    # 2) polish at 50 digits and identify
    fP = sp.lambdify((x, y), P, modules="mpmath")
    fQ = sp.lambdify((x, y), Q, modules="mpmath")
    J = [[sp.lambdify((x, y), sp.diff(e, v), modules="mpmath") for v in (x, y)] for e in (P, Q)]
    with mpmath.workdps(DPS):
        for sx, sy in cands:
            if any(abs(float(sp.N(q[0][0])) - sx) < 1e-6 and abs(float(sp.N(q[0][1])) - sy) < 1e-6 for q in found):
                continue
            try:
                r = mpmath.findroot(lambda a, b: [fP(a, b), fQ(a, b)], (mpmath.mpf(sx), mpmath.mpf(sy)),
                                    J=lambda a, b: [[J[0][0](a, b), J[0][1](a, b)], [J[1][0](a, b), J[1][1](a, b)]],
                                    tol=mpmath.mpf(10) ** -40, maxsteps=60)
            except Exception:  # noqa: BLE001 - singular Jacobian etc.
                continue
            rx, ry = r[0], r[1]
            if abs(fP(rx, ry)) > 1e-35 or abs(fQ(rx, ry)) > 1e-35:
                continue
            ex_ = []
            for v in (rx, ry):
                f = sp.Float(mpmath.nstr(v, DPS), DPS)
                cand = with_timeout(lambda f=f: sp.nsimplify(f, [sp.pi, sp.E], tolerance=sp.Float("1e-35")), 3.0)
                ex_.append(cand if cand is not None and cand.count_ops() <= 12 and abs(sp.N(cand - f, DPS)) < 1e-35
                           else None)
            if all(e is not None for e in ex_):
                subs = {x: ex_[0], y: ex_[1]}
                ok_ = with_timeout(lambda: sp.simplify(P.subs(subs)) == 0 and sp.simplify(Q.subs(subs)) == 0, 5.0)
                if ok_:
                    add(ex_[0], ex_[1], True, "exact (identified closed form, proved by substitution)")
                    continue
            add(sp.Float(mpmath.nstr(rx, 30), 30), sp.Float(mpmath.nstr(ry, 30), 30), False,
                "Newton at 50 digits (|residual| < 1e-35)")
    found.sort(key=lambda q: (float(sp.N(q[0][0])), float(sp.N(q[0][1]))))
    return found


# --------------------------------------------------------------------------- validation


def require_defined(expr, var, x0) -> sp.Expr:
    """f(x0) as an exact real number, or refuse with the reason."""
    val = with_timeout(lambda: sp.simplify(expr.subs(var, x0)), 5.0)
    if val is None:
        val = expr.subs(var, x0)
    num = sp.N(val, 30)
    if num.has(sp.zoo, sp.nan, sp.oo, -sp.oo) or not num.is_finite:
        raise MathRefusal(f"{expr} is not defined at {var} = {x0} (value {val}).")
    if num.is_real is False or (num.is_number and abs(sp.im(num)) > 1e-25):
        raise MathRefusal(f"{expr} is not real at {var} = {x0} (value {num}).")
    return val


def require_differentiable(expr, var, x0) -> sp.Expr:
    """f'(x0) exactly, or refuse (undefined, corner, cusp, vertical tangent)."""
    require_defined(expr, var, x0)
    h = sp.Symbol("h", positive=True)
    q = (expr.subs(var, x0 + h) - expr.subs(var, x0)) / h
    qm = (expr.subs(var, x0 - h) - expr.subs(var, x0)) / (-h)
    right = with_timeout(lambda: sp.limit(q, h, 0), 8.0)
    left = with_timeout(lambda: sp.limit(qm, h, 0), 8.0)
    if right is None or left is None:
        # numeric one-sided difference quotients at 50 digits
        fm = sp.lambdify(var, expr, modules="mpmath")
        with mpmath.workdps(DPS):
            x = mpmath.mpf(str(sp.N(x0, DPS)))
            hs = [mpmath.mpf(10) ** -k for k in (10, 15, 20)]
            r = [(fm(x + hh) - fm(x)) / hh for hh in hs]
            l_ = [(fm(x) - fm(x - hh)) / hh for hh in hs]
            if abs(r[-1] - r[-2]) > 1e-6 * max(1, abs(r[-1])) or abs(l_[-1] - l_[-2]) > 1e-6 * max(1, abs(l_[-1])):
                raise MathRefusal(f"{expr} does not appear differentiable at {var} = {x0} "
                                  "(difference quotients do not converge).")
            if abs(r[-1] - l_[-1]) > 1e-6 * max(1, abs(r[-1])):
                raise MathRefusal(f"{expr} is not differentiable at {var} = {x0}: left derivative ≈ "
                                  f"{mpmath.nstr(l_[-1], 8)}, right derivative ≈ {mpmath.nstr(r[-1], 8)}.")
        d = sp.diff(expr, var).subs(var, x0)
        return sp.simplify(d)
    if right.has(sp.oo, -sp.oo, sp.zoo) or left.has(sp.oo, -sp.oo, sp.zoo):
        raise MathRefusal(f"{expr} has a vertical tangent or cusp at {var} = {x0} (one-sided derivatives "
                          f"{left} and {right}); a tangent line y = mx + b does not exist there.")
    if sp.simplify(left - right) != 0:
        raise MathRefusal(f"{expr} is not differentiable at {var} = {x0}: left derivative {left}, "
                          f"right derivative {right}.")
    return sp.simplify(right)


__all__ = ["KeyPoint", "MathRefusal", "roots", "intersections", "extrema", "inflections", "zeros",
           "require_defined", "require_differentiable", "with_timeout", "ExpressionError"]
