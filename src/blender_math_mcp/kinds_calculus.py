"""Calculus node kinds: tangent/normal/secant lines, Riemann sums, areas, Taylor polynomials,
derivatives, antiderivatives, asymptotes."""

from __future__ import annotations

import mpmath
import numpy as np
import sympy as sp

from . import geometry as geo
from . import keypoints as kp
from . import sampling as smp
from .construction import Construction, Kind, Node, X, fnum, nice_latex, register
from .keypoints import MathRefusal, with_timeout
from .kinds_core import FunctionLike, graph_label, numeric
from .render import Frame, dash_polyline

# --------------------------------------------------------------------------- lines


def clip_line(P0, D, fr: Frame):
    """Exact clip of the infinite line P0 + s D to the axes box (2D).  Returns 2 points or None."""
    lo, hi = fr.lo[:2], fr.hi[:2]
    s0, s1 = -np.inf, np.inf
    for i in range(2):
        if abs(D[i]) < 1e-300:
            if not lo[i] <= P0[i] <= hi[i]:
                return None
            continue
        a, b = (lo[i] - P0[i]) / D[i], (hi[i] - P0[i]) / D[i]
        s0, s1 = max(s0, min(a, b)), min(s1, max(a, b))
    if s0 >= s1:
        return None
    return np.array([P0 + s0 * D, P0 + s1 * D])


def line_objects(c: Construction, node: Node, geom, fr: Frame, extent: str = "line") -> tuple[list, np.ndarray | None]:
    """Draw a sympy line / ray / segment exactly (clipped to the axes)."""
    p1 = np.array([fnum(geom.p1.x), fnum(geom.p1.y)])
    p2 = np.array([fnum(geom.p2.x), fnum(geom.p2.y)])
    D = p2 - p1
    if isinstance(geom, sp.Segment2D):
        pts = np.array([p1, p2])
    else:
        seg = clip_line(p1, D, fr)
        if seg is None:
            return [], None
        if isinstance(geom, sp.Ray2D):
            # keep the part in direction D from p1
            s = [(q - p1) @ D / (D @ D) for q in seg]
            s = [max(0.0, v) for v in s]
            if s[0] == s[1]:
                return [], None
            pts = np.array([p1 + s[0] * D, p1 + s[1] * D])
        else:
            pts = seg
    from .kinds_core import draw_curve

    spline = geo.polyline_spline(fr.to_local(pts, "curve"))
    return draw_curve(c, node, [spline], fr), pts


def line_equation_latex(geom) -> str:
    a, b, cc = (sp.nsimplify(v) for v in sp.Line2D(geom.p1, geom.p2).coefficients)  # rays/segments too
    if b == 0:
        return f"x = {nice_latex(sp.simplify(-cc / a))}"
    return f"y = {nice_latex(sp.expand(-(a * sp.Symbol('x') + cc) / b))}"


class LineLike(FunctionLike):
    """A straight line that is also a function (non-vertical) so it can be intersected like one."""

    def provides(self, node):
        e = node.data.get("expr")
        return {node.name: sp.Lambda(X, e)} if e is not None else {}

    def draw(self, c, node):
        fr = c.frame(node.spec["axes"])
        objs, pts = line_objects(c, node, node.data["geom"], fr)
        if pts is not None and node.style.get("label"):
            lab = node.style["label"]
            lab = line_equation_latex(node.data["geom"]) if lab is True else lab
            objs += graph_label(c, node, [pts[np.argsort(pts[:, 0])]], fr, lab)
        return objs + self.extra_draw(c, node, fr)

    def extra_draw(self, c, node, fr):
        return []

    def describe(self, node):
        g = node.data.get("geom")
        out = {"equation": line_equation_latex(g) if g is not None else None}
        if "slope" in node.data:
            out["slope"] = str(node.data["slope"])
        return out


def _line_from(p, m) -> tuple[sp.Expr, object]:
    x0, y0 = p
    expr = sp.expand(m * (X - x0) + y0)
    return expr, sp.Line2D(sp.Point2D(x0, y0), sp.Point2D(x0 + 1, y0 + m))


def _touch_point(c, node) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    s = node.spec
    f = c.function_of(s["function"])
    if "point" in s:
        P = c.point_of(s["point"])
        x0 = P[0]
        y0 = kp.require_defined(f, X, x0)
        if sp.simplify(y0 - P[1]) != 0 and abs(fnum(y0 - P[1])) > 1e-12:
            raise MathRefusal(f"Point {s['point']} = ({P[0]}, {P[1]}) is not on {s['function']} "
                              f"({s['function']}({P[0]}) = {y0}).")
    else:
        x0 = c.value(s["x"])
        y0 = kp.require_defined(f, X, x0)
    return f, x0, y0


def _mark(c, node, fr, P, name_suffix="pt"):
    if not node.style.get("show_point", True):
        return []
    loc = fr.to_local(np.asarray([fnum(v) for v in P], float), "point")[0]
    r = node.style.get("radius", 0.06)
    return [c.r.filled(f"{node.name}:{name_suffix}", [geo.bezier_circle([0, 0, 0], r)], fr, c.color(node),
                       node.name, location=loc.tolist())]


@register
class Tangent(LineLike):
    name = "tangent"
    text_fields = ("x", "function")
    ref_fields = ("function", "point")

    def compute(self, c, node):
        f, x0, y0 = _touch_point(c, node)
        m = kp.require_differentiable(f, X, x0)
        expr, geom = _line_from((x0, y0), m)
        return {"expr": expr, "geom": geom, "slope": m, "point": [x0, y0]}

    def extra_draw(self, c, node, fr):
        return _mark(c, node, fr, node.data["point"])


@register
class Normal(LineLike):
    name = "normal"
    text_fields = ("x", "function")
    ref_fields = ("function", "point")

    def compute(self, c, node):
        f, x0, y0 = _touch_point(c, node)
        m = kp.require_differentiable(f, X, x0)
        if m == 0:
            geom = sp.Line2D(sp.Point2D(x0, y0), sp.Point2D(x0, y0 + 1))
            return {"expr": None, "geom": geom, "slope": sp.zoo, "point": [x0, y0]}
        expr, geom = _line_from((x0, y0), -1 / m)
        return {"expr": expr, "geom": geom, "slope": sp.simplify(-1 / m), "point": [x0, y0]}

    def extra_draw(self, c, node, fr):
        return _mark(c, node, fr, node.data["point"])


@register
class Secant(LineLike):
    name = "secant"
    text_fields = ("x1", "x2", "function")
    ref_fields = ("function",)

    def compute(self, c, node):
        f = c.function_of(node.spec["function"])
        x1, x2 = c.value(node.spec["x1"]), c.value(node.spec["x2"])
        if sp.simplify(x1 - x2) == 0:
            raise MathRefusal("A secant needs two different x values (use a tangent for one point).")
        y1, y2 = kp.require_defined(f, X, x1), kp.require_defined(f, X, x2)
        m = sp.simplify((y2 - y1) / (x2 - x1))
        expr, geom = _line_from((x1, y1), m)
        return {"expr": expr, "geom": geom, "slope": m, "points": [[x1, y1], [x2, y2]]}

    def extra_draw(self, c, node, fr):
        return _mark(c, node, fr, node.data["points"][0], "p1") + _mark(c, node, fr, node.data["points"][1], "p2")


# --------------------------------------------------------------------------- integrals


def exact_integral(f, a, b, timeout: float = 10.0):
    """(value, exact?) of the definite integral; numeric fallback at 30 digits."""
    val = with_timeout(lambda: sp.integrate(f, (X, a, b)), timeout)
    if val is not None and not val.has(sp.Integral) and sp.N(val).is_finite:
        simp = with_timeout(lambda: sp.simplify(val), 5.0)
        return (simp if simp is not None else val), True
    fm = sp.lambdify(X, f, modules="mpmath")
    with mpmath.workdps(30):
        v = mpmath.quad(fm, [mpmath.mpf(str(sp.N(a, 30))), mpmath.mpf(str(sp.N(b, 30)))])
    return sp.Float(mpmath.nstr(v, 25), 25), False


def require_continuous(f, a, b, what="the function"):
    sing = with_timeout(lambda: sp.calculus.singularities(f, X, sp.Interval(a, b)), 6.0)
    if isinstance(sing, sp.FiniteSet) and len(sing):
        raise MathRefusal(f"{what} is not continuous on [{a}, {b}] (singular at {sorted(sing, key=fnum)}); "
                          "the integral/area would be improper or undefined.")
    fn = numeric(f, [X])
    xs = np.linspace(fnum(a), fnum(b), 2001)
    ys = fn(xs)
    if not np.all(np.isfinite(ys)):
        bad = xs[~np.isfinite(ys)][0]
        raise MathRefusal(f"{what} is undefined at x ≈ {bad:.6g} inside [{a}, {b}].")


@register
class RiemannSum(Kind):
    name = "riemann_sum"
    needs_axes = 2
    colored = True
    text_fields = ("a", "b", "n", "function")
    ref_fields = ("function",)
    METHODS = ("left", "right", "midpoint", "trapezoid", "upper", "lower")

    def compute(self, c, node):
        s = node.spec
        f = c.function_of(s["function"])
        a, b = c.value(s["a"]), c.value(s["b"])
        n = int(fnum(c.value(s.get("n", 8))))
        method = s.get("method", "left")
        if method not in self.METHODS:
            raise MathRefusal(f"method must be one of {self.METHODS}")
        if n < 1:
            raise MathRefusal("n must be at least 1.")
        if not fnum(b) > fnum(a):
            raise MathRefusal("Need a < b.")
        require_continuous(f, a, b)
        dx = (b - a) / n
        cells = []
        total = sp.Integer(0)
        for i in range(n):
            x0, x1 = a + i * dx, a + (i + 1) * dx
            if method == "left":
                h0 = h1 = f.subs(X, x0)
            elif method == "right":
                h0 = h1 = f.subs(X, x1)
            elif method == "midpoint":
                h0 = h1 = f.subs(X, (x0 + x1) / 2)
            elif method == "trapezoid":
                h0, h1 = f.subs(X, x0), f.subs(X, x1)
            else:
                cands = [x0, x1] + [k.x for k in kp.extrema(f, X, fnum(x0), fnum(x1))]
                vals = [f.subs(X, xx) for xx in cands]
                h0 = h1 = (max if method == "upper" else min)(vals, key=fnum)
            cells.append((x0, x1, sp.simplify(h0), sp.simplify(h1)))
            total += (h0 + h1) / 2 * dx
        total_s = with_timeout(lambda: sp.nsimplify(sp.simplify(total)) if total.is_number else total, 6.0) or total
        if abs(fnum(total_s - total)) > 1e-12 * max(1, abs(fnum(total))):
            total_s = sp.simplify(total)
        integral, exact = exact_integral(f, a, b)
        return {"cells": cells, "value": total_s, "integral": integral, "integral_exact": exact,
                "error": sp.simplify(total_s - integral) if exact else sp.Float(fnum(total_s) - fnum(integral), 20)}

    def provides(self, node):
        return {node.name: node.data["value"]} if "value" in node.data else {}

    def draw(self, c, node):
        fr = c.frame(node.spec["axes"])
        col = c.color(node)
        splines, outlines = [], []
        for x0, x1, h0, h1 in node.data["cells"]:
            q = np.array([[fnum(x0), 0], [fnum(x1), 0], [fnum(x1), fnum(h1)], [fnum(x0), fnum(h0)]])
            q[:, 1] = np.clip(q[:, 1], fr.lo[1], fr.hi[1])
            L = fr.to_local(q)
            L[:, 2] = 0
            splines.append(geo.polyline_spline(L, cyclic=True))
            outlines.append(geo.polyline_spline(fr.to_local(q, "curve"), cyclic=True))
        objs = [c.r.filled(f"{node.name}", splines, fr, col, node.name,
                           opacity=node.style.get("fill_opacity", c.r.theme.fill_opacity))]
        objs.append(c.r.curve(f"{node.name}:edges", outlines, fr, col, node.style.get("thickness", 0.02),
                              node.name, register=False))
        return objs

    def describe(self, node):
        d = node.data
        return {"sum": str(d.get("value")), "sum_latex": nice_latex(d.get("value", 0)),
                "sum_value": fnum(d.get("value", 0)), "integral": str(d.get("integral")),
                "integral_exact": d.get("integral_exact"), "error": str(d.get("error"))}


@register
class Area(Kind):
    name = "area"
    needs_axes = 2
    colored = True
    text_fields = ("a", "b", "upper", "lower")
    ref_fields = ("upper", "lower")

    def compute(self, c, node):
        s = node.spec
        u = c.function_of(s["upper"])
        lo_ = c.function_of(s.get("lower", "0"))
        a, b = c.value(s["a"]), c.value(s["b"])
        if not fnum(b) > fnum(a):
            raise MathRefusal("Need a < b.")
        require_continuous(u, a, b, "the upper function")
        require_continuous(lo_, a, b, "the lower function")
        cross = [k.x for k in kp.intersections(u, lo_, X, fnum(a), fnum(b))
                 if fnum(a) < k.xf < fnum(b)]
        pts = [a] + sorted(cross, key=fnum) + [b]
        pieces, signed, geometric, exact_all = [], sp.Integer(0), sp.Integer(0), True
        for p, q in zip(pts[:-1], pts[1:]):
            v, exact = exact_integral(u - lo_, p, q)
            exact_all &= exact
            signed += v
            geometric += sp.Abs(v) if exact else abs(v)
            pieces.append((p, q, v))
        return {"upper": u, "lower": lo_, "a": a, "b": b, "pieces": pieces,
                "signed": sp.simplify(signed), "area": sp.simplify(geometric), "exact": exact_all}

    def provides(self, node):
        return {node.name: node.data["area"]} if "area" in node.data else {}

    def draw(self, c, node):
        d = node.data
        fr = c.frame(node.spec["axes"])
        fu, fl = numeric(d["upper"], [X]), numeric(d["lower"], [X])
        splines = []
        for p, q, v in d["pieces"]:
            pa, pb = fnum(p), fnum(q)

            def F(t):
                return np.stack([t, fu(t), fl(t)], 1)

            t, P, _ = smp.adaptive_sample(lambda t: np.stack([t, fu(t)], 1), pa, pb, fr.scale[:2], 2e-4)
            t2, P2, _ = smp.adaptive_sample(lambda t: np.stack([t, fl(t)], 1), pa, pb, fr.scale[:2], 2e-4)
            top = np.maximum(P[:, 1], fl(P[:, 0]))
            bot = np.minimum(P2[:, 1], fu(P2[:, 0]))
            upper = np.stack([P[:, 0], np.clip(top, fr.lo[1], fr.hi[1])], 1)
            lower = np.stack([P2[:, 0], np.clip(bot, fr.lo[1], fr.hi[1])], 1)[::-1]
            poly = np.concatenate([upper, lower])
            L = fr.to_local(poly)
            L[:, 2] = 0
            splines.append(geo.polyline_spline(L, cyclic=True))
        col = c.color(node)
        objs = [c.r.filled(node.name, splines, fr, col, node.name,
                           opacity=node.style.get("fill_opacity", c.r.theme.fill_opacity))]
        if node.style.get("label"):
            lab = node.style["label"]
            if lab is True:
                lab = rf"A = {nice_latex(d['area'])}"
            mid = (fnum(d["a"]) + fnum(d["b"])) / 2
            ym = (fnum(d["upper"].subs(X, mid)) + fnum(d["lower"].subs(X, mid))) / 2
            ym = float(np.clip(ym, fr.lo[1], fr.hi[1]))
            objs += c.r.auto_label(f"{node.name}:label", lab, fr.to_local([mid, ym], "label")[0], fr, node.name,
                                   size=node.style.get("label_size", 0.42), color=c.r.theme.text, gap=0.0)
        return objs

    def describe(self, node):
        d = node.data
        return {"signed_integral": str(d.get("signed")), "area": str(d.get("area")),
                "area_latex": nice_latex(d.get("area", 0)), "area_value": fnum(d.get("area", 0)),
                "exact": d.get("exact"), "pieces": [[str(p), str(q), str(v)] for p, q, v in d.get("pieces", [])]}


# --------------------------------------------------------------------------- derived functions


@register
class Taylor(FunctionLike):
    name = "taylor"
    text_fields = ("a", "degree", "function")
    ref_fields = ("function",)

    def compute(self, c, node):
        f = c.function_of(node.spec["function"])
        a = c.value(node.spec.get("a", 0))
        n = int(fnum(c.value(node.spec.get("degree", 3))))
        if n < 0 or n > 40:
            raise MathRefusal("degree must be between 0 and 40.")
        terms = []
        d = f
        for k in range(n + 1):
            val = kp.require_defined(d, X, a) if k == 0 else _derivative_at(d, a, k, node)
            terms.append(val / sp.factorial(k) * (X - a) ** k)
            d = sp.diff(d, X)
        poly = sp.expand(sum(terms))
        return {"expr": poly, "center": a, "degree": n}

    def label_latex(self, node):
        return rf"T_{{{node.data['degree']}}}(x) = {nice_latex(node.data['expr'])}"


def _derivative_at(d, a, k, node):
    try:
        return kp.require_defined(d, X, a)
    except MathRefusal as exc:
        raise MathRefusal(f"{node.name}: derivative {k} is undefined at x = {a}, so the Taylor polynomial "
                          f"does not exist there ({exc}).") from exc


@register
class Derivative(FunctionLike):
    name = "derivative"
    text_fields = ("order", "function")
    ref_fields = ("function",)

    def compute(self, c, node):
        f = c.function_of(node.spec["function"])
        n = int(fnum(c.value(node.spec.get("order", 1))))
        if n < 1:
            raise MathRefusal("order must be >= 1")
        d = sp.diff(f, X, n)
        simp = with_timeout(lambda: sp.simplify(d), 5.0)
        return {"expr": simp if simp is not None else d, "order": n}

    def label_latex(self, node):
        base = node.spec["function"]
        primes = "'" * node.data["order"] if node.data["order"] <= 3 else f"^{{({node.data['order']})}}"
        return f"{base}{primes}(x) = {nice_latex(node.data['expr'])}"


@register
class Antiderivative(FunctionLike):
    name = "antiderivative"
    text_fields = ("a", "function")
    ref_fields = ("function",)

    def compute(self, c, node):
        f = c.function_of(node.spec["function"])
        a = c.value(node.spec.get("a", 0))
        tt = sp.Symbol("t", real=True)
        F = with_timeout(lambda: sp.integrate(f.subs(X, tt), (tt, a, X)), 12.0)
        if F is None or F.has(sp.Integral):
            raise MathRefusal(f"No closed-form antiderivative of {f} was found; it cannot be drawn exactly.")
        F = with_timeout(lambda: sp.simplify(F), 6.0) or F
        return {"expr": F, "lower": a}

    def label_latex(self, node):
        return rf"{node.name}(x) = {nice_latex(node.data['expr'])}"


# --------------------------------------------------------------------------- asymptotes


@register
class Asymptotes(Kind):
    name = "asymptotes"
    needs_axes = 2
    text_fields = ("function",)
    ref_fields = ("function",)

    def compute(self, c, node):
        f = c.function_of(node.spec["function"])
        fr = c.frame(node.spec["axes"])
        lines = []
        dom = sp.Interval(sp.nsimplify(fr.lo[0]), sp.nsimplify(fr.hi[0]))
        sing = with_timeout(lambda: sp.calculus.singularities(f, X, dom), 8.0)
        if isinstance(sing, sp.FiniteSet):
            for x0 in sing:
                lp = with_timeout(lambda x0=x0: sp.limit(f, X, x0, "+"), 6.0)
                lm = with_timeout(lambda x0=x0: sp.limit(f, X, x0, "-"), 6.0)
                if any(v is not None and v in (sp.oo, -sp.oo) for v in (lp, lm)):
                    lines.append(("vertical", x0, None))
        for side in (sp.oo, -sp.oo):
            L = with_timeout(lambda side=side: sp.limit(f, X, side), 8.0)
            if L is not None and L.is_finite and L.is_real:
                if not any(k == "horizontal" and sp.simplify(v - L) == 0 for k, v, _ in lines):
                    lines.append(("horizontal", L, side))
                continue
            m = with_timeout(lambda side=side: sp.limit(f / X, X, side), 8.0)
            if m is not None and m.is_finite and m.is_real and m != 0:
                b = with_timeout(lambda side=side, m=m: sp.limit(f - m * X, X, side), 8.0)
                if b is not None and b.is_finite and b.is_real:
                    if not any(k == "oblique" and sp.simplify(v[0] - m) == 0 and sp.simplify(v[1] - b) == 0
                               for k, v, _ in lines):
                        lines.append(("oblique", (m, b), side))
        return {"lines": lines}

    def draw(self, c, node):
        fr = c.frame(node.spec["axes"])
        objs, splines = [], []
        th = node.style.get("thickness", 0.025)
        col = node.style.get("color") or c.r.theme.axes
        labels = []
        for kind, v, _ in node.data["lines"]:
            if kind == "vertical":
                P0, D, lab = np.array([fnum(v), 0.0]), np.array([0.0, 1.0]), f"x = {nice_latex(v)}"
            elif kind == "horizontal":
                P0, D, lab = np.array([0.0, fnum(v)]), np.array([1.0, 0.0]), f"y = {nice_latex(v)}"
            else:
                m, b = v
                P0, D = np.array([0.0, fnum(b)]), np.array([1.0, fnum(m)])
                lab = f"y = {nice_latex(sp.expand(m * sp.Symbol('x') + b))}"
            seg = clip_line(P0, D, fr)
            if seg is None:
                continue
            for d in dash_polyline(fr.to_local(seg, "curve"), np.ones(3), th * 5, th * 4):
                splines.append(geo.polyline_spline(d))
            labels.append((lab, seg))
        if splines:
            objs.append(c.r.curve(node.name, splines, fr, col, th, node.name, opacity=0.9))
        if node.style.get("label", True):
            for i, (lab, seg) in enumerate(labels):
                end = seg[np.argmax(seg[:, 1] + seg[:, 0] * 1e-3)]
                anchor = fr.to_local(end, "label")[0]
                objs += c.r.auto_label(f"{node.name}:label{i}", lab, anchor, fr, node.name, size=0.34, color=col)
        return objs

    def describe(self, node):
        out = []
        for kind, v, side in node.data.get("lines", []):
            if kind == "vertical":
                out.append(f"x = {v}")
            elif kind == "horizontal":
                out.append(f"y = {v} (x -> {side})")
            else:
                out.append(f"y = {sp.expand(v[0] * sp.Symbol('x') + v[1])} (x -> {side})")
        return {"asymptotes": out}


__all__ = ["clip_line", "line_objects", "line_equation_latex", "exact_integral", "LineLike"]
_ = (Construction, Node)
