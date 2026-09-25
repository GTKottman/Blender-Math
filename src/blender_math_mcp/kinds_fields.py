"""Fields and differential equations: vector/slope fields, ODE solutions, phase portraits, complex maps."""

from __future__ import annotations

import numpy as np
import sympy as sp

from . import geometry as geo
from . import typeset as ts
from .construction import Kind, X, Y, coord_latex, fnum, nice_latex, register
from .fields import dopri45, solve_ivp_exact
from .keypoints import MathRefusal, solve_system_box, with_timeout
from .kinds_core import FunctionLike, curve_splines, draw_curve, numeric


def _xy_expr(c, text, node):
    e = c.parse(text, ("x", "y"), node.name)
    extra = sorted(q.name for q in e.free_symbols if q not in (X, Y))
    if extra:
        raise MathRefusal(f"{node.name}: unknown symbols {extra} (use x and y).")
    return e


@register
class VectorField(Kind):
    name = "vector_field"
    needs_axes = 2
    text_fields = ("field",)

    def compute(self, c, node):
        P, Q = (_xy_expr(c, t, node) for t in node.spec["field"])
        return {"P": P, "Q": Q}

    def draw(self, c, node):
        fr = c.frame(node.spec["axes"])
        d = node.data
        fP, fQ = numeric(d["P"], [X, Y]), numeric(d["Q"], [X, Y])
        n = int(node.spec.get("density", 17))
        xs = np.linspace(fr.lo[0], fr.hi[0], n + 2)[1:-1]
        ys = np.linspace(fr.lo[1], fr.hi[1], max(2, int(round(n * (fr.hi[1] - fr.lo[1]) / (fr.hi[0] - fr.lo[0])))) + 2)[1:-1]
        GX, GY = np.meshgrid(xs, ys)
        U, V = fP(GX, GY), fQ(GX, GY)
        mag = np.hypot(U, V)
        ok = np.isfinite(mag) & (mag > 1e-12)
        cell = min(xs[1] - xs[0] if len(xs) > 1 else 1, ys[1] - ys[0] if len(ys) > 1 else 1)
        max_len = 0.85 * cell
        normalize = node.spec.get("normalize", True)
        mmax = np.nanmax(mag[ok]) if ok.any() else 1.0
        verts, faces, cols = [], [], []
        mags = mag[ok]
        cm = geo.colormap(mags, node.style.get("colormap", "viridis")) if ok.any() else []
        for k, (x0, y0, u, v, m) in enumerate(zip(GX[ok], GY[ok], U[ok], V[ok], mags)):
            L = max_len if normalize else max_len * m / mmax
            if L < 1e-6:
                continue
            dvec = np.array([u, v]) / m * L
            A = fr.to_local([x0 - dvec[0] / 2, y0 - dvec[1] / 2], "vector")[0]
            B = fr.to_local([x0 + dvec[0] / 2, y0 + dvec[1] / 2], "vector")[0]
            th = node.style.get("thickness", 0.028)
            av, af = geo.arrow_mesh(A, B, th, style="flat", head_length=th * 3.5, head_width=th * 2.4)
            base = len(verts)
            verts += av
            faces += [[i + base for i in f] for f in af]
            cols += [list(cm[k])] * len(av)
        if not verts:
            return []
        extra = {"colors": cols, "vertex_colors": True} if node.style.get("colormap", True) and cols else {}
        return [c.r.mesh(node.name, verts, faces, fr, "white" if extra else c.color(node), node.name, **extra)]

    def describe(self, node):
        return {"field": [str(node.data.get("P")), str(node.data.get("Q"))]}


@register
class SlopeField(Kind):
    name = "slope_field"
    needs_axes = 2
    text_fields = ("rhs",)

    def compute(self, c, node):
        return {"rhs": _xy_expr(c, node.spec["rhs"], node)}

    def draw(self, c, node):
        fr = c.frame(node.spec["axes"])
        f = numeric(node.data["rhs"], [X, Y])
        n = int(node.spec.get("density", 21))
        xs = np.linspace(fr.lo[0], fr.hi[0], n + 2)[1:-1]
        ny = max(2, int(round(n * (fr.hi[1] - fr.lo[1]) / (fr.hi[0] - fr.lo[0]))))
        ys = np.linspace(fr.lo[1], fr.hi[1], ny + 2)[1:-1]
        GX, GY = np.meshgrid(xs, ys)
        S = f(GX, GY)
        L = 0.7 * min(xs[1] - xs[0], ys[1] - ys[0])
        verts, faces = [], []
        th = node.style.get("thickness", 0.025)
        for x0, y0, m in zip(GX.ravel(), GY.ravel(), S.ravel()):
            if not np.isfinite(m):
                continue
            dvec = np.array([1.0, m]) / np.hypot(1.0, m) * L / 2
            A = fr.to_local([x0 - dvec[0], y0 - dvec[1]], "grid")[0]
            B = fr.to_local([x0 + dvec[0], y0 + dvec[1]], "grid")[0]
            q, fq = geo.arrow_mesh(A, B, th, style="flat", head_length=0, head_width=0)
            base = len(verts)
            verts += q
            faces += [[i + base for i in fc] for fc in fq]
        if not verts:
            return []
        return [c.r.mesh(node.name, verts, faces, fr, node.style.get("color") or c.r.theme.axes, node.name,
                         opacity=node.style.get("opacity", 0.7))]

    def describe(self, node):
        return {"rhs": str(node.data.get("rhs"))}


@register
class ODESolution(FunctionLike):
    """Solution of y' = f(x, y), y(x0) = y0 through the initial point."""

    name = "ode_solution"
    text_fields = ("rhs", "initial")

    def compute(self, c, node):
        rhs = _xy_expr(c, node.spec["rhs"], node)
        x0, y0 = (c.value(v) for v in node.spec["initial"])
        f0 = rhs.subs({X: x0, Y: y0})
        if not sp.N(f0).is_finite:
            raise MathRefusal(f"y' = {rhs} is undefined at the initial point ({x0}, {y0}).")
        exact = None if node.spec.get("method") == "numeric" else solve_ivp_exact(rhs, X, Y, x0, y0)
        out = {"rhs": rhs, "x0": x0, "y0": y0}
        if exact is not None:
            fr = c.frame(node.spec["axes"])
            # The IVP solution lives on the interval around x0 between singularities.
            dom = sp.Interval(sp.nsimplify(fr.lo[0]), sp.nsimplify(fr.hi[0]))
            sing = with_timeout(lambda: sp.calculus.singularities(exact, X, dom), 6.0)
            lo_, hi_ = fr.lo[0], fr.hi[0]
            if isinstance(sing, sp.FiniteSet):
                for s_ in sing:
                    v = fnum(s_)
                    if v < fnum(x0):
                        lo_ = max(lo_, v)
                    elif v > fnum(x0):
                        hi_ = min(hi_, v)
            out.update({"expr": exact, "exact": True, "domain": [lo_, hi_]})
        else:
            out.update({"expr": None, "exact": False})
        return out

    def provides(self, node):
        e = node.data.get("expr")
        return {node.name: sp.Lambda(X, e)} if e is not None else {}

    def draw(self, c, node):
        d = node.data
        if d.get("exact"):
            return super().draw(c, node)
        fr = c.frame(node.spec["axes"])
        f = numeric(d["rhs"], [X, Y])
        span = fr.hi[0] - fr.lo[0]
        yb = 50 * max(abs(fr.lo[1]), abs(fr.hi[1]))

        def g(t, y):
            return np.array([f(np.array(t), y[0])]).ravel()

        runs, notes = [], []
        for end in (fr.hi[0], fr.lo[0]):
            ts_, ys_, note = dopri45(g, fnum(d["x0"]), [fnum(d["y0"])], end, bound=yb, max_step=span / 400)
            if note:
                notes.append(note)
            if len(ts_) > 1:
                runs.append((ts_, ys_[:, 0]))
        d["note"] = "; ".join(notes) if notes else None
        splines, polys = [], []
        for ts_, ys_ in runs:
            order = np.argsort(ts_)
            t_, y_ = ts_[order], ys_[order]
            m = f(t_, y_)
            P = np.stack([t_, y_], 1)
            # clip to the view vertically (drop points far outside)
            keep = (y_ >= fr.lo[1] - 1e-9) & (y_ <= fr.hi[1] + 1e-9)
            if keep.sum() < 2:
                continue
            idx = np.nonzero(keep)[0]
            groups = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)
            for gi in groups:
                if len(gi) < 2:
                    continue
                Pg, mg, tg = P[gi], m[gi], t_[gi]
                co = Pg
                h = np.diff(tg)
                hr = np.concatenate([Pg[:-1] + np.stack([h, mg[:-1] * h], 1) / 3, Pg[-1:]])
                hl = np.concatenate([Pg[:1], Pg[1:] - np.stack([h, mg[1:] * h], 1) / 3])
                from .splines import spline_to_blender

                splines.append(spline_to_blender({"co": co, "hl": hl, "hr": hr}, lambda A: fr.to_local(A, "curve")))
                polys.append(Pg)
        d["runs"] = polys
        return draw_curve(c, node, splines, fr)

    def label_latex(self, node):
        return f"y = {nice_latex(node.data['expr'])}" if node.data.get("expr") is not None else node.name

    def describe(self, node):
        d = node.data
        return {"solution": str(d.get("expr")) if d.get("exact") else "numeric (Dormand-Prince 5(4), rtol 1e-10)",
                "exact": d.get("exact"), "valid_interval": d.get("domain"), "note": d.get("note")}


@register
class PhasePortrait(Kind):
    name = "phase_portrait"
    needs_axes = 2
    colored = True
    text_fields = ("system", "initials", "t_range")

    def compute(self, c, node):
        P, Q = (_xy_expr(c, t, node) for t in node.spec["system"])
        fr = c.frame(node.spec["axes"])
        eq = []
        J = sp.Matrix([[sp.diff(P, X), sp.diff(P, Y)], [sp.diff(Q, X), sp.diff(Q, Y)]])
        for (px, py), exact, method in solve_system_box(P, Q, X, Y, (fr.lo[0], fr.hi[0], fr.lo[1], fr.hi[1])):
            Jp = J.subs({X: px, Y: py})
            eq.append({"point": [sp.simplify(px), sp.simplify(py)], "type": classify(Jp), "exact": exact,
                       "method": method, "eigenvalues": [str(sp.simplify(v)) for v in Jp.eigenvals()]})
        return {"P": P, "Q": Q, "equilibria": eq}

    def draw(self, c, node):
        fr = c.frame(node.spec["axes"])
        d = node.data
        fP, fQ = numeric(d["P"], [X, Y]), numeric(d["Q"], [X, Y])

        def g(t, z):
            return np.array([fP(np.array(z[0]), np.array(z[1])), fQ(np.array(z[0]), np.array(z[1]))], float).ravel()

        T0, T1 = (fnum(c.value(v)) for v in node.spec.get("t_range", [-10, 10]))
        bound = 20 * max(abs(fr.lo).max(), abs(fr.hi[:2]).max())
        splines = []
        for init in node.spec.get("initials", []):
            z0 = [fnum(c.value(v)) for v in init]
            for end in (T1, T0):
                if end == 0:
                    continue
                ts_, zs, _ = dopri45(g, 0.0, z0, end, bound=bound, max_step=abs(end) / 600)
                if len(ts_) < 2:
                    continue
                order = np.argsort(ts_)
                Z = zs[order]
                inside = (Z[:, 0] >= fr.lo[0]) & (Z[:, 0] <= fr.hi[0]) & (Z[:, 1] >= fr.lo[1]) & (Z[:, 1] <= fr.hi[1])
                idx = np.nonzero(inside)[0]
                for gi in np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1):
                    if len(gi) >= 2:
                        splines.append(geo.polyline_spline(fr.to_local(Z[gi], "curve")))
        objs = draw_curve(c, node, splines, fr) if splines else []
        for k, e in enumerate(d["equilibria"]):
            loc = fr.to_local([fnum(e["point"][0]), fnum(e["point"][1])], "point")[0]
            objs.append(c.r.filled(f"{node.name}:eq{k}", [geo.bezier_circle([0, 0, 0], 0.08)], fr, c.r.theme.text,
                                   node.name, location=loc.tolist()))
            if node.style.get("label", True):
                lab = rf"\text{{{e['type']}}}" if ts.resolve_backend(c.r.backend) == "latex" \
                    else r"\mathrm{" + e["type"].replace(" ", r"\ ") + "}"
                objs += c.r.auto_label(f"{node.name}:eqlabel{k}", lab, loc, fr, node.name, size=0.32)
        return objs

    def describe(self, node):
        return {"equilibria": [{"point": coord_latex(e["point"]), "type": e["type"], "eigenvalues": e["eigenvalues"]}
                               for e in node.data.get("equilibria", [])]}


def classify(J: sp.Matrix) -> str:
    """Linearization type of an equilibrium from the exact Jacobian."""
    tr, det = sp.simplify(J.trace()), sp.simplify(J.det())
    disc = sp.simplify(tr ** 2 - 4 * det)
    if det.is_negative:
        return "saddle"
    if det == 0:
        return "degenerate (non-isolated)"
    if tr == 0:
        return "center (linearization)"
    if disc.is_negative:
        return "stable spiral" if tr.is_negative else "unstable spiral"
    if disc == 0:
        return "stable degenerate node" if tr.is_negative else "unstable degenerate node"
    return "stable node" if tr.is_negative else "unstable node"


@register
class ComplexMap(Kind):
    """Image of the grid lines of a z-rectangle under w = f(z), drawn in the w-plane."""

    name = "complex_map"
    needs_axes = 2
    colored = True
    text_fields = ("expression", "x_range", "y_range")

    def compute(self, c, node):
        z = sp.Symbol("z")
        e = c.parse(node.spec["expression"], ("z",), node.name)
        e = e.subs(sp.Symbol("z", real=True), z)
        extra = sorted(q.name for q in e.free_symbols if q.name != "z")
        if extra:
            raise MathRefusal(f"{node.name}: unknown symbols {extra} (use z).")
        return {"f": e, "z": z}

    def draw(self, c, node):
        fr = c.frame(node.spec["axes"])
        d = node.data
        s = node.spec
        xr = [fnum(c.value(v)) for v in s.get("x_range", [-2, 2])]
        yr = [fnum(c.value(v)) for v in s.get("y_range", [-2, 2])]
        n = int(s.get("lines", 9))
        u = sp.Symbol("u", real=True)
        splines_h, splines_v = [], []
        th = node.style.get("thickness", 0.025)
        for yv in np.linspace(*yr, n):
            w = sp.expand_complex(d["f"].subs(d["z"], u + sp.I * sp.nsimplify(yv)))
            sp_, _ = curve_splines(c, [sp.re(w), sp.im(w)], u, xr[0], xr[1], fr, th)
            splines_h += sp_
        for xv in np.linspace(*xr, n):
            w = sp.expand_complex(d["f"].subs(d["z"], sp.nsimplify(xv) + sp.I * u))
            sp_, _ = curve_splines(c, [sp.re(w), sp.im(w)], u, yr[0], yr[1], fr, th)
            splines_v += sp_
        objs = []
        if splines_h:
            objs += draw_curve(c, node, splines_h, fr, name=f"{node.name}:h", color=c.r.theme.palette[0], thickness=th)
        if splines_v:
            objs += draw_curve(c, node, splines_v, fr, name=f"{node.name}:v", color=c.r.theme.palette[1], thickness=th)
        return objs

    def describe(self, node):
        return {"f": str(node.data.get("f"))}


