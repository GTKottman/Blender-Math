"""Core node kinds: parameters, axes, functions, points, text, derivations, curves, surfaces."""

from __future__ import annotations

import math
import re

import numpy as np
import sympy as sp

from . import expressions as ex
from . import geometry as geo
from . import keypoints as kp
from . import sampling as smp
from . import splines as spl
from . import typeset as ts
from .construction import (KINDS, Construction, Kind, Node, T, X, Y, coord_latex, fnum, nice_latex,
                           register)
from .keypoints import MathRefusal
from .render import LAYER_Z, Frame, dash_polyline

# --------------------------------------------------------------------------- drawing helpers


def numeric(expr: sp.Expr, variables):
    try:
        return ex.compile_numeric(expr, [v.name if isinstance(v, sp.Symbol) else v for v in variables])
    except ex.ExpressionError as exc:
        raise MathRefusal(str(exc)) from exc


def safe_numeric(expr, variables):
    """Numeric function or None if it cannot be compiled (e.g. derivative of floor)."""
    try:
        return ex.compile_numeric(expr, [v.name for v in variables])
    except Exception:  # noqa: BLE001
        return None


def curve_splines(c: Construction, comps: list[sp.Expr], var: sp.Symbol, t0: float, t1: float, frame: Frame,
                  thickness: float, clip: bool = True, layer: str = "curve"):
    """Exact-tangent Bezier splines for a parametric curve (graphs: comps = [x, f(x)]).

    Returns ``(blender_splines, runs_in_math_coords)``.
    """
    fns = [numeric(e, [var]) for e in comps]
    dfs = [safe_numeric(sp.diff(e, var), [var]) for e in comps]
    dims = len(comps)

    def F(t):
        return np.stack([np.broadcast_to(fn(t), np.shape(t)) for fn in fns], axis=1)

    dF = None
    if all(d is not None for d in dfs):
        def dF(t):  # noqa: E306
            return np.stack([np.broadcast_to(d(t), np.shape(t)) for d in dfs], axis=1)

    tol = max(thickness * 0.02, 1e-4)  # 2% of the line width (~0.1 px at 1080p)
    lo = hi = None
    if clip and frame.lo is not None:
        lo, hi = frame.lo[:dims], frame.hi[:dims]
    runs = smp.sample_curve_t(F, t0, t1, frame.scale[:dims], tol, lo, hi)
    if not runs:
        return [], []
    out = []
    for t, P in runs:
        closed = False
        if len(runs) == 1 and np.allclose(P[0], P[-1], atol=1e-9) and len(P) > 3:
            closed = True
        bz = spl.hermite_bezier(F, dF, t, P, frame.scale[:dims], tol)
        if closed:
            # drop the duplicate end point; its incoming handle becomes the first point's left handle
            bz = {"co": bz["co"][:-1], "hl": np.concatenate([[bz["hl"][-1]], bz["hl"][1:-1]]),
                  "hr": bz["hr"][:-1]}
        out.append(spl.spline_to_blender(bz, lambda A: frame.to_local(A, layer), cyclic=closed))
    return out, [P for _, P in runs]


def draw_curve(c: Construction, node: Node, splines, frame: Frame, name=None, layer="curve", color=None,
               thickness=None, dashed=None) -> list[str]:
    if not splines:
        return []
    th = thickness if thickness is not None else node.style.get("thickness", 0.04)
    col = color or c.color(node)
    dashed = node.style.get("dashed") if dashed is None else dashed
    if dashed:
        dsh = []
        for s_ in splines:
            P = np.asarray(s_.get("poly") or _bezier_points(s_), float)
            for d in dash_polyline(P, np.ones(3), dash=th * 4, gap=th * 3):
                dsh.append(geo.polyline_spline(d))
        splines = dsh
    return [c.r.curve(name or node.name, splines, frame, col, th, owner=node.name, layer=layer,
                      opacity=node.style.get("opacity", 1.0), meta={"kind": node.kind})]


def _bezier_points(s_, per=12):
    co, hl, hr = (np.asarray(s_[k], float) for k in ("co", "hl", "hr"))
    n = len(co)
    segs = n if s_.get("cyclic") else n - 1
    pts = []
    u = np.linspace(0, 1, per, endpoint=False)[:, None]
    for i in range(segs):
        j = (i + 1) % n
        p0, p1, p2, p3 = co[i], hr[i], hl[j], co[j]
        pts.append((1 - u) ** 3 * p0 + 3 * (1 - u) ** 2 * u * p1 + 3 * (1 - u) * u ** 2 * p2 + u ** 3 * p3)
    pts.append(co[0:1] if s_.get("cyclic") else co[-1:])
    return np.concatenate(pts)


def point_label(c: Construction, node: Node, P_math, frame: Frame, default: str | None) -> list[str]:
    """Draw the label of a point-like node (auto-placed to avoid overlaps)."""
    lab = node.style.get("label", True)
    if lab is False or lab is None:
        return []
    if lab is True:
        lab = default
    elif lab == "coords":
        lab = coord_latex(node.data["point"])
    elif lab == "name_coords":
        lab = f"{node.name} {coord_latex(node.data['point'])}"
    if not lab:
        return []
    size = node.style.get("label_size", 0.42)
    anchor = frame.to_local(np.asarray([fnum(v) for v in P_math], float), "label")[0]
    if frame.flat:
        return c.r.auto_label(f"{node.name}:label", lab, anchor, frame, node.name, size=size, color=c.color(node),
                              prefer=node.style.get("label_dir"))
    pos = anchor + np.array([0.18, 0, 0.18])
    return c.r.text(f"{node.name}:label", lab, pos, frame, node.name, size=size, color=c.color(node),
                    align_x="left", align_y="bottom", plane="xz", face_camera=_has_camera(c))


def _has_camera(c: Construction) -> bool:
    try:
        return c.r.send("get_scene_info").get("camera") is not None
    except Exception:  # noqa: BLE001
        return False


def graph_label(c: Construction, node: Node, runs, frame: Frame, default_latex: str) -> list[str]:
    lab = node.style.get("label")
    if not lab or not runs:
        return []
    if lab is True:
        lab = default_latex
    size = node.style.get("label_size", 0.45)
    # anchor: the visible point furthest to the right (or the end of the longest run)
    best = max(runs, key=lambda P: P[-1][0])
    end = best[-1]
    anchor = frame.to_local(end, "label")[0]
    if frame.flat:
        return c.r.auto_label(f"{node.name}:label", lab, anchor, frame, node.name, size=size, color=c.color(node),
                              prefer=node.style.get("label_dir", "ne"))
    return c.r.text(f"{node.name}:label", lab, anchor + np.array([0.2, 0, 0.2]), frame, node.name, size=size,
                    color=c.color(node), align_x="left", align_y="bottom", plane="xz", face_camera=_has_camera(c))


# --------------------------------------------------------------------------- parameter


@register
class Parameter(Kind):
    name = "parameter"
    text_fields = ("value", "min", "max")

    def compute(self, c, node):
        v = c.value(node.spec["value"], exclude=node.name)
        lo = c.value(node.spec["min"]) if "min" in node.spec else None
        hi = c.value(node.spec["max"]) if "max" in node.spec else None
        if lo is not None and fnum(v) < fnum(lo) or hi is not None and fnum(v) > fnum(hi):
            raise MathRefusal(f"{node.name} = {v} is outside its allowed range [{lo}, {hi}].")
        return {"value": v}

    def provides(self, node):
        return {node.name: node.data["value"]}

    def draw(self, c, node):
        if not node.style.get("show"):
            return []
        at = node.style.get("at", [0, 0])
        lab = f"{node.name} = {nice_latex(node.data['value'])}"
        return c.r.text(node.name, lab, [float(at[0]), float(at[1]), LAYER_Z["label"]], Frame(np.ones(3), dims=2),
                        node.name, size=node.style.get("label_size", 0.5), color=c.color(node))

    def describe(self, node):
        return {"value": str(node.data.get("value")), "value_latex": sp.latex(node.data.get("value", 0))}


# --------------------------------------------------------------------------- axes


@register
class Axes(Kind):
    name = "axes"
    text_fields = ("x_range", "y_range", "z_range")

    def compute(self, c, node):
        s = node.spec
        xr = [fnum(c.value(v)) for v in s.get("x_range", [-7, 7])]
        yr = [fnum(c.value(v)) for v in s.get("y_range", [-3.6, 3.6])]
        zr = [fnum(c.value(v)) for v in s["z_range"]] if s.get("z_range") else None
        for r in (xr, yr, zr):
            if r is not None and not r[1] > r[0]:
                raise MathRefusal(f"Axis range {r} must be increasing.")
        dims = 3 if zr else 2
        ranges = [xr, yr] + ([zr] if zr else [])
        scale = s.get("scale")
        if scale is None:
            if dims == 2:
                sc_ = min(14 / (xr[1] - xr[0]), 7.6 / (yr[1] - yr[0]))
            else:
                sc_ = 6 / max(r[1] - r[0] for r in ranges)
            sc = [sc_] * 3
        elif isinstance(scale, (int, float, str)):
            sc = [fnum(c.value(scale))] * 3
        else:
            sc = [fnum(c.value(v)) for v in scale] + [1.0] * (3 - len(scale))
        if any(v <= 0 for v in sc):
            raise MathRefusal("Axes scale must be positive.")
        style = s.get("tick_style", "decimal")
        styles = ([style] + ["decimal" if style == "pi" else style] * (dims - 1)) if isinstance(style, str) \
            else list(style) + ["decimal"] * (dims - len(style))
        tstep = s.get("tick_step")
        tstep = [tstep] * dims if not isinstance(tstep, (list, tuple)) else list(tstep) + [None] * dims
        steps = []
        for i, r in enumerate(ranges):
            st = fnum(c.value(tstep[i])) if tstep[i] is not None else None
            if st is None:
                target = max(2, int(round((r[1] - r[0]) * sc[i] / 1.1)))
                st = math.pi / 2 if styles[i] == "pi" else smp.nice_step(r[1] - r[0], min(target, 12))
                if styles[i] == "pi":
                    while (r[1] - r[0]) / st > 12:
                        st *= 2
            if st <= 0:
                raise MathRefusal("tick_step must be positive.")
            steps.append(st)
        loc = [fnum(c.value(v)) for v in s.get("location", [0, 0, 0])] + [0.0] * 3
        return {"dims": dims, "x_range": xr, "y_range": yr, "z_range": zr, "scale": sc, "tick_step": steps,
                "tick_style": styles, "location": loc[:3],
                "origin_world": [loc[i] for i in range(3)]}

    def describe(self, node):
        d = node.data
        return {"dims": d.get("dims"), "x_range": d.get("x_range"), "y_range": d.get("y_range"),
                "z_range": d.get("z_range"), "scale": d.get("scale"),
                "math_origin_world_position": d.get("location"),
                "note": "Math coordinates of these axes map to world as world = location + scale * math."}

    def draw(self, c, node):
        d, s = node.data, node.spec
        name = node.name
        dims = d["dims"]
        th = node.style.get("thickness", 0.022)
        color = node.style.get("color") or c.r.theme.axes
        lsize = node.style.get("label_size", 0.36)
        ranges = [d["x_range"], d["y_range"]] + ([d["z_range"]] if dims == 3 else [])
        root = c.r.empty(name, d["location"], {"kind": "axes", **{k: d[k] for k in
                         ("dims", "x_range", "y_range", "z_range", "scale", "tick_step")}}, owner=name,
                         size=0.0 if dims == 2 else 0.2)
        fr = Frame(np.asarray(d["scale"]), parent=root, dims=dims, origin=np.asarray(d["location"]))
        created = [root]
        cross = [min(max(0.0, r[0]), r[1]) for r in ranges] + [0.0] * (3 - dims)
        style3 = "flat" if dims == 2 else "3d"
        verts, faces, tips = [], [], []
        arrow = s.get("arrow_tips", True)
        for i in range(dims):
            a = np.array(cross, float)
            b = np.array(cross, float)
            a[i], b[i] = ranges[i][0], ranges[i][1]
            A, B = fr.to_local(a, "axes")[0], fr.to_local(b, "axes")[0]
            dvec = (B - A) / np.linalg.norm(B - A)
            tip = B + dvec * (0.3 if arrow else 0.0)
            tips.append(tip)
            v, f = geo.arrow_mesh(A, tip, th, style=style3, head_length=(7 * th if arrow else 0),
                                  head_width=(4 * th if arrow else 0), segments=12)
            off = len(verts)
            verts += v
            faces += [[k + off for k in fc] for fc in f]
            if dims == 2:
                c.r.occ(name).polylines.append(np.array([A[:2], tip[:2]]) + fr.origin[:2])
        created.append(c.r.mesh(f"{name}:lines", verts, faces, fr, color, owner=name))

        tick_len = s.get("tick_length", 0.14)
        tv, tf, label_items = [], [], []
        want_labels = s.get("tick_labels", True)
        for i in range(dims):
            for v in smp.tick_values(ranges[i][0], ranges[i][1], d["tick_step"][i]):
                if abs(v - cross[i]) < 1e-9 * d["tick_step"][i] and all(
                        ranges[j][0] < cross[j] < ranges[j][1] for j in range(dims) if j != i):
                    continue
                p = np.array(cross, float)
                p[i] = v
                P = fr.to_local(p, "axes")[0]
                perp = np.zeros(3)
                perp[1 if i == 0 else 0] = 1.0
                a, b = P - perp * tick_len / 2, P + perp * tick_len / 2
                if dims == 2:
                    side = np.cross([0, 0, 1], perp)
                    side = side / np.linalg.norm(side) * th * 0.45
                    base = len(tv)
                    tv += [(a - side).tolist(), (b - side).tolist(), (b + side).tolist(), (a + side).tolist()]
                    tf.append([base, base + 1, base + 2, base + 3])
                else:
                    cv, cf = geo.arrow_mesh(a, b, th * 0.8, style="3d", head_length=0, head_width=0, segments=8)
                    base = len(tv)
                    tv += cv
                    tf += [[k + base for k in fc] for fc in cf]
                if want_labels:
                    lab = smp.tick_label(v, d["tick_step"][i], d["tick_style"][i])
                    gap = tick_len / 2 + 0.07
                    if i == 0:
                        label_items.append((lab, P - perp * gap, "center", "top", i))
                    elif i == 1:
                        label_items.append((lab, P - perp * gap, "right", "center", i))
                    else:
                        label_items.append((lab, P - np.array([1, 0, 0]) * gap, "right", "center", i))
        if tv:
            created.append(c.r.mesh(f"{name}:ticks", tv, tf, fr, color, owner=name))

        if s.get("grid"):
            gv, gf = [], []
            gth = th * 0.4
            for i in range(2):
                j = 1 - i
                for v in smp.tick_values(ranges[i][0], ranges[i][1], d["tick_step"][i]):
                    a = np.zeros(3)
                    b = np.zeros(3)
                    a[i] = b[i] = v
                    a[j], b[j] = ranges[j][0], ranges[j][1]
                    A, B = fr.to_local(a, "grid")[0], fr.to_local(b, "grid")[0]
                    q, f = geo.arrow_mesh(A, B, gth, style="flat" if dims == 2 else "3d", head_length=0,
                                          head_width=0, segments=6)
                    base = len(gv)
                    gv += q
                    gf += [[k + base for k in fc] for fc in f]
            created.append(c.r.mesh(f"{name}:grid", gv, gf, fr, node.style.get("grid_color") or c.r.theme.grid,
                                    owner=name))

        if label_items:
            if dims == 2:
                parts = []
                for lab, pos, ax_, ay_, _ in label_items:
                    t = ts.anchored(c.r.typeset(lab), lsize, ax_, ay_)
                    tt = t.transformed(dx=pos[0], dy=pos[1])
                    parts.append(tt)
                    x0, y0, x1, y1 = tt.bbox
                    c.r.occ(name).rects.append((x0 + fr.origin[0], y0 + fr.origin[1], x1 + fr.origin[0],
                                                y1 + fr.origin[1]))
                merged = ts.merge(parts)
                created += c.r.text(f"{name}:ticklabels", "", [0, 0, LAYER_Z["label"]], fr, name, size=lsize,
                                    color=color, register=False, typeset=merged, raw=True)
            else:
                cam = _has_camera(c)
                for k, (lab, pos, ax_, ay_, i) in enumerate(label_items):
                    created += c.r.text(f"{name}:tick{'xyz'[i]}{k}", lab, pos, fr, name, size=lsize, color=color,
                                        align_x=ax_, align_y=ay_, plane="xz", face_camera=cam, halo=False)
        labels = [s.get("x_label", "x"), s.get("y_label", "y"), s.get("z_label", "z")][:dims]
        cam = dims == 3 and _has_camera(c)
        for i, lab in enumerate(labels):
            if not lab:
                continue
            tip = tips[i]
            if dims == 2:
                pos = tip + (np.array([0.14, 0, 0]) if i == 0 else np.array([0, 0.14, 0]))
                pos[2] = LAYER_Z["label"]
                ax_, ay_ = ("left", "center") if i == 0 else ("center", "bottom")
                created += c.r.text(f"{name}:{'xyz'[i]}label", lab, pos, fr, name, size=lsize * 1.3, color=color,
                                    align_x=ax_, align_y=ay_)
            else:
                dd = np.zeros(3)
                dd[i] = 0.3
                created += c.r.text(f"{name}:{'xyz'[i]}label", lab, tip + dd, fr, name, size=lsize * 1.3,
                                    color=color, plane="xz", face_camera=cam, halo=False)
        return created


# --------------------------------------------------------------------------- functions


class FunctionLike(Kind):
    """Nodes whose value is a function y = f(x) (functions, tangents, Taylor polynomials ...)."""

    needs_axes = 2
    colored = True

    def provides(self, node):
        e = node.data.get("expr")
        return {node.name: sp.Lambda(X, e)} if e is not None else {}

    def x_interval(self, c, node, fr: Frame):
        dom = node.data.get("domain")
        a, b = fr.lo[0], fr.hi[0]
        if dom:
            a, b = max(a, dom[0]), min(b, dom[1])
        if b <= a:
            raise MathRefusal(f"{node.name}: its domain does not intersect the visible x range.")
        return a, b

    def draw(self, c, node):
        fr = c.frame(node.spec["axes"])
        a, b = self.x_interval(c, node, fr)
        exact = self.exact_splines(c, node, fr, a, b)
        if exact is not None:
            splines, runs = exact
        else:
            splines, runs = curve_splines(c, [X, node.data["expr"]], X, a, b, fr,
                                          node.style.get("thickness", 0.04))
        if not splines:
            return []
        objs = draw_curve(c, node, splines, fr)
        node.data["runs"] = runs
        objs += graph_label(c, node, runs, fr, self.label_latex(node))
        return objs

    def exact_splines(self, c, node, fr, a, b):
        """Polynomials of degree <= 3 are drawn exactly (a cubic Bezier *is* the cubic)."""
        e = node.data["expr"]
        try:
            poly = sp.Poly(e, X)
        except sp.PolynomialError:
            return None
        if poly.degree() > 3 or poly.free_symbols - {X}:
            return None
        f = numeric(e, [X])
        runs = smp.sample_curve_t(lambda t: np.stack([t, f(t)], 1), a, b, fr.scale[:2], 1e-3, fr.lo[:2], fr.hi[:2])
        splines = [spl.spline_to_blender(self._exact_between(e, t[0], t[-1]), lambda A: fr.to_local(A, "curve"))
                   for t, P in runs if t[-1] - t[0] > 1e-12]
        return splines, [P for _, P in runs]

    @staticmethod
    def _exact_between(e, a, b):
        d = sp.diff(e, X)
        fa, fb = float(e.subs(X, a)), float(e.subs(X, b))
        da, db = float(d.subs(X, a)), float(d.subs(X, b))
        h = b - a
        co = np.array([[a, fa], [b, fb]])
        return {"co": co, "hl": np.array([[a, fa], [b - h / 3, fb - h / 3 * db]]),
                "hr": np.array([[a + h / 3, fa + h / 3 * da], [b, fb]])}

    def label_latex(self, node):
        return f"{node.name}(x) = {nice_latex(node.data['expr'])}"

    def describe(self, node):
        e = node.data.get("expr")
        return {"expression": str(e), "latex": sp.latex(e) if e is not None else None}


@register
class Function(FunctionLike):
    name = "function"
    text_fields = ("expression", "domain")

    def compute(self, c, node):
        var = sp.Symbol(node.spec.get("variable", "x"), real=True)
        e = c.parse(node.spec["expression"], (var.name,), exclude=node.name)
        extra = sorted(s_.name for s_ in e.free_symbols if s_ != var)
        if extra:
            raise MathRefusal(f"{node.name}: unknown symbol(s) {extra} in {node.spec['expression']!r}. "
                              "Create them with set_parameter, or use x as the variable.")
        e = e.subs(var, X)
        dom = node.spec.get("domain")
        dom = [fnum(c.value(v)) for v in dom] if dom else None
        if dom and not dom[1] > dom[0]:
            raise MathRefusal("domain must be [a, b] with a < b.")
        return {"expr": e, "domain": dom}


# --------------------------------------------------------------------------- points


def _intersection_points(c: Construction, a: str, b: str, fr: Frame) -> list[tuple[list, bool, str]]:
    na, nb = c.nodes.get(a), c.nodes.get(b)
    if na is None or nb is None:
        raise MathRefusal(f"Intersection needs two existing objects, got {a!r} and {b!r}.")
    ga, gb = na.data.get("geom"), nb.data.get("geom")
    la = KINDS[na.kind].provides(na).get(a)
    lb = KINDS[nb.kind].provides(nb).get(b)
    lo, hi = fr.lo[0], fr.hi[0]
    if ga is not None and gb is not None:
        pts = ga.intersection(gb)
        out = []
        for p in pts:
            if isinstance(p, sp.Point2D):
                out.append(([sp.simplify(p.x), sp.simplify(p.y)], True, "exact (sympy.geometry)"))
            else:
                raise MathRefusal(f"{a} and {b} overlap along {p}; there is no single intersection point.")
        return out
    if isinstance(la, sp.Lambda) and isinstance(lb, sp.Lambda):
        return [([k.x, k.y], k.exact, k.method) for k in kp.intersections(la.expr, lb.expr, X, lo, hi)]
    # function with a geometric object
    fnode, gnode, lam = (na, nb, la) if isinstance(la, sp.Lambda) else (nb, na, lb)
    g = gnode.data.get("geom")
    if g is None or not isinstance(lam, sp.Lambda):
        raise MathRefusal(f"Cannot intersect {a} ({na.kind}) with {b} ({nb.kind}).")
    f = lam.expr
    if isinstance(g, sp.Circle):
        eq = (X - g.center.x) ** 2 + (f - g.center.y) ** 2 - g.radius ** 2
    elif isinstance(g, sp.LinearEntity):
        A_, B_, C_ = g.coefficients
        eq = A_ * X + B_ * f + C_
    else:
        raise MathRefusal(f"Cannot intersect a function with {gnode.kind}.")
    out = []
    for r, exact, method in kp.zeros(sp.simplify(eq), X, lo, hi):
        y0 = sp.simplify(f.subs(X, r)) if exact else f.subs(X, r).evalf(30)
        if isinstance(g, sp.LinearEntity) and not isinstance(g, sp.Line2D):
            # segment / ray: keep only points that lie on it
            if exact:
                if not g.contains(sp.Point2D(r, y0)):
                    continue
            elif g.distance(sp.Point2D(sp.nsimplify(fnum(r)), sp.nsimplify(fnum(y0)))) > 1e-9:
                continue
        out.append(([r, y0], exact, method))
    return out


@register
class Point(Kind):
    name = "point"
    colored = False
    text_fields = ("coords", "x", "angle", "index", "x_range")
    ref_fields = ("on", "intersection", "of", "with", "midpoint", "center")

    def needs_axes(self, c, spec):
        return 3 if spec.get("coords") and len(spec["coords"]) == 3 else 2

    def compute(self, c, node):
        s = node.spec
        fr = c.frame(s["axes"])
        exact, method = True, "definition"
        if "coords" in s:
            P = [c.value(v, exclude=node.name) for v in s["coords"]]
        elif "on" in s:
            target = c.nodes.get(s["on"])
            if target is None:
                raise MathRefusal(f"{s['on']!r} does not exist.")
            g = target.data.get("geom")
            if "x" in s:
                x0 = c.value(s["x"], exclude=node.name)
                lam = KINDS[target.kind].provides(target).get(target.name)
                if not isinstance(lam, sp.Lambda):
                    raise MathRefusal(f"{s['on']} is not a function graph; use angle= for circles.")
                y0 = kp.require_defined(lam.expr, X, x0)
                P = [x0, y0]
                method = f"on {s['on']} at x = {x0}"
            elif "angle" in s and isinstance(g, sp.Circle):
                th = c.value(s["angle"], exclude=node.name)
                P = [sp.simplify(g.center.x + g.radius * sp.cos(th)), sp.simplify(g.center.y + g.radius * sp.sin(th))]
            else:
                raise MathRefusal("A point 'on' an object needs x= (graphs) or angle= (circles).")
        elif "intersection" in s:
            a, b = s["intersection"]
            pts = _intersection_points(c, a, b, fr)
            pts.sort(key=lambda p: (fnum(p[0][0]), fnum(p[0][1])))
            idx = int(s.get("index", 0))
            if not pts:
                raise MathRefusal(f"{a} and {b} do not intersect in the visible range.")
            if idx >= len(pts) or idx < -len(pts):
                raise MathRefusal(f"{a} and {b} have only {len(pts)} intersection(s); index {idx} does not exist.")
            P, exact, method = pts[idx]
            node_all = [coord_latex(p[0]) for p in pts]
            return {"point": list(P), "exact": exact, "method": method, "all": node_all}
        elif "of" in s and "keypoint" in s:
            f = c.function_of(s["of"])
            kind = s["keypoint"]
            xr = s.get("x_range") or [fr.lo[0], fr.hi[0]]
            a, b = fnum(c.value(xr[0])), fnum(c.value(xr[1]))
            if kind == "root":
                found = kp.roots(f, X, a, b)
            elif kind in ("local_max", "local_min", "extremum"):
                found = [k for k in kp.extrema(f, X, a, b) if kind == "extremum" or k.kind == kind]
            elif kind == "inflection":
                found = kp.inflections(f, X, a, b)
            elif kind == "y_intercept":
                y0 = kp.require_defined(f, X, sp.Integer(0))
                found = [kp.KeyPoint(sp.Integer(0), y0, "y_intercept", True, "f(0)")]
            else:
                raise MathRefusal(f"Unknown keypoint kind {kind!r}.")
            idx = int(s.get("index", 0))
            if not found:
                raise MathRefusal(f"{s['of']} has no {kind.replace('_', ' ')} in [{a:g}, {b:g}].")
            if idx >= len(found):
                raise MathRefusal(f"{s['of']} has only {len(found)} {kind}(s) in range; index {idx} does not exist.")
            k = found[idx]
            return {"point": [k.x, k.y], "exact": k.exact, "method": k.method, "keypoint_kind": k.kind}
        elif "midpoint" in s:
            A, B = (c.point_of(p) for p in s["midpoint"])
            P = [sp.simplify((u + v) / 2) for u, v in zip(A, B)]
        elif "center" in s:
            g = c.nodes[s["center"]].data.get("geom")
            if not isinstance(g, sp.Circle):
                raise MathRefusal(f"{s['center']} is not a circle.")
            P = [g.center.x, g.center.y]
        else:
            raise MathRefusal("A point needs coords, on+x, intersection, of+keypoint, midpoint or center.")
        P = [sp.nsimplify(v) if isinstance(v, sp.Float) else v for v in P]
        return {"point": P, "exact": exact, "method": method}

    def provides(self, node):
        P = node.data.get("point")
        if not P:
            return {}
        out = {f"{node.name}_x": P[0], f"{node.name}_y": P[1]}
        if len(P) > 2:
            out[f"{node.name}_z"] = P[2]
        return out

    def draw(self, c, node):
        fr = c.frame(node.spec["axes"])
        P = [fnum(v) for v in node.data["point"]]
        if fr.lo is not None and fr.flat and not (fr.lo[0] - 1e-9 <= P[0] <= fr.hi[0] + 1e-9 and
                                                   fr.lo[1] - 1e-9 <= P[1] <= fr.hi[1] + 1e-9):
            node.data["note"] = "outside the visible axes range (not drawn)"
            return []
        loc = fr.to_local(np.asarray(P, float), "point")[0]
        r = node.style.get("radius", 0.075)
        col = node.style.get("color") or c.r.theme.text
        if fr.flat:
            name = c.r.filled(node.name, [geo.bezier_circle([0, 0, 0], r)], fr, col, node.name, location=loc.tolist(),
                              meta={"kind": "point"})
            c.r.occ(node.name).rects.append((loc[0] - r + fr.origin[0], loc[1] - r + fr.origin[1],
                                             loc[0] + r + fr.origin[0], loc[1] + r + fr.origin[1]))
        else:
            v, f = geo.uv_sphere([0, 0, 0], r)
            name = c.r.mesh(node.name, v, f, fr, col, node.name, style="shaded", location=loc.tolist(), smooth=True)
        return [name] + point_label(c, node, node.data["point"], fr, node.name)

    def describe(self, node):
        P = node.data.get("point")
        if not P:
            return {}
        return {"coordinates": [str(v) for v in P], "coordinates_latex": coord_latex(P),
                "coordinates_value": [fnum(v) for v in P], "exact": node.data.get("exact"),
                "method": node.data.get("method"), **({"all_candidates": node.data["all"]} if "all" in node.data else {})}


# --------------------------------------------------------------------------- text & derivations

_VAL = re.compile(r"\\(val|approx)\{")


def fill_placeholders(c: Construction, latex: str, exclude=None) -> str:
    r"""Replace ``\val{expr}`` with the exact LaTeX value and ``\approx{expr}`` with a decimal."""
    out, i = [], 0
    for m in _VAL.finditer(latex):
        if m.start() < i:
            continue
        depth, j = 1, m.end()
        while j < len(latex) and depth:
            depth += {"{": 1, "}": -1}.get(latex[j], 0)
            j += 1
        if depth:
            raise MathRefusal(f"Unbalanced braces in placeholder of {latex!r}.")
        inner = latex[m.end():j - 1]
        e = c.parse(inner, ("x",), exclude=exclude)
        if m.group(1) == "val":
            val = nice_latex(sp.simplify(e)) if not e.free_symbols else sp.latex(e)
        else:
            val = f"{fnum(e):.6g}"
        out.append(latex[i:m.start()])
        out.append(val)
        i = j
    out.append(latex[i:])
    return "".join(out)


def placeholder_refs(c: Construction, latex: str) -> set[str]:
    refs = set()
    for m in _VAL.finditer(latex or ""):
        depth, j = 1, m.end()
        while j < len(latex) and depth:
            depth += {"{": 1, "}": -1}.get(latex[j], 0)
            j += 1
        refs |= c.env_refs(latex[m.end():j - 1])
    return refs


@register
class Text(Kind):
    name = "text"

    def refs(self, c, spec):
        out = placeholder_refs(c, spec.get("latex", ""))
        at = spec.get("at")
        if isinstance(at, str):
            out.add(at)
        elif isinstance(at, (list, tuple)):
            for v in at:
                out |= c.env_refs(str(v))
        if spec.get("axes"):
            out.add(spec["axes"])
        return out

    def compute(self, c, node):
        s = node.spec
        latex = fill_placeholders(c, s["latex"], exclude=node.name)
        t = ts.typeset(latex, mode=s.get("mode", "math"), backend=c.r.backend)
        at = s.get("at", [0, 0])
        if isinstance(at, str):
            P = [fnum(v) for v in c.point_of(at)]
        else:
            P = [fnum(c.value(v)) for v in at]
        return {"latex": latex, "at": P, "typeset": t, "backend": t.backend}

    def draw(self, c, node):
        s, d = node.spec, node.data
        fr = c.frame(s.get("axes")) if s.get("axes") else Frame(np.ones(3), dims=2)
        size = node.style.get("size", 0.7)
        loc = fr.to_local(np.asarray(d["at"] + [0.0] * (3 - len(d["at"])), float)[:3], "label")[0]
        if isinstance(s.get("at"), str) and fr.flat:
            return c.r.auto_label(node.name, d["latex"], loc, fr, node.name, size=size, color=c.color(node))
        else:
            ax, ay = node.style.get("align_x", "center"), node.style.get("align_y", "center")
        if not fr.flat and s.get("axes") is None and len(d["at"]) == 3:
            loc = np.asarray(d["at"], float)
        plane = node.style.get("plane", "xy")
        return c.r.text(node.name, d["latex"], loc, fr, node.name, size=size, color=c.color(node), align_x=ax,
                        align_y=ay, plane=plane, face_camera=node.style.get("face_camera", False),
                        mode=s.get("mode", "math"), halo=node.style.get("halo"), typeset=d["typeset"])

    def describe(self, node):
        d = node.data
        return {"latex": d.get("latex"), "backend": d.get("backend"),
                "size": [round(d["typeset"].width, 4), round(d["typeset"].height, 4)] if "typeset" in d else None}


@register
class Derivation(Kind):
    name = "derivation"
    text_fields = ("steps",)

    def compute(self, c, node):
        steps = node.spec["steps"]
        if len(steps) < 2:
            raise MathRefusal("A derivation needs at least two expressions.")
        parsed = [c.parse(s_, ("x", "y", "t")) for s_ in steps]
        checks = []
        for i in range(1, len(parsed)):
            v = ex.verify_equal(parsed[i - 1], parsed[i])
            checks.append({"step": i, "from": steps[i - 1], "to": steps[i], **v.as_dict()})
            if v.status == "not_equal":
                raise MathRefusal(f"Step {i} is false: {steps[i - 1]} ≠ {steps[i]} ({v.detail}"
                                  f"{', counterexample ' + str(v.counterexample) if v.counterexample else ''}). "
                                  "Nothing was drawn.")
        lines = [ex.display_latex(s_) for s_ in steps]
        t = ts.stack_derivation(lines, backend=c.r.backend)
        at = [fnum(c.value(v)) for v in node.spec.get("at", [0, 0])]
        return {"lines": lines, "checks": checks, "typeset": t, "at": at}

    def draw(self, c, node):
        d = node.data
        fr = Frame(np.ones(3), dims=2)
        size = node.style.get("size", 0.6)
        loc = [d["at"][0], d["at"][1], LAYER_Z["label"]]
        return c.r.text(node.name, " = ".join(d["lines"]), loc, fr, node.name, size=size, color=c.color(node),
                        align_x=node.style.get("align_x", "center"), align_y=node.style.get("align_y", "center"),
                        typeset=d["typeset"], halo=False)

    def describe(self, node):
        return {"latex_lines": node.data.get("lines"), "checks": node.data.get("checks")}


# --------------------------------------------------------------------------- curves


class CurveKind(Kind):
    colored = True

    def needs_axes(self, c, spec):
        return 3 if spec.get("z") else 2

    def comps(self, c, node) -> tuple:  # pragma: no cover
        raise NotImplementedError

    def compute(self, c, node):
        comps, var, t0, t1, *extra = self.comps(c, node)
        if not t1 > t0:
            raise MathRefusal("Parameter range must be increasing.")
        return {"comps": comps, "var": var, "t_range": [t0, t1], **(extra[0] if extra else {})}

    def draw(self, c, node):
        d = node.data
        fr = c.frame(node.spec["axes"])
        splines, runs = curve_splines(c, d["comps"], d["var"], d["t_range"][0], d["t_range"][1], fr,
                                      node.style.get("thickness", 0.04), clip=node.spec.get("clip", True))
        if not splines:
            node.data["note"] = "not visible in the axes range"
            return []
        d["runs"] = runs
        return draw_curve(c, node, splines, fr) + graph_label(c, node, runs, fr, self.label_latex(node))

    def label_latex(self, node):
        return node.name

    def describe(self, node):
        return {"components": [str(e) for e in node.data.get("comps", [])],
                "latex": [sp.latex(e) for e in node.data.get("comps", [])], "t_range": node.data.get("t_range")}


def _range(c, r, exclude=None):
    return fnum(c.value(r[0], exclude)), fnum(c.value(r[1], exclude))


@register
class Parametric(CurveKind):
    name = "parametric"
    text_fields = ("x", "y", "z", "t_range")

    def comps(self, c, node):
        s = node.spec
        var = sp.Symbol(s.get("variable", "t"), real=True)
        comps = [c.parse(s[k], (var.name,), exclude=node.name) for k in ("x", "y", "z") if s.get(k)]
        for e in comps:
            extra = sorted(q.name for q in e.free_symbols if q != var)
            if extra:
                raise MathRefusal(f"{node.name}: unknown symbols {extra}.")
        t0, t1 = _range(c, s.get("t_range", [0, "2*pi"]), node.name)
        return comps, var, t0, t1


@register
class Polar(CurveKind):
    name = "polar"
    text_fields = ("r", "theta_range")

    def needs_axes(self, c, spec):
        return 2

    def comps(self, c, node):
        th = sp.Symbol("theta", real=True)
        r = c.parse(node.spec["r"], ("theta",), exclude=node.name)
        extra = sorted(q.name for q in r.free_symbols if q != th)
        if extra:
            raise MathRefusal(f"{node.name}: unknown symbols {extra} (use theta as the variable).")
        t0, t1 = _range(c, node.spec.get("theta_range", [0, "2*pi"]), node.name)
        return [r * sp.cos(th), r * sp.sin(th)], th, t0, t1, {"r": r}

    def label_latex(self, node):
        return rf"r = {nice_latex(node.data['r'])}"


@register
class Implicit(Kind):
    name = "implicit"
    needs_axes = 2
    colored = True
    text_fields = ("equation",)

    def compute(self, c, node):
        eq = node.spec["equation"]
        if isinstance(eq, str) and ex._EQ.search(eq):
            lhs, rhs = ex._EQ.split(eq, maxsplit=1)
            e = c.parse(lhs, ("x", "y"), node.name) - c.parse(rhs, ("x", "y"), node.name)
        else:
            e = c.parse(eq, ("x", "y"), node.name)
        extra = sorted(q.name for q in e.free_symbols if q not in (X, Y))
        if extra:
            raise MathRefusal(f"{node.name}: unknown symbols {extra}.")
        return {"expr": e, "latex": sp.latex(sp.Eq(e, 0))}

    def draw(self, c, node):
        fr = c.frame(node.spec["axes"])
        f = numeric(node.data["expr"], [X, Y])
        res = int(node.spec.get("resolution", 500))
        polys = smp.implicit_curve(f, (fr.lo[0], fr.hi[0]), (fr.lo[1], fr.hi[1]), resolution=res)
        if not polys:
            node.data["note"] = "no points in the visible range"
            return []
        splines = []
        h = max(fr.hi[0] - fr.lo[0], fr.hi[1] - fr.lo[1]) / res
        for P in polys:
            closed = len(P) > 3 and np.linalg.norm(P[0] - P[-1]) < 2 * h
            Q = P[:-1] if closed and np.linalg.norm(P[0] - P[-1]) < 1e-9 else P
            splines.append(geo.polyline_spline(fr.to_local(Q, "curve"), cyclic=bool(closed)))
        node.data["runs"] = polys
        return draw_curve(c, node, splines, fr) + graph_label(c, node, polys, fr, node.data["latex"])

    def describe(self, node):
        return {"equation_latex": node.data.get("latex")}


# --------------------------------------------------------------------------- surfaces


class SurfaceKind(Kind):
    needs_axes = 3
    colored = True

    def draw(self, c, node):
        d = node.data
        fr = c.frame(node.spec["axes"])
        res = node.spec.get("resolution", 120)
        nu, nv = (res, res) if isinstance(res, int) else map(int, res)
        V, quads, uv = smp.grid_surface(d["F"], d["u_range"], d["v_range"], nu, nv, fr.lo, fr.hi)
        if not quads:
            node.data["note"] = "undefined or outside the axes box everywhere"
            return []
        local = fr.to_local(V)
        extra = {}
        cmap = node.style.get("colormap", d.get("default_colormap"))
        if cmap:
            by = node.style.get("color_by", "z")
            vals = V[:, 2] if by == "z" else uv[:, 1 if by == "v" else 0]
            extra["colors"] = np.round(geo.colormap(vals, cmap), 5).tolist()
            extra["vertex_colors"] = True
        if d.get("weld"):
            extra["merge_distance"] = float(np.max(np.ptp(local, axis=0)) or 1.0) * 1e-6
        objs = [c.r.mesh(node.name, local, quads, fr, "white" if cmap else c.color(node), node.name,
                         style=node.style.get("material", "shaded"), opacity=node.style.get("opacity", 1.0),
                         smooth=True, **extra)]
        lines = int(node.style.get("mesh_lines", 0))
        if lines and d.get("graph"):
            f = d["f"]
            polys = []
            (x0, x1), (y0, y1) = d["u_range"], d["v_range"]
            for xv in np.linspace(x0, x1, lines + 1):
                polys += smp.sample_curve(lambda t, xv=xv: np.stack([np.full_like(t, xv), t, f(np.full_like(t, xv), t)], 1),
                                          y0, y1, fr.scale, 2e-3, fr.lo, fr.hi)
            for yv in np.linspace(y0, y1, lines + 1):
                polys += smp.sample_curve(lambda t, yv=yv: np.stack([t, np.full_like(t, yv), f(t, np.full_like(t, yv))], 1),
                                          x0, x1, fr.scale, 2e-3, fr.lo, fr.hi)
            if polys:
                objs.append(c.r.curve(f"{node.name}:lines", [geo.polyline_spline(fr.to_local(P)) for P in polys], fr,
                                      node.style.get("line_color", "#000000"), 0.012, node.name))
        return objs


@register
class Surface(SurfaceKind):
    name = "surface"
    text_fields = ("expression", "x_range", "y_range")

    def compute(self, c, node):
        e = c.parse(node.spec["expression"], ("x", "y"), node.name)
        extra = sorted(q.name for q in e.free_symbols if q not in (X, Y))
        if extra:
            raise MathRefusal(f"{node.name}: unknown symbols {extra}.")
        f = numeric(e, [X, Y])
        fr = c.frame(node.spec["axes"])
        xr = _range(c, node.spec["x_range"]) if node.spec.get("x_range") else (fr.lo[0], fr.hi[0])
        yr = _range(c, node.spec["y_range"]) if node.spec.get("y_range") else (fr.lo[1], fr.hi[1])
        return {"expr": e, "f": f, "F": lambda U, V: np.stack([U, V, f(U, V)], -1), "u_range": xr, "v_range": yr,
                "graph": True, "default_colormap": "viridis"}

    def describe(self, node):
        return {"expression": str(node.data.get("expr")), "latex": sp.latex(node.data.get("expr", 0))}


@register
class ParametricSurface(SurfaceKind):
    name = "parametric_surface"
    text_fields = ("x", "y", "z", "u_range", "v_range")

    def compute(self, c, node):
        s = node.spec
        U, V = sp.symbols("u v", real=True)
        comps = [c.parse(s[k], ("u", "v"), node.name) for k in ("x", "y", "z")]
        for e in comps:
            extra = sorted(q.name for q in e.free_symbols if q not in (U, V))
            if extra:
                raise MathRefusal(f"{node.name}: unknown symbols {extra}.")
        fns = [numeric(e, [U, V]) for e in comps]
        return {"comps": comps, "F": lambda A, B: np.stack([np.broadcast_to(fn(A, B), np.shape(A)) for fn in fns], -1),
                "u_range": _range(c, s.get("u_range", [0, "2*pi"])), "v_range": _range(c, s.get("v_range", [0, "pi"])),
                "weld": True, "graph": False}

    def describe(self, node):
        return {"components": [str(e) for e in node.data.get("comps", [])]}


__all__ = ["curve_splines", "draw_curve", "point_label", "graph_label", "FunctionLike", "numeric",
           "fill_placeholders", "placeholder_refs", "_range", "T"]
