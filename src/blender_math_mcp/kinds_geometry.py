"""Euclidean geometry node kinds, computed exactly with sympy.geometry."""

from __future__ import annotations

import math

import numpy as np
import sympy as sp

from . import geometry as geo
from .construction import Kind, X, coord_latex, fnum, nice_latex, register
from .keypoints import MathRefusal
from .kinds_calculus import LineLike, line_equation_latex, line_objects
from .render import Frame


def P2(c, ref):
    p = c.point_of(ref)
    if len(p) != 2:
        raise MathRefusal("Plane geometry needs 2D points.")
    return sp.Point2D(p[0], p[1])


def _point_pair(v) -> bool:
    """[A, B] of two point references (names or coordinate lists), not a single [x, y] point."""
    return isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(p, (str, list, tuple)) for p in v) \
        and not all(isinstance(p, str) and _is_number(p) for p in v)


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _line_geom(c, ref):
    n = c.nodes.get(ref)
    g = n.data.get("geom") if n else None
    if not isinstance(g, sp.LinearEntity):
        raise MathRefusal(f"{ref!r} is not a line, ray or segment.")
    return g


def arc_splines(center, r, a0, a1, max_step=math.pi / 4) -> dict:
    """Circular arc as cubic Beziers (<= 45 deg each: radial error < 5e-6 * r)."""
    n = max(1, int(math.ceil(abs(a1 - a0) / max_step - 1e-12)))
    angs = np.linspace(a0, a1, n + 1)
    k = 4 / 3 * math.tan((a1 - a0) / n / 4) * r
    cx, cy = center
    co, hl, hr = [], [], []
    for i, a in enumerate(angs):
        p = np.array([cx + r * math.cos(a), cy + r * math.sin(a)])
        tang = np.array([-math.sin(a), math.cos(a)])
        co.append(p)
        hl.append(p - tang * k)
        hr.append(p + tang * k)
    return {"co": np.array(co), "hl": np.array(hl), "hr": np.array(hr)}


def full_circle(center, r) -> dict:
    sp_ = arc_splines(center, r, 0, 2 * math.pi)
    return {"co": sp_["co"][:-1], "hl": sp_["hl"][:-1], "hr": sp_["hr"][:-1]}


def _to_blender(sp_, fr: Frame, layer, cyclic):
    from .splines import spline_to_blender

    return spline_to_blender(sp_, lambda A: fr.to_local(A, layer), cyclic=cyclic)


@register
class Line(LineLike):
    name = "line"
    text_fields = ("slope", "equation", "through", "point", "perpendicular_bisector", "angle_bisector")
    ref_fields = ("through", "point", "perpendicular", "parallel", "perpendicular_bisector", "angle_bisector")

    def compute(self, c, node):
        s = node.spec
        if "equation" in s:
            eq = s["equation"]
            from .expressions import _EQ

            if not _EQ.search(eq):
                raise MathRefusal("equation must contain '=' (e.g. 'y = 2x + 1' or '2x + 3y = 6').")
            lhs, rhs = _EQ.split(eq, maxsplit=1)
            Y = sp.Symbol("y", real=True)
            e = sp.expand(c.parse(lhs, ("x", "y")) - c.parse(rhs, ("x", "y")))
            poly = sp.Poly(e, X, Y)
            if poly.total_degree() != 1:
                raise MathRefusal(f"{eq!r} is not a linear equation, so it is not a line.")
            a, b = poly.coeff_monomial(X), poly.coeff_monomial(Y)
            k = poly.coeff_monomial(1)
            if b != 0:
                p1 = sp.Point2D(0, -k / b)
            else:
                p1 = sp.Point2D(-k / a, 0)
            geom = sp.Line2D(p1, sp.Point2D(p1.x - b, p1.y + a))
        elif "through" in s and _point_pair(s["through"]) and "perpendicular" not in s and "parallel" not in s:
            A, B = P2(c, s["through"][0]), P2(c, s["through"][1])
            if A == B:
                raise MathRefusal("Two equal points do not determine a line.")
            geom = sp.Line2D(A, B)
        elif "slope" in s:
            A = P2(c, s["point"])
            m = c.value(s["slope"])
            geom = sp.Line2D(A, sp.Point2D(A.x + 1, A.y + m))
        elif "perpendicular" in s or "parallel" in s:
            base = _line_geom(c, s.get("perpendicular") or s.get("parallel"))
            if "through" not in s:
                raise MathRefusal("perpendicular/parallel lines need through=<point>.")
            A = P2(c, s["through"])
            geom = base.perpendicular_line(A) if "perpendicular" in s else base.parallel_line(A)
            geom = sp.Line2D(geom.p1, geom.p2)
        elif "perpendicular_bisector" in s:
            A, B = (P2(c, p) for p in s["perpendicular_bisector"])
            if A == B:
                raise MathRefusal("The perpendicular bisector needs two different points.")
            geom = sp.Segment2D(A, B).perpendicular_bisector()
        elif "angle_bisector" in s:
            A, B, C = (P2(c, p) for p in s["angle_bisector"])
            u = (A - B) / A.distance(B)
            v = (C - B) / C.distance(B)
            d = u + v
            if sp.simplify(d.x) == 0 and sp.simplify(d.y) == 0:
                d = sp.Point2D(-u.y, u.x)  # straight angle: bisector is perpendicular
            geom = sp.Line2D(B, B + d)
        else:
            raise MathRefusal("A line needs equation, through [A, B], point+slope, perpendicular/parallel+through, "
                              "perpendicular_bisector or angle_bisector.")
        a, b, k = geom.coefficients
        expr = None if b == 0 else sp.expand(-(a * X + k) / b)
        slope = sp.zoo if b == 0 else sp.simplify(-a / b)
        return {"geom": geom, "expr": expr, "slope": slope}


@register
class Segment(Kind):
    name = "segment"
    needs_axes = 2
    ref_fields = ("points",)

    def compute(self, c, node):
        A, B = (P2(c, p) for p in node.spec["points"])
        if A == B:
            raise MathRefusal("A segment needs two different points.")
        g = sp.Segment2D(A, B)
        return {"geom": g, "length": sp.simplify(g.length)}

    def provides(self, node):
        return {node.name: node.data["length"]} if "length" in node.data else {}

    def draw(self, c, node):
        fr = c.frame(node.spec["axes"])
        objs, pts = line_objects(c, node, node.data["geom"], fr)
        lab = node.style.get("label")
        if lab and pts is not None:
            if lab in (True, "length"):
                lab = nice_latex(node.data["length"])
            mid = pts.mean(0)
            anchor = fr.to_local(mid, "label")[0]
            objs += c.r.auto_label(f"{node.name}:label", lab, anchor, fr, node.name,
                                   size=node.style.get("label_size", 0.38), color=c.color(node))
        return objs

    def describe(self, node):
        return {"length": str(node.data.get("length")), "length_latex": nice_latex(node.data.get("length", 0))}


@register
class Ray(LineLike):
    name = "ray"
    ref_fields = ("from", "through")

    def compute(self, c, node):
        A, B = P2(c, node.spec["from"]), P2(c, node.spec["through"])
        if A == B:
            raise MathRefusal("A ray needs two different points.")
        return {"geom": sp.Ray2D(A, B), "expr": None}

    def provides(self, node):
        return {}


@register
class Circle(Kind):
    name = "circle"
    needs_axes = 2
    colored = True
    text_fields = ("radius",)
    ref_fields = ("center", "through", "incircle")

    def compute(self, c, node):
        s = node.spec
        if "center" in s and "radius" in s:
            C = P2(c, s["center"])
            r = c.value(s["radius"])
            if not fnum(r) > 0:
                raise MathRefusal(f"Radius must be positive (got {r}).")
            g = sp.Circle(C, r)
        elif "center" in s and "through" in s:
            C, A = P2(c, s["center"]), P2(c, s["through"])
            if C == A:
                raise MathRefusal("The point must differ from the center.")
            g = sp.Circle(C, sp.simplify(C.distance(A)))
        elif "through" in s and len(s["through"]) == 3:
            A, B, C = (P2(c, p) for p in s["through"])
            if sp.Point.is_collinear(A, B, C):
                raise MathRefusal("The three points are collinear; no circle passes through them.")
            g = sp.Triangle(A, B, C).circumcircle
        elif "incircle" in s:
            A, B, C = (P2(c, p) for p in s["incircle"])
            if sp.Point.is_collinear(A, B, C):
                raise MathRefusal("Degenerate triangle (collinear points) has no incircle.")
            g = sp.Triangle(A, B, C).incircle
        else:
            raise MathRefusal("A circle needs center+radius, center+through, through [A, B, C] or incircle [A, B, C].")
        return {"geom": g, "center": [sp.simplify(g.center.x), sp.simplify(g.center.y)], "radius": sp.simplify(g.radius)}

    def provides(self, node):
        d = node.data
        if not d:
            return {}
        return {f"{node.name}_r": d["radius"], f"{node.name}_x": d["center"][0], f"{node.name}_y": d["center"][1]}

    def draw(self, c, node):
        fr = c.frame(node.spec["axes"])
        d = node.data
        if not np.allclose(fr.scale[0], fr.scale[1]):
            node.data["note"] = "axes have unequal x/y scale, so the circle appears as an ellipse (correctly)"
        sp_ = full_circle([fnum(d["center"][0]), fnum(d["center"][1])], fnum(d["radius"]))
        from .kinds_core import draw_curve

        objs = draw_curve(c, node, [_to_blender(sp_, fr, "curve", True)], fr)
        if node.style.get("fill"):
            L = _to_blender(sp_, fr, "region", True)
            for k in ("co", "hl", "hr"):
                L[k] = [[p[0], p[1], 0.0] for p in L[k]]
            objs.append(c.r.filled(f"{node.name}:fill", [L], fr, c.color(node), node.name,
                                   opacity=node.style.get("fill_opacity", c.r.theme.fill_opacity)))
        return objs

    def describe(self, node):
        d = node.data
        g = d.get("geom")
        return {"center": coord_latex(d["center"]) if d else None, "radius": str(d.get("radius")),
                "equation": sp.latex(g.equation(x=sp.Symbol("x"), y=sp.Symbol("y"))) + " = 0" if g is not None else None}


@register
class Polygon(Kind):
    name = "polygon"
    needs_axes = 2
    colored = True
    ref_fields = ("points",)

    def compute(self, c, node):
        pts = [P2(c, p) for p in node.spec["points"]]
        if len(pts) < 3:
            raise MathRefusal("A polygon needs at least three points.")
        poly = sp.Polygon(*pts)
        if not isinstance(poly, sp.Polygon):
            raise MathRefusal("The points are collinear or repeated; they do not form a polygon.")
        area = sp.simplify(sp.Abs(poly.area))
        return {"geom": poly, "points": [[p.x, p.y] for p in pts], "area": area,
                "perimeter": sp.simplify(poly.perimeter)}

    def provides(self, node):
        return {node.name: node.data["area"]} if "area" in node.data else {}

    def draw(self, c, node):
        fr = c.frame(node.spec["axes"])
        P = np.array([[fnum(p[0]), fnum(p[1])] for p in node.data["points"]])
        L = fr.to_local(P)
        L[:, 2] = 0
        col = c.color(node)
        objs = [c.r.filled(node.name, [geo.polyline_spline(L, cyclic=True)], fr, col, node.name,
                           opacity=node.style.get("fill_opacity", c.r.theme.fill_opacity))]
        objs.append(c.r.curve(f"{node.name}:edges", [geo.polyline_spline(fr.to_local(P, "curve"), cyclic=True)], fr,
                              col, node.style.get("thickness", 0.03), node.name))
        return objs

    def describe(self, node):
        d = node.data
        return {"area": str(d.get("area")), "area_latex": nice_latex(d.get("area", 0)),
                "perimeter": str(d.get("perimeter"))}


@register
class Angle(Kind):
    name = "angle"
    needs_axes = 2
    colored = True
    ref_fields = ("points",)

    def compute(self, c, node):
        A, B, C = (P2(c, p) for p in node.spec["points"])
        if A == B or C == B:
            raise MathRefusal("An angle needs its arm points to differ from the vertex.")
        u, v = A - B, C - B
        cosv = sp.simplify((u.x * v.x + u.y * v.y) / (sp.sqrt(u.x ** 2 + u.y ** 2) * sp.sqrt(v.x ** 2 + v.y ** 2)))
        ang = sp.simplify(sp.acos(cosv))
        a0 = math.atan2(fnum(u.y), fnum(u.x))
        a1 = math.atan2(fnum(v.y), fnum(v.x))
        cross = fnum(u.x * v.y - u.y * v.x)
        if node.spec.get("directed"):
            # counter-clockwise from BA to BC (may exceed pi)
            if cross < 0:
                ang = 2 * sp.pi - ang
            sweep = (a1 - a0) % (2 * math.pi)
        else:
            sweep = (a1 - a0) % (2 * math.pi)
            if sweep > math.pi:
                a0, sweep = a1, 2 * math.pi - sweep
        return {"value": ang, "vertex": [B.x, B.y], "start": a0, "sweep": sweep,
                "right": sp.simplify(ang - sp.pi / 2) == 0}

    def provides(self, node):
        return {node.name: node.data["value"]} if "value" in node.data else {}

    def draw(self, c, node):
        fr = c.frame(node.spec["axes"])
        d = node.data
        V = np.array([fnum(d["vertex"][0]), fnum(d["vertex"][1])])
        r = node.style.get("radius", 0.5) / float(fr.scale[0])
        col = c.color(node)
        objs = []
        if d["right"]:
            a0 = d["start"]
            e1 = np.array([math.cos(a0), math.sin(a0)]) * r * 0.7
            e2 = np.array([math.cos(a0 + d["sweep"]), math.sin(a0 + d["sweep"])]) * r * 0.7
            sq = np.array([V + e1, V + e1 + e2, V + e2])
            objs.append(c.r.curve(node.name, [geo.polyline_spline(fr.to_local(sq, "curve"))], fr, col,
                                  node.style.get("thickness", 0.025), node.name))
            fillpoly = np.array([V, V + e1, V + e1 + e2, V + e2])
            L = fr.to_local(fillpoly)
            L[:, 2] = 0
            objs.append(c.r.filled(f"{node.name}:fill", [geo.polyline_spline(L, cyclic=True)], fr, col, node.name,
                                   opacity=node.style.get("fill_opacity", 0.25)))
        else:
            arc = arc_splines(V, r, d["start"], d["start"] + d["sweep"])
            objs.append(c.r.curve(node.name, [_to_blender(arc, fr, "curve", False)], fr, col,
                                  node.style.get("thickness", 0.025), node.name))
            sector = {"co": np.concatenate([[V], arc["co"]]), "hl": np.concatenate([[V], arc["hl"]]),
                      "hr": np.concatenate([[V], arc["hr"]])}
            sector["hr"][0] = V + (arc["co"][0] - V) / 3
            sector["hl"][1] = arc["co"][0] - (arc["co"][0] - V) / 3
            last = sector["co"][-1]
            sector["hr"][-1] = last + (V - last) / 3
            sector["hl"][0] = V + (last - V) / 3
            L = _to_blender(sector, fr, "region", True)
            for k in ("co", "hl", "hr"):
                L[k] = [[p[0], p[1], 0.0] for p in L[k]]
            objs.append(c.r.filled(f"{node.name}:fill", [L], fr, col, node.name,
                                   opacity=node.style.get("fill_opacity", 0.25)))
        lab = node.style.get("label", True)
        if lab:
            if lab is True:
                lab = self.value_latex(d["value"], node.style.get("unit", "deg"))
            mid = d["start"] + d["sweep"] / 2
            size = node.style.get("label_size", 0.36)
            anchor = V + np.array([math.cos(mid), math.sin(mid)]) * r
            dirs = ["e", "ne", "n", "nw", "w", "sw", "s", "se"]
            prefer = dirs[int(round((mid % (2 * math.pi)) / (math.pi / 4))) % 8]
            objs += c.r.auto_label(f"{node.name}:label", lab, fr.to_local(anchor, "label")[0], fr, node.name,
                                   size=size, color=col, prefer=prefer)
        return objs

    @staticmethod
    def value_latex(v, unit):
        if unit == "rad":
            return nice_latex(v)
        deg = sp.simplify(v * 180 / sp.pi)
        s = nice_latex(deg)
        return (s + r"^\circ") if not s.startswith(r"\approx") else s + r"^\circ"

    def describe(self, node):
        v = node.data.get("value")
        if v is None:
            return {}
        return {"radians": str(v), "degrees": str(sp.simplify(v * 180 / sp.pi)), "value": fnum(v),
                "right_angle": bool(node.data.get("right"))}


__all__ = ["arc_splines", "full_circle", "line_equation_latex"]
