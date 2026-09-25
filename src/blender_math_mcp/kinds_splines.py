"""Spline node kinds: interpolating splines, Bezier curves (with de Casteljau), B-splines and their basis."""

from __future__ import annotations

import numpy as np
import sympy as sp

from . import geometry as geo
from . import splines as spl
from .construction import Kind, X, fnum, register
from .keypoints import MathRefusal
from .kinds_core import FunctionLike, curve_splines, draw_curve

TT = sp.Symbol("t", real=True)


def _points(c, refs):
    pts = [c.point_of(p) for p in refs]
    dims = {len(p) for p in pts}
    if len(dims) != 1:
        raise MathRefusal("All control points must have the same dimension.")
    return pts


def _dots(c, node, fr, pts, color, name="pts", r=0.06):
    splines_objs = []
    for k, p in enumerate(pts):
        loc = fr.to_local(np.asarray([fnum(v) for v in p], float), "point")[0]
        if fr.flat:
            splines_objs.append(c.r.filled(f"{node.name}:{name}{k}", [geo.bezier_circle([0, 0, 0], r)], fr, color,
                                           node.name, location=loc.tolist()))
        else:
            v, f = geo.uv_sphere([0, 0, 0], r, 8, 16)
            splines_objs.append(c.r.mesh(f"{node.name}:{name}{k}", v, f, fr, color, node.name, location=loc.tolist()))
    return splines_objs


def _polygon(c, node, fr, pts, color, name="polygon", thickness=0.018):
    P = np.array([[fnum(v) for v in p] for p in pts])
    return [c.r.curve(f"{node.name}:{name}", [geo.polyline_spline(fr.to_local(P, "grid"))], fr, color, thickness,
                      node.name, layer="grid", opacity=0.8)]


def _piece_splines(pieces_comps, fr, layer="curve"):
    """Exact Beziers for polynomial parametric pieces of degree <= 3 (one spline, continuous)."""
    co, hl, hr = [], [], []
    for k, (comps, a, b) in enumerate(pieces_comps):
        C = spl.parametric_piece_bezier(comps, TT, a, b)
        if k == 0:
            co.append(C[0])
            hl.append(C[0])
        hr.append(C[1])
        hl.append(C[2])
        co.append(C[3])
    hr.append(co[-1])
    bz = {"co": np.array(co), "hl": np.array(hl), "hr": np.array(hr)}
    return spl.spline_to_blender(bz, lambda A: fr.to_local(A, layer))


@register
class InterpolatingSpline(FunctionLike):
    name = "spline"
    ref_fields = ("points",)
    text_fields = ("end_slopes",)

    def compute(self, c, node):
        pts = _points(c, node.spec["points"])
        if len(pts[0]) != 2:
            raise MathRefusal("Interpolating splines need 2D points (x, y).")
        kind = node.spec.get("type", "natural")
        slopes = node.spec.get("end_slopes")
        slopes = [c.value(v) for v in slopes] if slopes else None
        pieces = spl.interpolating_spline(pts, kind, slopes)
        pw = sp.Piecewise(*[(p, (X >= a) & (X <= b)) for p, a, b in pieces], (sp.nan, True))
        return {"pieces": pieces, "expr": pw, "points": pts, "domain": [fnum(pieces[0][1]), fnum(pieces[-1][2])],
                "type": kind}

    def exact_splines(self, c, node, fr, a, b):
        pieces = node.data["pieces"]
        if any(sp.Poly(p, X).degree() > 3 for p, _, _ in pieces):
            return None
        # exact only when the whole spline is inside the view (otherwise fall back to clipped Hermite)
        xs = np.linspace(fnum(pieces[0][1]), fnum(pieces[-1][2]), 400)
        ys = np.array([fnum(node.data["expr"].subs(X, sp.Float(v))) for v in xs[::8]])
        if ys.min() < fr.lo[1] or ys.max() > fr.hi[1] or fnum(pieces[0][1]) < fr.lo[0] or fnum(pieces[-1][2]) > fr.hi[0]:
            return None
        bz = spl.piecewise_graph_spline(pieces, X)
        run = np.stack([xs, [fnum(node.data["expr"].subs(X, sp.Float(v))) for v in xs]], 1)
        return [spl.spline_to_blender(bz, lambda A: fr.to_local(A, "curve"))], [run]

    def draw(self, c, node):
        objs = super().draw(c, node)
        if node.style.get("show_points", True):
            fr = c.frame(node.spec["axes"])
            objs += _dots(c, node, fr, node.data["points"], c.r.theme.text)
        return objs

    def label_latex(self, node):
        return f"{node.name}(x) = " + spl.piecewise_latex(node.data["pieces"])

    def describe(self, node):
        d = node.data
        return {"type": d.get("type"), "pieces": [{"polynomial": str(p), "interval": [str(a), str(b)]}
                                                  for p, a, b in d.get("pieces", [])],
                "latex": spl.piecewise_latex(d["pieces"]) if d.get("pieces") else None}


@register
class Bezier(Kind):
    name = "bezier"
    colored = True
    ref_fields = ("points",)
    text_fields = ("t",)

    def needs_axes(self, c, spec):
        pts = spec.get("points") or []
        return 3 if pts and isinstance(pts[0], (list, tuple)) and len(pts[0]) == 3 else 2

    def compute(self, c, node):
        pts = _points(c, node.spec["points"])
        if len(pts) < 2:
            raise MathRefusal("A Bezier curve needs at least two control points.")
        comps = spl.bezier_polynomial(pts, TT)
        out = {"points": pts, "comps": comps, "degree": len(pts) - 1}
        if node.spec.get("t") is not None:
            t0 = c.value(node.spec["t"])
            if not 0 <= fnum(t0) <= 1:
                raise MathRefusal("de Casteljau parameter t must be in [0, 1].")
            out["levels"] = spl.de_casteljau(pts, t0)
            out["t"] = t0
        return out

    def provides(self, node):
        comps = node.data.get("comps")
        if not comps:
            return {}
        return {f"{node.name}_{a}": sp.Lambda(TT, e) for a, e in zip("xyz", comps)}

    def draw(self, c, node):
        fr = c.frame(node.spec["axes"])
        d = node.data
        if d["degree"] <= 3:
            C = spl.elevate_to_cubic([[fnum(v) for v in p] for p in d["points"]])
            bz = {"co": np.array([C[0], C[3]]), "hl": np.array([C[0], C[2]]), "hr": np.array([C[1], C[3]])}
            splines = [spl.spline_to_blender(bz, lambda A: fr.to_local(A, "curve"))]
        else:
            splines, _ = curve_splines(c, d["comps"], TT, 0.0, 1.0, fr, node.style.get("thickness", 0.04), clip=False)
        objs = draw_curve(c, node, splines, fr)
        if node.style.get("show_polygon", True):
            objs += _polygon(c, node, fr, d["points"], c.r.theme.axes)
            objs += _dots(c, node, fr, d["points"], c.r.theme.text)
        if "levels" in d:
            pal = c.r.theme.palette
            for li, level in enumerate(d["levels"][1:], start=1):
                colr = pal[(li + 2) % len(pal)]
                if len(level) > 1:
                    objs += _polygon(c, node, fr, level, colr, name=f"level{li}", thickness=0.02)
                objs += _dots(c, node, fr, level, colr, name=f"lv{li}_", r=0.05 if len(level) > 1 else 0.08)
        return objs

    def describe(self, node):
        d = node.data
        out = {"degree": d.get("degree"), "components": [str(e) for e in d.get("comps", [])]}
        if "levels" in d:
            out["point_at_t"] = [str(v) for v in d["levels"][-1][0]]
        return out


@register
class BSpline(Kind):
    name = "bspline"
    colored = True
    ref_fields = ("points",)
    text_fields = ("knots", "degree")

    def needs_axes(self, c, spec):
        return Bezier.needs_axes(self, c, spec)

    def compute(self, c, node):
        pts = _points(c, node.spec["points"])
        p = int(fnum(c.value(node.spec.get("degree", 3))))
        if p < 1:
            raise MathRefusal("degree must be >= 1")
        knots = [c.value(k) for k in node.spec["knots"]] if node.spec.get("knots") else \
            spl.clamped_uniform_knots(len(pts), p)
        if any(fnum(b) < fnum(a) for a, b in zip(knots, knots[1:])):
            raise MathRefusal("Knot vector must be non-decreasing.")
        if len(knots) != len(pts) + p + 1:
            raise MathRefusal(f"{len(pts)} control points of degree {p} need {len(pts) + p + 1} knots "
                              f"(got {len(knots)}).")
        spans = spl.bspline_pieces(pts, p, knots, TT)
        return {"points": pts, "degree": p, "knots": knots, "spans": spans}

    def draw(self, c, node):
        fr = c.frame(node.spec["axes"])
        d = node.data
        if d["degree"] <= 3:
            splines = [_piece_splines(d["spans"], fr)]
        else:
            splines = []
            for comps, a, b in d["spans"]:
                s_, _ = curve_splines(c, comps, TT, fnum(a), fnum(b), fr, 0.04, clip=False)
                splines += s_
        objs = draw_curve(c, node, splines, fr)
        if node.style.get("show_polygon", True):
            objs += _polygon(c, node, fr, d["points"], c.r.theme.axes)
            objs += _dots(c, node, fr, d["points"], c.r.theme.text)
        return objs

    def describe(self, node):
        d = node.data
        return {"degree": d.get("degree"), "knots": [str(k) for k in d.get("knots", [])],
                "spans": [{"interval": [str(a), str(b)], "components": [str(e) for e in comps]}
                          for comps, a, b in d.get("spans", [])]}


@register
class BSplineBasis(Kind):
    """Plots the B-spline basis functions N_{i,p}(t) on 2D axes (t on the horizontal axis)."""

    name = "bspline_basis"
    needs_axes = 2
    text_fields = ("knots", "degree", "count")

    def compute(self, c, node):
        p = int(fnum(c.value(node.spec.get("degree", 3))))
        if node.spec.get("knots"):
            knots = [c.value(k) for k in node.spec["knots"]]
        else:
            knots = spl.clamped_uniform_knots(int(fnum(c.value(node.spec.get("count", 6)))), p)
        basis = spl.bspline_basis(p, knots, TT)
        return {"degree": p, "knots": knots, "basis": basis}

    def draw(self, c, node):
        fr = c.frame(node.spec["axes"])
        d = node.data
        objs = []
        K = d["knots"]
        pal = c.r.theme.palette
        for i, N in enumerate(d["basis"]):
            pieces = []
            for a, b in zip(K[:-1], K[1:]):
                if a == b:
                    continue
                poly = spl._piece_at(N, TT, (a + b) / 2)
                pieces.append(([TT, poly], a, b))
            support = [(cm, a, b) for cm, a, b in pieces]
            if d["degree"] <= 3 and support:
                spline_ = _piece_splines(support, fr)
            else:
                spline_ = curve_splines(c, [TT, N], TT, fnum(K[0]), fnum(K[-1]), fr, 0.035, clip=True)[0][0]
            col = pal[i % len(pal)]
            objs += draw_curve(c, node, [spline_], fr, name=f"{node.name}:N{i}", color=col, thickness=0.035)
            if node.style.get("label", True):
                # label at the maximum of N_i
                ts_ = np.linspace(fnum(K[0]), fnum(K[-1]), 400)
                vals = [fnum(N.subs(TT, sp.Float(v))) for v in ts_]
                k = int(np.argmax(vals))
                anchor = fr.to_local([ts_[k], vals[k]], "label")[0]
                lab = f"N_{{{i},{d['degree']}}}"
                objs += c.r.auto_label(f"{node.name}:label{i}", lab, anchor, fr, node.name, size=0.34, color=col, prefer="n")
        return objs

    def describe(self, node):
        return {"degree": node.data.get("degree"), "knots": [str(k) for k in node.data.get("knots", [])],
                "basis": [str(b) for b in node.data.get("basis", [])]}


