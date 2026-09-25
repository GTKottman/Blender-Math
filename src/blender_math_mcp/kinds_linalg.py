"""Linear algebra node kinds: vectors, matrix transformations of the plane, eigenvectors, determinants."""

from __future__ import annotations

import numpy as np
import sympy as sp

from . import geometry as geo
from .construction import Kind, coord_latex, fnum, nice_latex, register
from .fields import eigen_real
from .keypoints import MathRefusal
from .kinds_calculus import clip_line
from .render import Frame


def parse_matrix(c, M) -> sp.Matrix:
    if isinstance(M, str) and M in c.nodes and "matrix" in c.nodes[M].data:
        return c.nodes[M].data["matrix"]
    rows = [[c.value(v) for v in row] for row in M]
    if len({len(r) for r in rows}) != 1:
        raise MathRefusal("Matrix rows must have equal length.")
    return sp.Matrix(rows)


def matrix_refs(c, spec) -> set:
    M = spec.get("matrix")
    if isinstance(M, str):
        return {M} if M in c.nodes else set()
    out = set()
    for row in M or []:
        for v in row:
            out |= c.env_refs(str(v))
    return out


def draw_arrow(c, node, name, start, end, fr: Frame, color, thickness, label=None, label_size=0.45):
    """Arrow whose tip is exactly at ``end``; object origin at the tail (so it can grow from it)."""
    A = fr.to_local(np.asarray(start, float), "vector")[0]
    B = fr.to_local(np.asarray(end, float), "vector")[0]
    if np.linalg.norm(B - A) < 1e-12:
        return []
    style = "flat" if fr.flat else "3d"
    v, f = geo.arrow_mesh(np.zeros(3), B - A, thickness, style=style)
    objs = [c.r.mesh(name, v, f, fr, color, node.name, location=A.tolist(), style="flat")]
    if fr.flat:
        c.r.occ(node.name).polylines.append(np.array([A[:2], B[:2]]) + fr.origin[:2])
    if label:
        if fr.flat:
            objs += c.r.auto_label(f"{name}:label", label, B, fr, node.name, size=label_size, color=color, prefer=None)
        else:
            objs += c.r.text(f"{name}:label", label, B + np.array([0.15, 0, 0.15]), fr, node.name, size=label_size,
                             color=color, plane="xz", halo=False)
    return objs


@register
class Vector(Kind):
    name = "vector"
    colored = True
    text_fields = ("components",)
    ref_fields = ("tail", "from", "to")

    def needs_axes(self, c, spec):
        comps = spec.get("components")
        return 3 if comps is not None and len(comps) == 3 else 2

    def compute(self, c, node):
        s = node.spec
        if "components" in s:
            v = [c.value(x, node.name) for x in s["components"]]
            tail = c.point_of(s["tail"]) if "tail" in s else [sp.Integer(0)] * len(v)
        elif "from" in s and "to" in s:
            A, B = c.point_of(s["from"]), c.point_of(s["to"])
            v = [sp.simplify(b - a) for a, b in zip(A, B)]
            tail = A
        else:
            raise MathRefusal("A vector needs components (and optional tail) or from/to points.")
        if len(tail) != len(v):
            raise MathRefusal("Tail and components must have the same dimension.")
        norm = sp.simplify(sp.sqrt(sum(x ** 2 for x in v)))
        return {"vector": v, "tail": tail, "head": [a + b for a, b in zip(tail, v)], "norm": norm}

    def provides(self, node):
        v = node.data.get("vector")
        if not v:
            return {}
        out = {f"{node.name}_x": v[0], f"{node.name}_y": v[1]}
        if len(v) > 2:
            out[f"{node.name}_z"] = v[2]
        return out

    def draw(self, c, node):
        fr = c.frame(node.spec["axes"])
        d = node.data
        lab = node.style.get("label", True)
        if lab is True:
            lab = rf"\vec{{{node.name}}}"
        elif lab == "components":
            lab = r"\begin{pmatrix}" + r"\\".join(nice_latex(x) for x in d["vector"]) + r"\end{pmatrix}"
        return draw_arrow(c, node, node.name, [fnum(x) for x in d["tail"]], [fnum(x) for x in d["head"]], fr,
                          c.color(node), node.style.get("thickness", 0.05), lab or None,
                          node.style.get("label_size", 0.45))

    def describe(self, node):
        d = node.data
        return {"components": [str(x) for x in d.get("vector", [])], "norm": str(d.get("norm")),
                "tail": coord_latex(d["tail"]) if d else None}


@register
class MatrixTransform(Kind):
    """Shows the linear map x -> A x: the transformed grid, basis vectors and chosen vectors/polygons.

    This is also the object the 'apply_matrix' animation morphs (identity -> A)."""

    name = "matrix_transform"
    needs_axes = 2
    colored = True
    ref_fields = ("apply_to",)

    def refs(self, c, spec):
        return Kind.refs(self, c, spec) | matrix_refs(c, spec)

    def compute(self, c, node):
        A = parse_matrix(c, node.spec["matrix"])
        if A.shape != (2, 2):
            raise MathRefusal("matrix_transform draws 2x2 matrices (plane maps).")
        det = sp.simplify(A.det())
        eig, cplx = eigen_real(A)
        return {"matrix": A, "det": det, "eigen": eig, "complex_eigenvalues": cplx}

    def provides(self, node):
        return {f"{node.name}_det": node.data["det"]} if "det" in node.data else {}

    def grid_lines(self, c, node, fr, Mf):
        """Lines of the integer grid mapped by Mf (2x2 float), clipped exactly to the axes box."""
        step = node.spec.get("grid_step", 1)
        lines = []
        span = max(abs(fr.lo[0]), abs(fr.hi[0]), abs(fr.lo[1]), abs(fr.hi[1]))
        # enough source lines to cover the image of the box
        try:
            inv = np.linalg.inv(Mf)
            reach = span * np.linalg.norm(inv, 2) * 1.5
        except np.linalg.LinAlgError:
            reach = span * 1.5
        k = int(np.ceil(reach / step))
        for i in range(-k, k + 1):
            for P0, D in ((np.array([i * step, 0.0]), np.array([0.0, 1.0])), (np.array([0.0, i * step]),
                                                                                 np.array([1.0, 0.0]))):
                Q0, QD = Mf @ P0, Mf @ D
                if np.linalg.norm(QD) < 1e-12:
                    continue
                seg = clip_line(Q0, QD, fr)
                if seg is not None:
                    lines.append(seg)
        return lines

    def draw(self, c, node):
        fr = c.frame(node.spec["axes"])
        d = node.data
        Mf = np.array(d["matrix"].tolist(), float)
        objs = []
        if node.spec.get("grid", True):
            lines = self.grid_lines(c, node, fr, Mf)
            if lines:
                sps = [geo.polyline_spline(fr.to_local(L, "grid")) for L in lines]
                objs.append(c.r.curve(f"{node.name}:grid", sps, fr, c.color(node), node.style.get("thickness", 0.02),
                                      node.name, layer="grid", register=False,
                                      opacity=node.style.get("grid_opacity", 0.8)))
        if node.spec.get("basis", True):
            e1, e2 = Mf @ [1, 0], Mf @ [0, 1]
            objs += draw_arrow(c, node, f"{node.name}:i", [0, 0], e1, fr, c.r.theme.palette[2], 0.055, r"\hat{\imath}")
            objs += draw_arrow(c, node, f"{node.name}:j", [0, 0], e2, fr, c.r.theme.palette[3], 0.055, r"\hat{\jmath}")
        for ref in node.spec.get("apply_to", []) or []:
            tgt = c.nodes.get(ref)
            if tgt is None:
                raise MathRefusal(f"{ref!r} does not exist.")
            if "vector" in tgt.data:
                tail = np.array([fnum(x) for x in tgt.data["tail"]])
                head = np.array([fnum(x) for x in tgt.data["head"]])
                objs += draw_arrow(c, node, f"{node.name}:{ref}", Mf @ tail, Mf @ head, fr, c.color(tgt), 0.05,
                                   rf"A\vec{{{ref}}}")
            elif "points" in tgt.data and tgt.kind == "polygon":
                P = np.array([[fnum(p[0]), fnum(p[1])] for p in tgt.data["points"]]) @ Mf.T
                L = fr.to_local(P)
                L[:, 2] = 0
                objs.append(c.r.filled(f"{node.name}:{ref}", [geo.polyline_spline(L, cyclic=True)], fr, c.color(tgt),
                                       node.name, opacity=c.r.theme.fill_opacity))
        if node.spec.get("show_determinant"):
            sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float) @ Mf.T
            L = fr.to_local(sq)
            L[:, 2] = 0
            objs.append(c.r.filled(f"{node.name}:det", [geo.polyline_spline(L, cyclic=True)], fr, c.r.theme.palette[1],
                                   node.name, opacity=0.35))
            cen = fr.to_local(sq.mean(0), "label")[0]
            objs += c.r.auto_label(f"{node.name}:detlabel", rf"\det = {nice_latex(d['det'])}", cen, fr, node.name,
                                   size=0.4, gap=0.0)
        return objs

    def describe(self, node):
        d = node.data
        if not d:
            return {}
        return {"matrix": str(d["matrix"].tolist()), "matrix_latex": sp.latex(d["matrix"]),
                "determinant": str(d["det"]),
                "eigen": [{"eigenvalue": str(v), "multiplicity": m, "eigenvectors": [str(list(w)) for w in vs]}
                          for v, m, vs in d["eigen"]],
                "complex_eigenvalues": [str(v) for v in d["complex_eigenvalues"]]}


@register
class Eigenvectors(Kind):
    name = "eigenvectors"
    needs_axes = 2
    colored = True

    def refs(self, c, spec):
        return Kind.refs(self, c, spec) | matrix_refs(c, spec)

    def compute(self, c, node):
        A = parse_matrix(c, node.spec["matrix"])
        if A.shape != (2, 2):
            raise MathRefusal("eigenvectors draws 2x2 matrices.")
        eig, cplx = eigen_real(A)
        if not eig:
            raise MathRefusal(f"The matrix has no real eigenvalues ({', '.join(map(str, cplx))}); no eigenvector "
                              "lines exist in the real plane.")
        return {"matrix": A, "eigen": eig, "complex": cplx}

    def draw(self, c, node):
        fr = c.frame(node.spec["axes"])
        objs = []
        k = 0
        for val, _mult, vecs in node.data["eigen"]:
            for w in vecs:
                D = np.array([fnum(w[0]), fnum(w[1])])
                seg = clip_line(np.zeros(2), D, fr)
                if seg is None:
                    continue
                col = c.r.theme.palette[(4 + k) % len(c.r.theme.palette)]
                objs.append(c.r.curve(f"{node.name}:line{k}", [geo.polyline_spline(fr.to_local(seg, "curve"))], fr, col,
                                      node.style.get("thickness", 0.03), node.name, opacity=0.9))
                end = seg[np.argmax(seg @ D)]
                anchor = fr.to_local(end, "label")[0]
                lab = rf"\lambda = {nice_latex(val)}"
                objs += c.r.auto_label(f"{node.name}:label{k}", lab, anchor, fr, node.name, size=0.4, color=col)
                k += 1
        return objs

    def describe(self, node):
        return {"eigen": [{"eigenvalue": str(v), "eigenvectors": [str(list(w)) for w in vs]}
                          for v, _m, vs in node.data.get("eigen", [])]}


