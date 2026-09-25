"""High-level math drawing operations (the logic behind every MCP tool).

``MathStudio`` turns requests like "plot tan(x) on these axes" into verified,
exactly-computed geometry and sends it to Blender through a transport.  It is
independent of MCP so it can be tested (and scripted) directly.

Coordinate systems
------------------
Without ``axes`` every coordinate is a Blender world coordinate (1 unit = 1 m).
With ``axes='<name>'`` coordinates are *math* coordinates of that axes object:
they are mapped with the axes' per-axis scale, clipped to the axes' ranges and
parented to the axes, so moving/rotating the axes moves the whole graph.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import sympy as sp

from . import expressions as ex
from . import geometry as geo
from . import sampling as smp
from . import typeset as ts
from .blender_client import Transport

# Stacking order for flat (2D) scenes, as local z offsets.  Everything is drawn
# with emission shading and viewed from above, so higher z = drawn on top.
LAYER_Z = {"grid": -0.06, "region": -0.04, "axes": 0.0, "curve": 0.02,
           "vector": 0.03, "point": 0.04, "label": 0.08}

PLANES = {  # rotation (radians) that maps the typeset XY plane onto the named plane
    "xy": (0.0, 0.0, 0.0),
    "xz": (math.pi / 2, 0.0, 0.0),
    "yz": (math.pi / 2, 0.0, math.pi / 2),
}


@dataclass
class Frame:
    """Mapping from math coordinates to the local space of the created objects."""

    scale: np.ndarray  # (3,)
    parent: str | None = None
    lo: np.ndarray | None = None  # clip box in math coords (3,)
    hi: np.ndarray | None = None
    dims: int = 3

    @property
    def flat(self) -> bool:
        return self.dims == 2

    def to_local(self, P: np.ndarray, layer: str | None = None) -> np.ndarray:
        P = np.asarray(P, float)
        if P.ndim == 1:
            P = P[None, :]
        if P.shape[1] == 2:
            P = np.concatenate([P, np.zeros((len(P), 1))], axis=1)
        out = P * self.scale
        if self.flat and layer:
            out[:, 2] += LAYER_Z[layer]
        return out


WORLD = Frame(scale=np.ones(3))


class MathStudio:
    def __init__(self, transport: Transport, backend: str = "auto"):
        self.t = transport
        self.backend = backend

    # ------------------------------------------------------------------ plumbing

    def _send(self, cmd: str, **params) -> Any:
        return self.t.send(cmd, {k: v for k, v in params.items() if v is not None})

    def frame(self, axes: str | None) -> Frame:
        if not axes:
            return WORLD
        info = self._send("get_object", name=axes)
        meta = info.get("math") or {}
        if meta.get("kind") != "axes":
            raise ValueError(f"{axes!r} is not an axes object (create one with create_axes)")
        lo = [meta["x_range"][0], meta["y_range"][0], (meta.get("z_range") or [-1e9, 1e9])[0]]
        hi = [meta["x_range"][1], meta["y_range"][1], (meta.get("z_range") or [-1e9, 1e9])[1]]
        return Frame(np.asarray(meta["scale"], float), parent=info["name"], lo=np.asarray(lo, float),
                     hi=np.asarray(hi, float), dims=int(meta.get("dims", 3)))

    def _typeset(self, latex: str, mode: str = "math", backend: str | None = None) -> ts.Typeset:
        return ts.typeset(latex, mode=mode, backend=backend or self.backend)

    def _material(self, color, style="flat", opacity: float = 1.0, **extra) -> dict:
        rgba = geo.parse_color(color)
        rgba[3] = rgba[3] * float(opacity)
        return {"color": rgba, "style": style, **extra}

    def _label_object(self, latex: str, name: str, local_loc, size: float, color, frame: Frame,
                      align_x="center", align_y="center", plane="xy", face_camera=False,
                      mode="math", metadata=None, backend=None) -> dict:
        t = ts.anchored(self._typeset(latex, mode, backend), size, align_x, align_y)
        return self._send(
            "create_curve", name=name, splines=t.splines, fill=True,
            location=list(map(float, local_loc)), rotation=list(PLANES[plane]),
            parent=frame.parent, face_camera=face_camera or None,
            material=self._material(color), metadata=metadata or {"kind": "latex", "latex": latex},
        )

    # ------------------------------------------------------------------ status

    def status(self) -> dict:
        info: dict[str, Any] = {
            "typesetting_backend": ts.resolve_backend(self.backend),
            "latex_installed": ts.latex_available(),
        }
        try:
            info["blender"] = self._send("ping")
            info["connected"] = True
        except Exception as exc:  # noqa: BLE001 - report, don't fail
            info["connected"] = False
            info["error"] = str(exc)
        return info

    # ------------------------------------------------------------------ typesetting

    def render_latex(self, latex: str, name: str = "Equation", location: Sequence[float] = (0, 0, 0),
                     size: float = 0.8, color="white", align_x: str = "center", align_y: str = "center",
                     plane: str = "xy", rotation_deg: Sequence[float] | None = None, extrude: float = 0.0,
                     mode: str = "math", backend: str | None = None, face_camera: bool = False,
                     axes: str | None = None) -> dict:
        if plane not in PLANES:
            raise ValueError(f"plane must be one of {list(PLANES)}")
        fr = self.frame(axes)
        t = ts.anchored(self._typeset(latex, mode, backend), size, align_x, align_y)
        loc = fr.to_local(np.asarray(location, float), "label")[0]
        rot = [math.radians(a) for a in rotation_deg] if rotation_deg is not None else list(PLANES[plane])
        obj = self._send(
            "create_curve", name=name, splines=t.splines, fill=True, extrude=float(extrude) or None,
            location=loc.tolist(), rotation=rot, parent=fr.parent, face_camera=face_camera or None,
            material=self._material(color), metadata={"kind": "latex", "latex": latex, "backend": t.backend},
        )
        return {"object": obj["name"], "backend": t.backend, "width": round(t.width, 4),
                "height": round(t.height, 4), "latex": latex}

    def render_expression(self, expression: str, lhs: str | None = None, evaluate: bool = False,
                          **kw) -> dict:
        """Render a SymPy expression; the LaTeX is generated from the parsed math.

        The expression is shown as written (not auto-simplified) unless ``evaluate``.
        """
        if evaluate:
            latex = ex.to_latex(sp.simplify(ex.parse_equation(expression)))
        else:
            latex = ex.display_latex(expression)
        if lhs:
            latex = f"{lhs} = {latex}"
        res = self.render_latex(latex, name=kw.pop("name", "Expression"), **kw)
        return res

    def render_derivation(self, steps: Sequence[str], verify: bool = True, name: str = "Derivation",
                          location=(0, 0, 0), size: float = 0.7, color="white", align_x="center",
                          align_y="center", plane="xy", backend: str | None = None,
                          axes: str | None = None, face_camera: bool = False) -> dict:
        """Render ``steps[0] = steps[1] = ...`` aligned at '=', verifying every step first."""
        if len(steps) < 2:
            raise ValueError("Give at least two expressions.")
        parsed = [ex.parse(s) for s in steps]
        checks = []
        if verify:
            for i in range(1, len(parsed)):
                v = ex.verify_equal(parsed[i - 1], parsed[i])
                checks.append({"step": i, "from": steps[i - 1], "to": steps[i], **v.as_dict()})
            bad = [c for c in checks if c["status"] == "not_equal"]
            if bad:
                return {"rendered": False, "error": "Derivation contains a false step; nothing was drawn.",
                        "checks": checks}
        lines = [ex.display_latex(s) for s in steps]  # as written, not auto-simplified
        t = ts.stack_derivation(lines, backend=backend or self.backend)
        t = ts.anchored(t, size, align_x, align_y if align_y != "baseline" else "top")
        fr = self.frame(axes)
        loc = fr.to_local(np.asarray(location, float), "label")[0]
        obj = self._send(
            "create_curve", name=name, splines=t.splines, fill=True, location=loc.tolist(),
            rotation=list(PLANES[plane]), parent=fr.parent, face_camera=face_camera or None,
            material=self._material(color),
            metadata={"kind": "derivation", "steps": list(steps), "latex": lines},
        )
        return {"rendered": True, "object": obj["name"], "checks": checks, "latex": lines}

    # ------------------------------------------------------------------ axes

    def create_axes(self, name: str = "Axes", x_range=(-5, 5), y_range=(-3, 3), z_range=None,
                    scale=None, location=(0, 0, 0), tick_step=None, tick_style: str = "decimal",
                    x_label: str | None = "x", y_label: str | None = "y", z_label: str | None = "z",
                    grid: bool = False, color="white", grid_color="#3a3a3a", thickness: float = 0.025,
                    label_size: float = 0.4, tick_labels: bool = True, arrow_tips: bool = True,
                    face_camera_labels: bool = True) -> dict:
        xr, yr = [float(v) for v in x_range], [float(v) for v in y_range]
        zr = [float(v) for v in z_range] if z_range is not None else None
        for r in (xr, yr, zr):
            if r is not None and not r[1] > r[0]:
                raise ValueError(f"Range {r} must be increasing")
        dims = 3 if zr else 2
        ranges = [xr, yr] + ([zr] if zr else [])
        if scale is None:
            if dims == 2:  # equal aspect, fit a 12 x 7 region
                s = min(12 / (xr[1] - xr[0]), 7 / (yr[1] - yr[0]))
            else:
                s = 6 / max(r[1] - r[0] for r in ranges)
            sc = [s, s, s]
        elif isinstance(scale, (int, float)):
            sc = [float(scale)] * 3
        else:
            sc = [float(v) for v in scale] + [1.0] * (3 - len(scale))
        sc3 = np.asarray(sc)

        # tick_style may be one style per axis; a single 'pi' applies to the x axis only.
        if isinstance(tick_style, str):
            styles = [tick_style] + ["decimal" if tick_style == "pi" else tick_style] * (dims - 1)
        else:
            styles = list(tick_style) + ["decimal"] * (dims - len(tick_style))
        steps = []
        if tick_step is None:
            tick_step = [None] * dims
        elif isinstance(tick_step, (int, float)):
            tick_step = [float(tick_step)] * dims
        for i, r in enumerate(ranges):
            st = tick_step[i] if i < len(tick_step) else None
            if st is None:
                target = max(2, int(round((r[1] - r[0]) * sc[i] / 1.2)))  # ~one tick per 1.2 units
                st = math.pi / 2 if styles[i] == "pi" else smp.nice_step(r[1] - r[0], min(target, 10))
                if styles[i] == "pi":
                    while (r[1] - r[0]) / st > 12:
                        st *= 2
            steps.append(float(st))

        meta = {"kind": "axes", "dims": dims, "x_range": xr, "y_range": yr, "z_range": zr,
                "scale": sc, "tick_step": steps}
        root = self._send("create_empty", name=name, location=list(map(float, location)), metadata=meta,
                          display="PLAIN_AXES", size=0.0 if dims == 2 else 0.2)
        rname = root["name"]
        frame = Frame(sc3, parent=rname, dims=dims)
        created = [rname]

        # Where the axes cross: at 0 if inside the range, else at the low end.
        cross = [min(max(0.0, r[0]), r[1]) for r in ranges] + [0.0] * (3 - dims)
        style = "flat" if dims == 2 else "3d"
        verts, faces = [], []
        tips = []
        for i in range(dims):
            a = np.array(cross, float)
            b = np.array(cross, float)
            a[i], b[i] = ranges[i][0], ranges[i][1]
            A, B = frame.to_local(a, "axes")[0], frame.to_local(b, "axes")[0]
            d = (B - A) / np.linalg.norm(B - A)
            tip = B + d * (0.35 if arrow_tips else 0.0)
            tips.append(tip)
            if arrow_tips:
                v, f = geo.arrow_mesh(A, tip, thickness, style=style, head_length=6 * thickness,
                                      head_width=4 * thickness, segments=12)
            else:
                v, f = geo.arrow_mesh(A, B, thickness, style=style, head_length=0, head_width=0,
                                      segments=12)
            off = len(verts)
            verts += v
            faces += [[k + off for k in fc] for fc in f]
        obj = self._send("create_mesh", name=f"{rname}_lines", vertices=verts, faces=faces, parent=rname,
                         material=self._material(color), metadata={"kind": "axes_part", "axes": rname})
        created.append(obj["name"])

        # Ticks
        tick_len = 0.12
        tv, tf = [], []
        label_items = []  # (latex, local position, align_x, align_y, axis)
        for i in range(dims):
            for v in smp.tick_values(ranges[i][0], ranges[i][1], steps[i]):
                if dims == 2 and abs(v - cross[i]) < 1e-9 * steps[i] and all(
                        ranges[j][0] < cross[j] < ranges[j][1] for j in range(dims) if j != i):
                    continue  # skip the tick at the crossing point
                p = np.array(cross, float)
                p[i] = v
                P = frame.to_local(p, "axes")[0]
                perp = np.zeros(3)
                perp[1 if i == 0 else 0] = 1.0
                a, b = P - perp * tick_len / 2, P + perp * tick_len / 2
                if dims == 2:
                    side = np.cross([0, 0, 1], perp)
                    side = side / np.linalg.norm(side) * thickness * 0.4
                    base = len(tv)
                    tv += [(a - side).tolist(), (b - side).tolist(), (b + side).tolist(), (a + side).tolist()]
                    tf.append([base, base + 1, base + 2, base + 3])
                else:
                    cv, cf = geo.arrow_mesh(a, b, thickness * 0.8, style="3d", head_length=0, head_width=0,
                                            segments=8)
                    base = len(tv)
                    tv += cv
                    tf += [[k + base for k in fc] for fc in cf]
                if tick_labels:
                    lab = smp.tick_label(v, steps[i], styles[i])
                    gap = tick_len / 2 + 0.08
                    if i == 0:
                        label_items.append((lab, P - perp * gap, "center", "top", i))
                    elif i == 1:
                        label_items.append((lab, P - perp * gap, "right", "center", i))
                    else:
                        label_items.append((lab, P - np.array([1, 0, 0]) * gap, "right", "center", i))
        if tv:
            obj = self._send("create_mesh", name=f"{rname}_ticks", vertices=tv, faces=tf, parent=rname,
                             material=self._material(color), metadata={"kind": "axes_part", "axes": rname})
            created.append(obj["name"])

        # Grid
        if grid:
            gv, gf = [], []
            gth = thickness * 0.35
            for i in range(2):  # grid lines in the xy plane
                j = 1 - i
                for v in smp.tick_values(ranges[i][0], ranges[i][1], steps[i]):
                    a = np.zeros(3)
                    b = np.zeros(3)
                    a[i] = b[i] = v
                    a[j], b[j] = ranges[j][0], ranges[j][1]
                    A, B = frame.to_local(a, "grid")[0], frame.to_local(b, "grid")[0]
                    q, f = geo.arrow_mesh(A, B, gth, style="flat", head_length=0, head_width=0)
                    base = len(gv)
                    gv += q
                    gf += [[k + base for k in fc] for fc in f]
            obj = self._send("create_mesh", name=f"{rname}_grid", vertices=gv, faces=gf, parent=rname,
                             material=self._material(grid_color), metadata={"kind": "axes_part", "axes": rname})
            created.append(obj["name"])

        # Tick labels: merged into one object in 2D, one camera-facing object each in 3D.
        if label_items:
            if dims == 2:
                parts = []
                for lab, pos, ax_, ay_, _ in label_items:
                    t = ts.anchored(self._typeset(lab), label_size, ax_, ay_)
                    parts.append(t.transformed(dx=pos[0], dy=pos[1]))
                merged = ts.merge(parts)
                z = LAYER_Z["label"]
                obj = self._send("create_curve", name=f"{rname}_tick_labels", splines=merged.splines, fill=True,
                                 location=[0, 0, z], parent=rname, material=self._material(color),
                                 metadata={"kind": "axes_part", "axes": rname})
                created.append(obj["name"])
            else:
                for k, (lab, pos, ax_, ay_, i) in enumerate(label_items):
                    obj = self._label_object(lab, f"{rname}_tick_{'xyz'[i]}{k}", pos, label_size, color, frame,
                                             ax_, ay_, plane="xz", face_camera=face_camera_labels and
                                             self._has_camera(), metadata={"kind": "axes_part", "axes": rname})
                    created.append(obj["name"])

        # Axis names
        for i, lab in enumerate([x_label, y_label, z_label][:dims]):
            if not lab:
                continue
            tip = tips[i]
            if dims == 2:
                pos = tip + (np.array([0.15, 0, 0]) if i == 0 else np.array([0, 0.15, 0]))
                pos[2] = LAYER_Z["label"]
                ax_, ay_ = ("left", "center") if i == 0 else ("center", "bottom")
                obj = self._label_object(lab, f"{rname}_{'xyz'[i]}_label", pos, label_size * 1.25, color,
                                         frame, ax_, ay_, metadata={"kind": "axes_part", "axes": rname})
            else:
                d = np.zeros(3)
                d[i] = 0.3
                obj = self._label_object(lab, f"{rname}_{'xyz'[i]}_label", tip + d, label_size * 1.25, color,
                                         frame, "center", "center", plane="xz",
                                         face_camera=face_camera_labels and self._has_camera(),
                                         metadata={"kind": "axes_part", "axes": rname})
            created.append(obj["name"])

        return {"axes": rname, "objects": created, "scale": sc, "tick_step": steps, "dims": dims,
                "note": "Pass axes='%s' to plotting tools to draw in these math coordinates." % rname}

    def _has_camera(self) -> bool:
        try:
            return self._send("get_scene_info").get("camera") is not None
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------ curves

    def _curve_object(self, polys: list[np.ndarray], frame: Frame, name: str, color, thickness: float,
                      layer: str, metadata: dict, opacity: float = 1.0) -> dict:
        if not polys:
            raise ValueError("Nothing visible to draw in the requested range.")
        splines = [geo.polyline_spline(frame.to_local(p, layer)) for p in polys]
        return self._send("create_curve", name=name, splines=splines, fill=False, bevel_depth=thickness / 2,
                          parent=frame.parent, material=self._material(color, opacity=opacity),
                          metadata=metadata)

    def _auto_clip(self, F, a: float, b: float, dims: int) -> tuple[list, list]:
        """Robust view box for unbounded functions when no axes/range is given."""
        t = np.linspace(a, b, 4001)
        P = F(t)
        lo, hi = [], []
        for d in range(dims):
            v = P[:, d][np.isfinite(P[:, d])]
            if v.size == 0:
                raise ValueError("The function is undefined everywhere on the given range.")
            q1, q9 = np.percentile(v, [2, 98])
            mn, mx = v.min(), v.max()
            span = max(q9 - q1, 1e-9)
            lo.append(max(mn, q1 - 0.5 * span))
            hi.append(min(mx, q9 + 0.5 * span))
        return lo, hi

    def plot_function(self, expression: str, variable: str = "x", x_range=None, y_range=None,
                      axes: str | None = None, name: str | None = None, color="math_blue",
                      thickness: float = 0.04, label: str | bool | None = None, label_size: float = 0.45,
                      tolerance: float | None = None) -> dict:
        """Graph y = f(x)."""
        expr = ex.parse(expression, [variable])
        f = ex.compile_numeric(expr, [variable])
        fr = self.frame(axes)
        if x_range is None:
            x_range = (fr.lo[0], fr.hi[0]) if fr.lo is not None else (-5, 5)
        a, b = float(x_range[0]), float(x_range[1])

        def F(t):
            return np.stack([t, f(t)], axis=1)

        if y_range is not None:
            lo, hi = [a, float(y_range[0])], [b, float(y_range[1])]
        elif fr.lo is not None:
            lo, hi = [max(a, fr.lo[0]), fr.lo[1]], [min(b, fr.hi[0]), fr.hi[1]]
        else:
            lo, hi = self._auto_clip(F, a, b, 2)
            lo[0], hi[0] = a, b
        tol = tolerance or max(thickness * 0.05, 1e-4)
        polys = smp.sample_curve(F, a, b, fr.scale[:2], tol, lo, hi)
        latex = ex.to_latex(expr)
        nm = name or "Graph"
        meta = {"kind": "function", "expression": str(expr), "latex": latex, "variable": variable,
                "x_range": [a, b], "axes": axes}
        obj = self._curve_object(polys, fr, nm, color, thickness, "curve", meta)
        out = {"object": obj["name"], "latex": latex, "pieces": len(polys),
               "points": int(sum(len(p) for p in polys)),
               "clip_box": [[float(v) for v in lo], [float(v) for v in hi]]}
        if len(polys) > 1:
            out["note"] = f"Curve drawn in {len(polys)} separate pieces (discontinuities/undefined regions/clipping)."
        if label:
            lab = f"y = {latex}" if label is True else str(label)
            end = max(polys, key=lambda p: p[-1][0])[-1]
            pos = fr.to_local(end, "label")[0] + np.array([0.15, 0.0, 0.0])
            lo_obj = self._label_object(lab, f"{obj['name']}_label", pos, label_size, color, fr, "left", "bottom",
                                        face_camera=not fr.flat and self._has_camera())
            out["label_object"] = lo_obj["name"]
        return out

    def plot_parametric_curve(self, x: str, y: str, z: str | None = None, variable: str = "t",
                              t_range=(0, 2 * math.pi), axes: str | None = None, name: str | None = None,
                              color="math_yellow", thickness: float = 0.04, clip: bool = True,
                              tolerance: float | None = None) -> dict:
        comps = [x, y] + ([z] if z is not None else [])
        exprs = [ex.parse(c, [variable]) for c in comps]
        fns = [ex.compile_numeric(e, [variable]) for e in exprs]
        dims = len(fns)
        fr = self.frame(axes)

        def F(t):
            return np.stack([fn(t) for fn in fns], axis=1)

        t0, t1 = (float(ex.parse(str(v)).evalf()) for v in t_range)
        lo = hi = None
        if clip and fr.lo is not None:
            lo, hi = fr.lo[:dims], fr.hi[:dims]
        tol = tolerance or max(thickness * 0.05, 1e-4)
        polys = smp.sample_curve(F, t0, t1, fr.scale[:dims], tol, lo, hi)
        # Closed curves (circle, ellipse...): join the ends exactly.
        if len(polys) == 1 and np.allclose(polys[0][0], polys[0][-1], atol=1e-9):
            polys[0] = polys[0][:-1]
            cyclic = True
        else:
            cyclic = False
        meta = {"kind": "parametric_curve", "components": [str(e) for e in exprs],
                "latex": [ex.to_latex(e) for e in exprs], "t_range": [t0, t1], "axes": axes}
        splines = [geo.polyline_spline(fr.to_local(p, "curve"), cyclic=cyclic) for p in polys]
        obj = self._send("create_curve", name=name or "ParametricCurve", splines=splines, fill=False,
                         bevel_depth=thickness / 2, parent=fr.parent, material=self._material(color),
                         metadata=meta)
        return {"object": obj["name"], "pieces": len(polys), "closed": cyclic,
                "latex": meta["latex"], "points": int(sum(len(p) for p in polys))}

    def plot_implicit_curve(self, equation: str, x_range=None, y_range=None, axes: str | None = None,
                            name: str | None = None, color="math_green", thickness: float = 0.04,
                            resolution: int = 400) -> dict:
        eq = ex.parse_equation(equation)
        expr = eq.lhs - eq.rhs if isinstance(eq, sp.Equality) else eq
        f = ex.compile_numeric(expr, ["x", "y"])
        fr = self.frame(axes)
        xr = x_range or ((fr.lo[0], fr.hi[0]) if fr.lo is not None else (-5, 5))
        yr = y_range or ((fr.lo[1], fr.hi[1]) if fr.lo is not None else (-5, 5))
        polys = smp.implicit_curve(f, xr, yr, resolution=int(resolution))
        if not polys:
            raise ValueError(f"No points of {equation!r} found in x∈{list(xr)}, y∈{list(yr)}.")
        splines = []
        h = max(xr[1] - xr[0], yr[1] - yr[0]) / resolution
        for p in polys:
            closed = len(p) > 3 and np.linalg.norm(p[0] - p[-1]) < 1e-6 * max(h, 1e-12) * 1e3
            q = p[:-1] if closed else p
            splines.append(geo.polyline_spline(fr.to_local(q, "curve"), cyclic=bool(closed)))
        latex = ex.to_latex(eq) if isinstance(eq, sp.Equality) else ex.to_latex(expr) + " = 0"
        obj = self._send("create_curve", name=name or "ImplicitCurve", splines=splines, fill=False,
                         bevel_depth=thickness / 2, parent=fr.parent, material=self._material(color),
                         metadata={"kind": "implicit_curve", "equation": equation, "latex": latex, "axes": axes})
        return {"object": obj["name"], "pieces": len(polys), "latex": latex}

    # ------------------------------------------------------------------ surfaces

    def _surface(self, F, u_range, v_range, resolution, fr: Frame, name, color, colormap, opacity,
                 style, meta, merge: bool, color_by: str = "z") -> dict:
        nu, nv = (resolution, resolution) if isinstance(resolution, int) else map(int, resolution)
        lo = fr.lo if fr.lo is not None else None
        hi = fr.hi if fr.hi is not None else None
        V, quads, uv = smp.grid_surface(F, u_range, v_range, nu, nv, lo, hi)
        if len(quads) == 0:
            raise ValueError("Surface is undefined (or outside the axes box) everywhere.")
        local = fr.to_local(V)
        params = dict(name=name, vertices=np.round(local, 6).tolist(), faces=quads, smooth=True,
                      parent=fr.parent, metadata=meta)
        if colormap:
            vals = V[:, 2] if color_by == "z" else uv[:, 1 if color_by == "v" else 0]
            params["colors"] = np.round(geo.colormap(vals, colormap), 5).tolist()
            params["material"] = self._material("white", style=style, opacity=opacity, vertex_colors=True)
        else:
            params["material"] = self._material(color, style=style, opacity=opacity)
        if merge:
            span = float(np.max(np.ptp(local, axis=0))) or 1.0
            params["merge_distance"] = span * 1e-6
        obj = self._send("create_mesh", **params)
        return {"object": obj["name"], "vertices": len(V), "faces": len(quads)}

    def plot_surface(self, expression: str, x_range=None, y_range=None, axes: str | None = None,
                     name: str | None = None, resolution: int = 120, color="math_blue",
                     colormap: str | None = "viridis", opacity: float = 1.0, style: str = "shaded",
                     mesh_lines: int = 0, line_color="black") -> dict:
        """Graph z = f(x, y)."""
        expr = ex.parse(expression, ["x", "y"])
        f = ex.compile_numeric(expr, ["x", "y"])
        fr = self.frame(axes)
        xr = [float(v) for v in (x_range or ((fr.lo[0], fr.hi[0]) if fr.lo is not None else (-3, 3)))]
        yr = [float(v) for v in (y_range or ((fr.lo[1], fr.hi[1]) if fr.lo is not None else (-3, 3)))]

        def F(X, Y):
            return np.stack([X, Y, f(X, Y)], axis=-1)

        latex = ex.to_latex(expr)
        meta = {"kind": "surface", "expression": str(expr), "latex": latex, "axes": axes}
        out = self._surface(F, xr, yr, resolution, fr, name or "Surface", color, colormap, opacity, style,
                            meta, merge=False)
        out["latex"] = latex
        if mesh_lines:
            polys = []
            for xv in np.linspace(xr[0], xr[1], mesh_lines + 1):
                polys += smp.sample_curve(lambda t, xv=xv: np.stack([np.full_like(t, xv), t, f(np.full_like(t, xv), t)], 1),
                                          yr[0], yr[1], fr.scale, 2e-3, fr.lo, fr.hi)
            for yv in np.linspace(yr[0], yr[1], mesh_lines + 1):
                polys += smp.sample_curve(lambda t, yv=yv: np.stack([t, np.full_like(t, yv), f(t, np.full_like(t, yv))], 1),
                                          xr[0], xr[1], fr.scale, 2e-3, fr.lo, fr.hi)
            if polys:
                lo = self._curve_object(polys, fr, f"{out['object']}_lines", line_color, 0.012, "curve",
                                        {"kind": "surface_lines", "surface": out["object"]})
                out["lines_object"] = lo["name"]
        return out

    def plot_parametric_surface(self, x: str, y: str, z: str, u_range=(0, 2 * math.pi), v_range=(0, math.pi),
                                axes: str | None = None, name: str | None = None, resolution: int = 96,
                                color="math_teal", colormap: str | None = None, opacity: float = 1.0,
                                style: str = "shaded", color_by: str = "z") -> dict:
        exprs = [ex.parse(c, ["u", "v"]) for c in (x, y, z)]
        fns = [ex.compile_numeric(e, ["u", "v"]) for e in exprs]

        def F(U, V):
            return np.stack([fn(U, V) for fn in fns], axis=-1)

        ur = [float(ex.parse(str(v)).evalf()) for v in u_range]
        vr = [float(ex.parse(str(v)).evalf()) for v in v_range]
        fr = self.frame(axes)
        meta = {"kind": "parametric_surface", "components": [str(e) for e in exprs], "axes": axes}
        out = self._surface(F, ur, vr, resolution, fr, name or "ParametricSurface", color, colormap, opacity,
                            style, meta, merge=True, color_by=color_by)
        out["latex"] = [ex.to_latex(e) for e in exprs]
        return out

    # ------------------------------------------------------------------ primitives

    def draw_vector(self, end, start=(0, 0, 0), axes: str | None = None, name: str | None = None,
                    color="math_yellow", thickness: float = 0.05, label: str | None = None,
                    label_size: float = 0.5, style: str = "auto") -> dict:
        fr = self.frame(axes)
        planar = len(end) == 2 and len(start) in (2, 3) and (len(start) == 2 or start[2] == 0)
        start, end = _vec3(start), _vec3(end)
        A = fr.to_local(start, "vector")[0]
        B = fr.to_local(end, "vector")[0]
        if style == "auto":
            style = "flat" if (fr.flat or planar) else "3d"
        v, f = geo.arrow_mesh(A, B, thickness, style=style)
        obj = self._send("create_mesh", name=name or "Vector", vertices=v, faces=f, parent=fr.parent,
                         material=self._material(color, style="flat"),
                         metadata={"kind": "vector", "start": start.tolist(), "end": end.tolist(), "axes": axes})
        out = {"object": obj["name"], "length": float(np.linalg.norm(end - start))}
        if label:
            d = (B - A) / np.linalg.norm(B - A)
            nrm = np.array([-d[1], d[0], 0.0])
            if np.linalg.norm(nrm) < 1e-9:
                nrm = np.array([1.0, 0, 0])
            pos = B + d * 0.15 + nrm / np.linalg.norm(nrm) * 0.2
            pos[2] = B[2] + (LAYER_Z["label"] - LAYER_Z["vector"] if fr.flat else 0)
            lo = self._label_object(label, f"{obj['name']}_label", pos, label_size, color, fr, "center", "center",
                                    face_camera=not fr.flat and self._has_camera())
            out["label_object"] = lo["name"]
        return out

    def draw_point(self, position, axes: str | None = None, name: str | None = None, color="white",
                   radius: float = 0.08, label: str | None = None, label_size: float = 0.45,
                   label_offset=(0.15, 0.15)) -> dict:
        fr = self.frame(axes)
        P = fr.to_local(np.asarray(position, float), "point")[0]
        if fr.flat or (len(position) == 2 and not axes):
            obj = self._send("create_curve", name=name or "Point", splines=[geo.bezier_circle([0, 0, 0], radius)],
                             fill=True, location=P.tolist(), parent=fr.parent, material=self._material(color),
                             metadata={"kind": "point", "position": list(position), "axes": axes})
        else:
            v, f = geo.uv_sphere([0, 0, 0], radius)
            obj = self._send("create_mesh", name=name or "Point", vertices=v, faces=f, smooth=True,
                             location=P.tolist(), parent=fr.parent, material=self._material(color, style="shaded"),
                             metadata={"kind": "point", "position": list(position), "axes": axes})
        out = {"object": obj["name"]}
        if label:
            pos = P + np.array([label_offset[0], label_offset[1], 0.0])
            if fr.flat:
                pos[2] = LAYER_Z["label"]
            lo = self._label_object(label, f"{obj['name']}_label", pos, label_size, color, fr, "left", "bottom",
                                    face_camera=not fr.flat and self._has_camera())
            out["label_object"] = lo["name"]
        return out

    def draw_polyline(self, points, closed: bool = False, filled: bool = False, axes: str | None = None,
                      name: str | None = None, color="white", thickness: float = 0.03,
                      opacity: float = 1.0) -> dict:
        fr = self.frame(axes)
        P = np.asarray(points, float)
        if len(P) < 2:
            raise ValueError("Need at least two points.")
        if filled:
            if P.shape[1] == 3 and np.ptp(P[:, 2]) > 1e-12:
                raise ValueError("Filled polygons must be planar in XY (z constant).")
            L = fr.to_local(P, "region")
            z = float(L[0, 2])
            L[:, 2] = 0.0
            obj = self._send("create_curve", name=name or "Polygon", splines=[geo.polyline_spline(L, cyclic=True)],
                             fill=True, location=[0, 0, z], parent=fr.parent,
                             material=self._material(color, opacity=opacity),
                             metadata={"kind": "polygon", "points": P.tolist(), "axes": axes})
        else:
            obj = self._send("create_curve", name=name or "Polyline",
                             splines=[geo.polyline_spline(fr.to_local(P, "curve"), cyclic=closed)],
                             fill=False, bevel_depth=thickness / 2, parent=fr.parent,
                             material=self._material(color, opacity=opacity),
                             metadata={"kind": "polyline", "points": P.tolist(), "axes": axes})
        return {"object": obj["name"]}

    def shade_region(self, upper: str, lower: str = "0", x_range=(0, 1), variable: str = "x",
                     axes: str | None = None, name: str | None = None, color="math_blue",
                     opacity: float = 0.4) -> dict:
        """Fill the region between two graphs (e.g. the area an integral measures)."""
        fu = ex.compile_numeric(ex.parse(upper, [variable]), [variable])
        fl = ex.compile_numeric(ex.parse(lower, [variable]), [variable])
        a, b = (float(ex.parse(str(v)).evalf()) for v in x_range)
        fr = self.frame(axes)
        t = np.linspace(a, b, 2001)
        yu, yl = fu(t), fl(t)
        if not (np.all(np.isfinite(yu)) and np.all(np.isfinite(yl))):
            raise ValueError("Both bounding functions must be defined on the whole interval.")
        if fr.lo is not None:
            yu = np.clip(yu, fr.lo[1], fr.hi[1])
            yl = np.clip(yl, fr.lo[1], fr.hi[1])
        # Boundary of the region (the curves may cross; the fill follows the signed area).
        poly = np.concatenate([np.stack([t, yu], 1), np.stack([t[::-1], yl[::-1]], 1)])
        poly = smp.simplify_polyline(poly, fr.scale[:2], 1e-4)
        L = fr.to_local(poly, "region")
        z = float(L[0, 2])
        L[:, 2] = 0
        obj = self._send("create_curve", name=name or "Region", splines=[geo.polyline_spline(L, cyclic=True)],
                         fill=True, location=[0, 0, z], parent=fr.parent,
                         material=self._material(color, opacity=opacity),
                         metadata={"kind": "region", "upper": upper, "lower": lower, "x_range": [a, b], "axes": axes})
        area = sp.integrate(ex.parse(upper, [variable]) - ex.parse(lower, [variable]),
                            (sp.Symbol(variable, real=True), ex.parse(str(x_range[0])), ex.parse(str(x_range[1]))))
        return {"object": obj["name"], "signed_area": str(area), "signed_area_numeric": _safe_float(area),
                "area_latex": ex.to_latex(area)}

    # ------------------------------------------------------------------ scene

    def setup_scene(self, mode: str = "2d", background="#000000", view_width: float = 16.0,
                    center=(0, 0, 0), resolution=(1920, 1080), camera_location=None,
                    engine: str = "eevee", transparent: bool = False) -> dict:
        if mode not in ("2d", "3d"):
            raise ValueError("mode must be '2d' or '3d'")
        return self._send("setup_scene", mode=mode, background=geo.parse_color(background),
                          view_width=float(view_width), center=list(map(float, center)),
                          resolution=list(map(int, resolution)),
                          camera_location=list(map(float, camera_location)) if camera_location else None,
                          engine=engine, transparent=transparent)


def _vec3(p) -> np.ndarray:
    v = np.asarray(p, float).ravel()
    if v.size not in (2, 3):
        raise ValueError(f"Expected a 2D or 3D point, got {list(p)}")
    return np.append(v, 0.0) if v.size == 2 else v


def _safe_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
