"""Studio: one object tying together the construction graph, scene/camera/layout and animation.

This is what the MCP tools call.  It is independent of MCP so it can be used
from scripts and tests (with any transport).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from . import geometry as geo
from . import keypoints as kp
from . import kinds  # noqa: F401 - registers node kinds
from .animation import Animator
from .blender_client import Transport
from .construction import Construction, X, coord_latex
from .keypoints import MathRefusal
from .render import Renderer
from .themes import get_theme
from . import typeset as ts


class Studio:
    def __init__(self, transport: Transport, backend: str = "auto", theme: str | None = None):
        self.r = Renderer(transport, backend=backend, theme=theme)
        self.c = Construction(self.r)
        self.anim = Animator(self.c)
        self.mode = "2d"

    # ------------------------------------------------------------------ status / scene

    def status(self) -> dict:
        info: dict[str, Any] = {"typesetting_backend": ts.resolve_backend(self.r.backend),
                                "latex_installed": ts.latex_available(), "theme": self.r.theme.name}
        try:
            info["blender"] = self.r.send("ping")
            info["connected"] = True
            self.c.ensure_loaded()
            info["objects"] = len(self.c.nodes)
        except Exception as exc:  # noqa: BLE001
            info["connected"] = False
            info["error"] = str(exc)
        return info

    def setup_scene(self, mode: str = "2d", theme: str | None = None, background=None, view_width: float = 16.0,
                    center=(0, 0, 0), resolution=(1920, 1080), camera_location=None, engine: str = "eevee") -> dict:
        if mode not in ("2d", "3d"):
            raise MathRefusal("mode must be '2d' or '3d'")
        self.c.ensure_loaded()
        redraw = False
        if theme and theme != self.r.theme.name:
            self.r.theme = get_theme(theme)
            redraw = bool(self.c.nodes)
        bg = background or self.r.theme.background
        res = self.r.send("setup_scene", mode=mode, background=geo.parse_color(bg), view_width=float(view_width),
                          center=[float(v) for v in center], resolution=[int(v) for v in resolution],
                          camera_location=[float(v) for v in camera_location] if camera_location else None,
                          engine=engine)
        self.mode = mode
        aspect = resolution[0] / resolution[1]
        if mode == "2d":
            self.r.view_box = (center[0] - view_width / 2, center[1] - view_width / aspect / 2,
                               center[0] + view_width / 2, center[1] + view_width / aspect / 2)
        if redraw:
            self.c.redraw_all()
        self.c.save()
        res.update({"theme": self.r.theme.name, "view_box": self.r.view_box,
                    "note": "Math origin (0,0) of default axes is the world origin; 1 math unit = 1 Blender unit "
                            "unless the axes have another scale."})
        return res

    def frame_view(self, names: Sequence[str] | None = None, center_on: str = "content", margin: float = 0.06):
        objs = None
        if names:
            objs = [o for n in names for o in (self.c.nodes[n].objects if n in self.c.nodes else [n])
                    if not o.endswith(":halo")]
        res = self.r.send("frame", names=objs, margin=margin, center=[0, 0, 0] if center_on == "origin" else None)
        if res.get("view_box"):
            self.r.view_box = tuple(res["view_box"])
            self.r.relayout()
        return res

    # ------------------------------------------------------------------ layout

    def bounds(self, names: Sequence[str]) -> dict:
        out = {}
        for n in names:
            objs = [o for o in (self.c.nodes[n].objects if n in self.c.nodes else [n]) if not o.endswith(":halo")]
            if not objs:
                out[n] = None
                continue
            b = self.r.send("bounds", names=objs)
            lo = np.min([v["min"] for v in b.values()], axis=0)
            hi = np.max([v["max"] for v in b.values()], axis=0)
            out[n] = {"min": lo.tolist(), "max": hi.tolist(), "center": ((lo + hi) / 2).tolist(),
                      "size": (hi - lo).tolist()}
        return out

    def align(self, name: str, reference: str, side: str = "below", gap: float = 0.3, align: str = "center") -> dict:
        """Place a text/derivation next to another object (by bounding boxes), updating its definition."""
        node = self.c.nodes.get(name)
        if node is None or node.kind not in ("text", "derivation"):
            raise MathRefusal("align works on text and derivation objects (their position is free).")
        if node.spec.get("axes"):
            raise MathRefusal("This text is placed in axes coordinates; move it by updating 'at'.")
        b = self.bounds([name, reference])
        me, ref = b[name], b[reference]
        if me is None or ref is None:
            raise MathRefusal("Both objects must be visible.")
        w, h = me["size"][0], me["size"][1]
        cx, cy = me["center"][0], me["center"][1]
        at = node.data["at"]  # current anchor (object location) in world coords
        off = (cx - at[0], cy - at[1])  # anchor -> bbox center
        rx0, ry0, rx1, ry1 = ref["min"][0], ref["min"][1], ref["max"][0], ref["max"][1]
        rcx, rcy = (rx0 + rx1) / 2, (ry0 + ry1) / 2
        tx = {"left": rx0 + w / 2, "right": rx1 - w / 2}.get(align, rcx)
        ty = {"bottom": ry0 + h / 2, "top": ry1 - h / 2}.get(align, rcy)
        target = {"below": (tx, ry0 - gap - h / 2), "above": (tx, ry1 + gap + h / 2),
                  "left": (rx0 - gap - w / 2, ty), "right": (rx1 + gap + w / 2, ty),
                  "center": (rcx, rcy)}.get(side)
        if target is None:
            raise MathRefusal("side must be below, above, left, right or center.")
        new_at = [round(target[0] - off[0], 6), round(target[1] - off[1], 6)]
        return self.c.update(name, {"at": new_at})

    # ------------------------------------------------------------------ key points

    def find_points(self, kind: str, of: str, with_: str | None = None, x_range=None, create: bool = True,
                    prefix: str | None = None, label: str | bool = "coords") -> dict:
        self.c.ensure_loaded()
        fnode = self.c.nodes.get(of)
        axes = fnode.spec.get("axes") if fnode is not None else None
        axes = axes or self.c.resolve_axes({}, 2)
        fr = self.c.frame(axes)
        a, b = (float(self.c.value(v)) for v in x_range) if x_range else (fr.lo[0], fr.hi[0])
        if kind == "intersections":
            if not with_:
                raise MathRefusal("intersections needs with=<other object>.")
            spec_base = {"intersection": [of, with_]}
            from .kinds_core import _intersection_points

            found = [{"point": p, "exact": e, "method": m} for p, e, m in
                     sorted(_intersection_points(self.c, of, with_, fr), key=lambda q: (float(q[0][0]), float(q[0][1])))]
        else:
            f = self.c.function_of(of)
            fn = {"roots": kp.roots, "extrema": kp.extrema, "inflections": kp.inflections}.get(kind)
            if fn is None:
                raise MathRefusal("kind must be roots, extrema, inflections or intersections.")
            kps = fn(f, X, a, b)
            found = [{"point": [k.x, k.y], "exact": k.exact, "method": k.method, "kind": k.kind} for k in kps]
            spec_base = {"of": of}
        created = []
        if create:
            prefix = prefix or f"{of}_{kind[:-1] if kind.endswith('s') else kind}"
            counters: dict[str, int] = {}
            for i, item in enumerate(found):
                if kind == "intersections":
                    spec = {**spec_base, "index": i}
                else:
                    kk = item.get("kind", kind)
                    kpkind = {"roots": "root", "inflections": "inflection"}.get(kind, kk)
                    idx = counters.get(kpkind, 0)
                    counters[kpkind] = idx + 1
                    spec = {**spec_base, "keypoint": kpkind, "index": idx}
                    if x_range:
                        spec["x_range"] = list(x_range)
                name = f"{prefix}{i + 1}"
                while name in self.c.nodes:
                    name += "_"
                self.c.add("point", name, {**spec, "axes": axes}, {"label": label})
                created.append(name)
        return {"count": len(found), "points": [{"x": str(p["point"][0]), "y": str(p["point"][1]),
                                                  "latex": coord_latex(p["point"]), "exact": p["exact"],
                                                  "method": p["method"], **({"kind": p["kind"]} if "kind" in p else {})}
                                                 for p in found], "created": created}

    # ------------------------------------------------------------------ graph passthrough

    def add(self, kind: str, name: str | None, spec: dict, style: dict | None = None) -> dict:
        return self.c.add(kind, name, spec, style)

    def clear(self) -> dict:
        self.c.clear()
        self.anim.cursor = self.anim.end = 0.0
        return {"cleared": True}


