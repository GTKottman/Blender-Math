"""Low-level drawing: turns computed geometry into Blender objects.

The renderer knows the theme, the coordinate frames and the 2D *occupancy map*
(everything already drawn, in world XY), which the label placer uses to put
labels where they don't collide with curves or other labels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from . import geometry as geo
from . import typeset as ts
from .blender_client import Transport
from .themes import Theme, get_theme

# Stacking order for flat (2D) scenes, as local z offsets.  Everything is drawn
# with emission shading and viewed from above, so higher z = drawn on top.
LAYER_Z = {"grid": -0.06, "region": -0.04, "axes": 0.0, "curve": 0.02, "vector": 0.03,
           "point": 0.045, "label": 0.16}

PLANES = {
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
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))  # world position of the math origin

    @property
    def flat(self) -> bool:
        return self.dims == 2

    def to_local(self, P, layer: str | None = None) -> np.ndarray:
        P = np.asarray(P, float)
        if P.ndim == 1:
            P = P[None, :]
        if P.shape[1] == 2:
            P = np.concatenate([P, np.zeros((len(P), 1))], axis=1)
        out = P * self.scale
        if self.flat and layer:
            out[:, 2] += LAYER_Z[layer]
        return out

    def to_world2d(self, P) -> np.ndarray:
        """XY world coordinates (for the occupancy map; assumes an unrotated 2D frame)."""
        L = self.to_local(P)
        return L[:, :2] + self.origin[:2]


WORLD = Frame(scale=np.ones(3), dims=2)


@dataclass
class Occupancy:
    polylines: list[np.ndarray] = field(default_factory=list)  # world XY
    rects: list[tuple[float, float, float, float]] = field(default_factory=list)


class Renderer:
    def __init__(self, transport: Transport, backend: str = "auto", theme: str | None = None):
        self.t = transport
        self.backend = backend
        self.theme: Theme = get_theme(theme)
        self.occupancy: dict[str, Occupancy] = {}
        self.labels: dict[str, dict] = {}  # auto-placed labels (re-laid out after every change)
        self.placed: dict[str, dict] = {}  # every text object: anchored typeset + location (for morphs)
        self.view_box: tuple[float, float, float, float] | None = None  # world XY of the 2D view

    # ------------------------------------------------------------------ plumbing

    def send(self, cmd: str, **params) -> Any:
        return self.t.send(cmd, {k: v for k, v in params.items() if v is not None})

    def material(self, color, style="flat", opacity: float = 1.0, **extra) -> dict:
        rgba = geo.parse_color(color)
        rgba[3] = rgba[3] * float(opacity)
        return {"color": rgba, "style": style, **extra}

    def occ(self, owner: str) -> Occupancy:
        return self.occupancy.setdefault(owner, Occupancy())

    def forget(self, owner: str) -> None:
        self.occupancy.pop(owner, None)
        for k in [k for k, v in self.labels.items() if v["owner"] == owner]:
            del self.labels[k]
        for k in [k for k, v in self.placed.items() if v["owner"] == owner]:
            del self.placed[k]

    # ------------------------------------------------------------------ primitives

    def curve(self, name: str, splines: list[dict], frame: Frame, color, thickness: float, owner: str,
              layer: str = "curve", opacity: float = 1.0, meta: dict | None = None, register=True,
              style="flat") -> str:
        """Tube curve (bezier or poly splines already in local coordinates)."""
        obj = self.send("create_curve", name=name, splines=splines, fill=False, bevel_depth=thickness / 2,
                        parent=frame.parent, material=self.material(color, style=style, opacity=opacity),
                        metadata={"node": owner, **(meta or {})})
        if register and frame.flat:
            for s in splines:
                pts = np.asarray(s.get("poly") or s.get("co"), float)[:, :2]
                self.occ(owner).polylines.append(pts + frame.origin[:2])
        return obj["name"]

    def filled(self, name: str, splines: list[dict], frame: Frame, color, owner: str, layer: str = "region",
               opacity: float = 1.0, meta: dict | None = None, location=None) -> str:
        z = LAYER_Z[layer] if frame.flat else 0.0
        loc = list(location) if location is not None else [0.0, 0.0, z]
        obj = self.send("create_curve", name=name, splines=splines, fill=True, location=loc, parent=frame.parent,
                        material=self.material(color, opacity=opacity), metadata={"node": owner, **(meta or {})})
        return obj["name"]

    def mesh(self, name: str, verts, faces, frame: Frame, color, owner: str, style="flat", opacity=1.0,
             meta: dict | None = None, location=None, **extra) -> str:
        obj = self.send("create_mesh", name=name, vertices=np.round(np.asarray(verts, float), 7).tolist(),
                        faces=faces, parent=frame.parent, location=location,
                        material=self.material(color, style=style, opacity=opacity, **{
                            k: extra.pop(k) for k in list(extra) if k == "vertex_colors"}),
                        metadata={"node": owner, **(meta or {})}, **extra)
        return obj["name"]

    def empty(self, name: str, location, meta: dict, owner: str, size: float = 0.0, parent=None) -> str:
        return self.send("create_empty", name=name, location=list(map(float, location)),
                         metadata={"node": owner, **meta}, size=size, parent=parent)["name"]

    def delete(self, names: Sequence[str] = (), node: str | None = None) -> None:
        if names or node:
            self.send("delete", names=list(names), node=node)

    # ------------------------------------------------------------------ text

    def typeset(self, latex: str, mode: str = "math") -> ts.Typeset:
        return ts.typeset(latex, mode=mode, backend=self.backend)

    def text(self, name: str, latex: str, local_pos, frame: Frame, owner: str, size: float = 0.5,
             color=None, align_x="center", align_y="center", plane="xy", face_camera=False, halo=None,
             mode="math", meta: dict | None = None, register=True, typeset: ts.Typeset | None = None,
             raw: bool = False) -> list[str]:
        """Typeset LaTeX at ``local_pos`` (frame-local).  In 2D a background-colored halo
        keeps it legible over curves and grid lines."""
        color = color or self.theme.text
        t = typeset if raw else ts.anchored(typeset or self.typeset(latex, mode), size, align_x, align_y)
        pos = np.asarray(local_pos, float)
        names = []
        if halo is None:
            halo = frame.flat
        if halo:
            # The halo is a tube of radius r around the glyph outlines, placed so that its top
            # stays below the text but above everything else in the 2D stack.
            r = size * 0.05
            hz = pos.copy()
            hz[2] = pos[2] - r - 0.01
            names.append(self.send(
                "create_curve", name=f"{name}:halo", splines=t.splines, fill=False, bevel_depth=size * 0.05,
                bevel_resolution=2, location=hz.tolist(), rotation=list(PLANES[plane]), parent=frame.parent,
                face_camera=face_camera or None, material=self.material(self.theme.background),
                metadata={"node": owner, "part": "halo"})["name"])
        names.insert(0, self.send(
            "create_curve", name=name, splines=t.splines, fill=True, location=pos.tolist(),
            rotation=list(PLANES[plane]), parent=frame.parent, face_camera=face_camera or None,
            material=self.material(color), metadata={"node": owner, "latex": latex, **(meta or {})})["name"])
        self.placed[names[0]] = {"typeset": t, "loc": pos.copy(), "parent": frame.parent, "size": size,
                                 "color": color, "plane": plane, "owner": owner}
        if register and frame.flat and plane == "xy":
            x0, y0, x1, y1 = t.bbox
            ox, oy = pos[0] + frame.origin[0], pos[1] + frame.origin[1]
            self.occ(owner).rects.append((x0 + ox, y0 + oy, x1 + ox, y1 + oy))
        return names

    # ------------------------------------------------------------------ label placement

    def auto_label(self, name: str, latex: str, anchor_local, frame: Frame, owner: str, size: float = 0.4,
                   color=None, prefer: str | None = None, gap: float = 0.14) -> list[str]:
        """A label near an anchor, positioned to avoid curves and other labels (2D).  It takes part
        in ``relayout`` so later objects can push it to a better free spot."""
        t = self.typeset(latex)
        anchor = np.asarray(anchor_local, float)
        pos = self._best(latex, anchor, frame, size, owner, prefer, gap, exclude=name)
        pos[2] = LAYER_Z["label"] if frame.flat else anchor[2]
        names = self.text(name, latex, pos, frame, owner, size=size, color=color, register=False, typeset=t)
        ta = ts.anchored(t, size, "center", "center")
        self.labels[name] = {"owner": owner, "anchor": anchor, "origin": frame.origin.copy(), "flat": frame.flat,
                             "w": ta.width, "h": ta.height, "size": size, "prefer": prefer, "gap": gap,
                             "pos": pos, "objects": names, "latex": latex}
        return names

    def _label_rects(self, exclude: str | None = None):
        out = []
        for k, L in self.labels.items():
            if k == exclude:
                continue
            x, y = L["pos"][0] + L["origin"][0], L["pos"][1] + L["origin"][1]
            out.append((x - L["w"] / 2, y - L["h"] / 2, x + L["w"] / 2, y + L["h"] / 2))
        return out

    def relayout(self) -> list[str]:
        """Re-place every auto label given everything currently drawn; move the ones that improve."""
        moved = []
        for name, L in list(self.labels.items()):
            fr = Frame(np.ones(3), dims=2 if L["flat"] else 3, origin=L["origin"])
            pos = self._best(L["latex"], L["anchor"], fr, L["size"], L["owner"], L["prefer"], L["gap"], exclude=name,
                             wh=(L["w"], L["h"]))
            pos[2] = L["pos"][2]
            if np.linalg.norm(pos[:2] - L["pos"][:2]) > 1e-6:
                r = L["size"] * 0.05
                for obj in L["objects"]:
                    z = pos[2] - r - 0.01 if obj.endswith(":halo") else pos[2]
                    try:
                        self.send("transform", name=obj, location=[float(pos[0]), float(pos[1]), float(z)])
                    except Exception:  # noqa: BLE001 - object may have been removed meanwhile
                        continue
                L["pos"] = pos
                if name in self.placed:
                    self.placed[name]["loc"] = pos.copy()
                moved.append(name)
        return moved

    def _best(self, latex, anchor, frame, size, owner, prefer, gap, exclude=None, wh=None):
        return self.place_label(latex, anchor, frame, size, owner, prefer=prefer, gap=gap, exclude=exclude, wh=wh)[0]

    def place_label(self, latex: str, anchor_local, frame: Frame, size: float, owner: str,
                    prefer: str | None = None, gap: float = 0.14, exclude: str | None = None,
                    wh: tuple[float, float] | None = None) -> tuple[np.ndarray, str, str]:
        """Choose the label position around an anchor with the least overlap.

        Candidates: 8 compass directions at two distances.  The score counts curve
        points inside the (padded) label box, overlaps with other labels, and
        leaving the visible area.  Returns ``(local_pos, align_x, align_y)``.
        """
        if wh is None:
            t = ts.anchored(self.typeset(latex), size, "center", "center")
            w, h = t.width, t.height
        else:
            w, h = wh
        A = np.asarray(anchor_local, float)
        dirs = {"ne": (1, 1), "e": (1, 0), "se": (1, -1), "s": (0, -1), "sw": (-1, -1), "w": (-1, 0),
                "nw": (-1, 1), "n": (0, 1)}
        order = list(dirs)
        if prefer in dirs:
            order.remove(prefer)
            order.insert(0, prefer)
        others = [(k, o) for k, o in self.occupancy.items() if k != owner]
        polys = [p for _, o in others for p in o.polylines] + self.occ(owner).polylines
        rects = [r for _, o in others for r in o.rects] + self._label_rects(exclude)
        best = None
        for rank, key in enumerate(order):
            dx, dy = dirs[key]
            for dist in (1.0, 1.8):
                cx = A[0] + dx * (gap * dist + w / 2)
                cy = A[1] + dy * (gap * dist + h / 2)
                wx, wy = cx + frame.origin[0], cy + frame.origin[1]
                box = (wx - w / 2 - 0.04, wy - h / 2 - 0.04, wx + w / 2 + 0.04, wy + h / 2 + 0.04)
                score = 0.0
                for P in polys:
                    inside = ((P[:, 0] > box[0]) & (P[:, 0] < box[2]) & (P[:, 1] > box[1]) & (P[:, 1] < box[3]))
                    score += 3.0 * float(inside.sum())
                    # also segments crossing the box between sparse points
                    score += 2.0 * _segments_crossing(P, box)
                for r in rects:
                    ox = max(0.0, min(box[2], r[2]) - max(box[0], r[0]))
                    oy = max(0.0, min(box[3], r[3]) - max(box[1], r[1]))
                    score += 50.0 * ox * oy / max(w * h, 1e-9)
                if self.view_box is not None:
                    vb = self.view_box
                    if box[0] < vb[0] or box[2] > vb[2] or box[1] < vb[1] or box[3] > vb[3]:
                        score += 40.0
                score += 0.2 * rank + (0.5 if dist > 1 else 0.0)
                if best is None or score < best[0]:
                    best = (score, np.array([cx, cy, A[2]]))
        return best[1], "center", "center"


def _segments_crossing(P: np.ndarray, box) -> int:
    """Number of polyline segments that intersect the box (cheap Liang-Barsky test)."""
    if len(P) < 2:
        return 0
    a, b = P[:-1], P[1:]
    x0, y0, x1, y1 = box
    # quick reject: both endpoints on the same outside side
    out = ((a[:, 0] < x0) & (b[:, 0] < x0)) | ((a[:, 0] > x1) & (b[:, 0] > x1)) | \
          ((a[:, 1] < y0) & (b[:, 1] < y0)) | ((a[:, 1] > y1) & (b[:, 1] > y1))
    cand = np.nonzero(~out)[0]
    n = 0
    for i in cand[:400]:
        p, q = a[i], b[i]
        d = q - p
        t0, t1 = 0.0, 1.0
        ok = True
        for pv, qv in ((-d[0], p[0] - x0), (d[0], x1 - p[0]), (-d[1], p[1] - y0), (d[1], y1 - p[1])):
            if pv == 0:
                if qv < 0:
                    ok = False
                    break
            else:
                r = qv / pv
                if pv < 0:
                    t0 = max(t0, r)
                else:
                    t1 = min(t1, r)
                if t0 > t1:
                    ok = False
                    break
        n += ok
    return n


def dash_polyline(P: np.ndarray, scale: np.ndarray, dash: float, gap: float) -> list[np.ndarray]:
    """Split a polyline into dashes of exact arclength (in scene units)."""
    P = np.asarray(P, float)
    Q = P * scale[: P.shape[1]]
    seg = np.linalg.norm(np.diff(Q, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    total = s[-1]
    out = []
    start = 0.0
    while start < total:
        end = min(start + dash, total)
        idx = np.nonzero((s > start) & (s < end))[0]
        pts = [np.array([np.interp(start, s, P[:, d]) for d in range(P.shape[1])])]
        pts += [P[i] for i in idx]
        pts.append(np.array([np.interp(end, s, P[:, d]) for d in range(P.shape[1])]))
        out.append(np.array(pts))
        start = end + gap
    return out
