"""Animation: write-on, draw-on, fades, morphs (curves, equations, matrix maps), camera moves, video.

The animator keeps a timeline cursor (seconds).  Each ``play`` call animates from
the cursor for ``duration`` seconds and advances the cursor, like a script:

    create axes (1 s) -> draw f (2 s) -> write its equation (1.5 s) -> transform f into g ...

Correctness during motion
-------------------------
* Function morphs blend pointwise: the curve at time s is exactly the graph of
  (1 - s) f + s g, sampled where both are defined, clipped by a shader mask at
  the axes box (never clamped).  Breaks where either function jumps or is undefined.
* Matrix animations use absolute shape keys at exact intermediate maps
  M_s = (1 - s) I + s A; points move on straight lines, which is exactly the path
  a point follows under M_s, so vector tips are exact at every frame.
* Equation morphs move matching glyphs (same symbol) to their new places and fade
  the others, so the formula is legible at the start and end, never garbled.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from . import geometry as geo
from . import sampling as smp
from . import typeset as ts
from .construction import KINDS, Construction, X, fnum
from .keypoints import MathRefusal
from .render import LAYER_Z


class Animator:
    def __init__(self, construction: Construction):
        self.c = construction
        self.fps = 30
        self.cursor = 0.0  # seconds
        self.end = 0.0
        self._helpers = 0
        self.history: list[dict] = []  # every play() call, replayed when the graph is redrawn
        self._replaying = False

    # ------------------------------------------------------------------ helpers

    @property
    def r(self):
        return self.c.r

    def frame(self, seconds: float) -> int:
        return int(round(seconds * self.fps)) + 1

    def _objects(self, target: str) -> list[str]:
        node = self.c.nodes.get(target)
        if node is not None:
            return list(node.objects)
        return [target]

    def _info(self, names: Sequence[str]) -> dict:
        return self.r.send("describe_objects", names=list(names)) if names else {}

    def _keys(self, items: list[dict]) -> None:
        if items:
            self.r.send("keyframes", items=items)

    def _span(self, start, duration):
        s = self.cursor if start is None else float(start)
        return s, s + float(duration)

    def _advance(self, t_end: float, advance: bool = True):
        self.end = max(self.end, t_end)
        if advance:
            self.cursor = t_end
        self.r.send("timeline", fps=self.fps, frame_start=1, frame_end=self.frame(self.end + 0.5))

    def _interp(self, easing: str) -> str:
        return {"smooth": "BEZIER", "linear": "LINEAR", "step": "CONSTANT"}.get(easing, "BEZIER")

    def _helper_name(self, base: str) -> str:
        self._helpers += 1
        return f"anim{self._helpers}:{base}"

    # ------------------------------------------------------------------ public API

    def configure(self, fps: int | None = None, reset: bool = False) -> dict:
        if reset:
            self.r.send("clear_animation")
            self.cursor = self.end = 0.0
            self.history.clear()
        if fps:
            self.fps = int(fps)
        self.r.send("timeline", fps=self.fps, frame_start=1, frame_end=max(self.frame(self.end + 0.5), 2))
        return self.state()

    def replay(self) -> list[str]:
        """Rebuild all animation after objects were redrawn (keyframes live on the Blender objects)."""
        if not self.history or self._replaying:
            return []
        skipped = []
        self._replaying = True
        try:
            self.r.send("clear_animation")
            cursor = self.cursor
            self.end = 0.0
            for h in self.history:
                if any(t not in self.c.nodes for t in h["targets"]):
                    skipped.append(f"{h['action']} {h['targets']} (object deleted)")
                    continue
                try:
                    self.play(h["action"], h["targets"], h["duration"], h["start"], h["lag"], h["easing"],
                              h["advance"], **h["opts"])
                except MathRefusal as exc:
                    skipped.append(f"{h['action']} {h['targets']}: {exc}")
            self.cursor = cursor
        finally:
            self._replaying = False
        return skipped

    def state(self) -> dict:
        return {"fps": self.fps, "cursor_seconds": self.cursor, "end_seconds": self.end,
                "cursor_frame": self.frame(self.cursor), "end_frame": self.frame(self.end)}

    def play(self, action: str, targets: Sequence[str] | str | None = None, duration: float = 1.0,
             start: float | None = None, lag: float = 0.0, easing: str = "smooth", advance: bool = True,
             **opts) -> dict:
        if isinstance(targets, str):
            targets = [targets]
        targets = list(targets or [])
        for t in targets:
            if t not in self.c.nodes and not opts.get("allow_objects"):
                raise MathRefusal(f"Unknown object {t!r}.")
        t0, t1 = self._span(start, duration)
        fn = {
            "create": self._create, "draw": self._create, "write": self._write, "fade_in": self._fade_in,
            "fade_out": self._fade_out, "grow": self._grow, "indicate": self._indicate,
            "transform": self._transform, "apply_matrix": self._apply_matrix,
            "camera_move": self._camera_move, "camera_orbit": self._camera_orbit, "wait": self._wait,
        }.get(action)
        if fn is None:
            raise MathRefusal(f"Unknown animation {action!r}.")
        info = fn(targets, t0, t1, lag=lag, interp=self._interp(easing), **opts) or {}
        if not self._replaying:
            self.history.append({"action": action, "targets": targets, "duration": duration, "start": t0,
                                 "lag": lag, "easing": easing, "advance": advance, "opts": opts})
        self._advance(t1 + lag * max(0, len(targets) - 1), advance)
        return {"action": action, "targets": targets, "start_seconds": t0, "end_seconds": t1,
                "frames": [self.frame(t0), self.frame(t1)], **info, "timeline": self.state()}

    # ------------------------------------------------------------------ building blocks

    def _hide_until(self, names, t, interp="CONSTANT"):
        f = self.frame(t)
        return [{"object": n, "path": "visible", "keys": [[1, 0], [f, 1]] if f > 1 else [[f, 1]],
                 "include_children": False} for n in names]

    def _alpha(self, names, pairs, interp="BEZIER"):
        return [{"object": n, "path": "alpha", "keys": [[self.frame(t), a] for t, a in pairs], "interpolation": interp,
                 "include_children": False} for n in names]

    def _wait(self, targets, t0, t1, **_):
        return {}

    def _create(self, targets, t0, t1, lag=0.0, interp="BEZIER", **_):
        items = []
        for k, tgt in enumerate(targets):
            s0 = t0 + k * lag
            s1 = t1 + k * lag
            names = self._objects(tgt)
            info = self._info(names)
            node = self.c.nodes.get(tgt)
            main = [n for n in names if not (":label" in n or n.endswith(":halo") or ":tick" in n)]
            labels = [n for n in names if n not in main]
            items += self._hide_until(names, s0)
            for n in main:
                i = info.get(n, {})
                if i.get("type") == "CURVE" and not i.get("filled") and i.get("bevel", 0) > 0:
                    items.append({"object": n, "path": "data.bevel_factor_end",
                                  "keys": [[self.frame(s0), 0.0], [self.frame(s1), 1.0]], "interpolation": interp})
                elif i.get("type") == "MESH" and (node is not None and node.kind == "vector" and n == tgt
                                                  or n.endswith((":i", ":j"))):
                    items.append({"object": n, "path": "scale", "keys": [[self.frame(s0), [0.0, 0.0, 0.0]],
                                                                         [self.frame(s1), [1.0, 1.0, 1.0]]],
                                  "interpolation": interp})
                elif i.get("type") == "EMPTY":
                    continue
                else:
                    items += self._alpha([n], [(s0, 0.0), (s1, 1.0)], interp)
            if labels:
                ls = s0 + 0.6 * (s1 - s0)
                items += self._hide_until(labels, ls)
                items += self._alpha(labels, [(ls, 0.0), (s1 + 0.3 * (s1 - s0), 1.0)], interp)
        self._keys(items)
        return {}

    def _write(self, targets, t0, t1, lag=0.0, interp="BEZIER", **_):
        """Handwriting effect: glyph outlines are drawn, then the fill fades in and the outline fades out."""
        items = []
        helpers = []
        for k, tgt in enumerate(targets):
            s0, s1 = t0 + k * lag, t1 + k * lag
            names = self._objects(tgt)
            info = self._info(names)
            items += self._hide_until(names, s0)
            for n in names:
                i = info.get(n, {})
                if n.endswith(":halo"):
                    items += self._alpha([n], [(s0 + 0.5 * (s1 - s0), 0.0), (s1, 1.0)], interp)
                    continue
                if i.get("type") == "CURVE" and i.get("filled"):
                    size = self.r.placed.get(n, {}).get("size", 0.5)
                    stroke = self._helper_name(n)
                    col = self.r.placed.get(n, {}).get("color") or self.r.theme.text
                    self.r.send("stroke_copy", name=n, new_name=stroke, bevel_depth=size * 0.012, node=tgt,
                                material=self.r.material(col))
                    helpers.append(stroke)
                    sm = s0 + 0.65 * (s1 - s0)
                    items += self._hide_until([stroke], s0)
                    items.append({"object": stroke, "path": "data.bevel_factor_end",
                                  "keys": [[self.frame(s0), 0.0], [self.frame(sm), 1.0]], "interpolation": interp})
                    items += self._alpha([stroke], [(sm, 1.0), (s1, 0.0)], interp)
                    items.append({"object": stroke, "path": "visible", "keys": [[self.frame(s1), 0]],
                                  "include_children": False})
                    items += self._alpha([n], [(s0 + 0.45 * (s1 - s0), 0.0), (s1, 1.0)], interp)
                elif i.get("type") == "CURVE" and i.get("bevel", 0) > 0:
                    items.append({"object": n, "path": "data.bevel_factor_end",
                                  "keys": [[self.frame(s0), 0.0], [self.frame(s1), 1.0]], "interpolation": interp})
                else:
                    items += self._alpha([n], [(s0, 0.0), (s1, 1.0)], interp)
        self._keys(items)
        return {"helpers": helpers}

    def _fade_in(self, targets, t0, t1, lag=0.0, interp="BEZIER", **_):
        items = []
        for k, tgt in enumerate(targets):
            names = self._objects(tgt)
            s0, s1 = t0 + k * lag, t1 + k * lag
            items += self._hide_until(names, s0) + self._alpha(names, [(s0, 0.0), (s1, 1.0)], interp)
        self._keys(items)

    def _fade_out(self, targets, t0, t1, lag=0.0, interp="BEZIER", **_):
        items = []
        for k, tgt in enumerate(targets):
            names = self._objects(tgt)
            s0, s1 = t0 + k * lag, t1 + k * lag
            items += self._alpha(names, [(s0, 1.0), (s1, 0.0)], interp)
            items += [{"object": n, "path": "visible", "keys": [[self.frame(s1), 0]], "include_children": False}
                      for n in names]
        self._keys(items)

    def _grow(self, targets, t0, t1, lag=0.0, interp="BEZIER", **_):
        items = []
        for k, tgt in enumerate(targets):
            names = self._objects(tgt)
            s0, s1 = t0 + k * lag, t1 + k * lag
            info = self._info(names[:1])
            sc = info.get(names[0], {}).get("scale", [1, 1, 1])
            items += self._hide_until(names, s0)
            items.append({"object": names[0], "path": "scale", "keys": [[self.frame(s0), [0.0, 0.0, 0.0]],
                                                                       [self.frame(s1), sc]], "interpolation": interp})
        self._keys(items)

    def _indicate(self, targets, t0, t1, lag=0.0, interp="BEZIER", scale: float = 1.25, **_):
        items = []
        for k, tgt in enumerate(targets):
            s0, s1 = t0 + k * lag, t1 + k * lag
            for n in self._objects(tgt):
                if n.endswith(":halo"):
                    continue
                info = self._info([n]).get(n, {})
                sc = info.get("scale", [1, 1, 1])
                items.append({"object": n, "path": "scale", "interpolation": interp,
                              "keys": [[self.frame(s0), sc], [self.frame((s0 + s1) / 2), [v * scale for v in sc]],
                                       [self.frame(s1), sc]]})
        self._keys(items)

    # ------------------------------------------------------------------ transforms

    def _transform(self, targets, t0, t1, lag=0.0, interp="BEZIER", **opts):
        if len(targets) != 2:
            raise MathRefusal("transform needs [source, target].")
        src, dst = (self.c.nodes[t] for t in targets)
        ks = KINDS[src.kind]
        if "expr" in src.data and "expr" in dst.data and src.data["expr"] is not None and dst.data["expr"] is not None \
                and src.spec.get("axes") == dst.spec.get("axes") and hasattr(ks, "x_interval"):
            return self._morph_functions(src, dst, t0, t1, interp)
        if src.kind in ("text", "derivation") and dst.kind in ("text", "derivation"):
            return self._morph_text(src, dst, t0, t1, interp)
        if "runs" in src.data and "runs" in dst.data and src.spec.get("axes") == dst.spec.get("axes"):
            return self._morph_curves(src, dst, t0, t1, interp)
        # fallback: cross-fade
        self._fade_out([src.name], t0, t1, interp=interp)
        self._fade_in([dst.name], t0, t1, interp=interp)
        return {"method": "cross-fade (no shape correspondence between these kinds)"}

    def _swap(self, src_names, dst_names, helper, t0, t1):
        f0, f1 = self.frame(t0), self.frame(t1)
        items = [{"object": n, "path": "visible", "keys": [[f0, 0]], "include_children": False} for n in src_names]
        items += [{"object": n, "path": "visible", "keys": ([[1, 0]] if f1 > 1 else []) + [[f1, 1]],
                   "include_children": False} for n in dst_names]
        items += [{"object": h, "path": "visible", "keys": ([[1, 0]] if f0 > 1 else []) + [[f0, 1], [f1, 0]],
                   "include_children": False} for h in helper]
        return items

    def _morph_functions(self, src, dst, t0, t1, interp):
        """Pointwise morph (1-s) f + s g on the common visible x-range."""
        fr = self.c.frame(src.spec["axes"])
        a = max(KINDS[src.kind].x_interval(self.c, src, fr)[0], KINDS[dst.kind].x_interval(self.c, dst, fr)[0])
        b = min(KINDS[src.kind].x_interval(self.c, src, fr)[1], KINDS[dst.kind].x_interval(self.c, dst, fr)[1])
        if b <= a:
            raise MathRefusal("The two functions have no common x-range to morph over.")
        f = _num(src.data["expr"])
        g = _num(dst.data["expr"])
        big = 50 * max(abs(fr.lo[1]), abs(fr.hi[1]), 1.0)
        ts_, brk = _union_samples(f, g, a, b, fr)
        F, G = f(ts_), g(ts_)
        ok = np.isfinite(F) & np.isfinite(G)
        F = np.clip(F, -big, big)
        G = np.clip(G, -big, big)
        runs = []
        cur = []
        for i in range(len(ts_)):
            if not ok[i]:
                if len(cur) > 1:
                    runs.append(cur)
                cur = []
                continue
            cur.append(i)
            if i < len(brk) and brk[i]:
                if len(cur) > 1:
                    runs.append(cur)
                cur = []
        if len(cur) > 1:
            runs.append(cur)
        if not runs:
            raise MathRefusal("The functions are nowhere both defined on the common range.")
        base = [geo.polyline_spline(fr.to_local(np.stack([ts_[r], F[r]], 1), "curve")) for r in runs]
        key = [fr.to_local(np.stack([ts_[r], G[r]], 1), "curve") for r in runs]
        name = self._helper_name(f"{src.name}->{dst.name}")
        lo_l = fr.to_local(fr.lo[:2])[0]
        hi_l = fr.to_local(fr.hi[:2])[0]
        mat = self.r.material(src.style.get("color") or self.r.theme.text)
        mat["clip_box"] = [lo_l[0], hi_l[0], lo_l[1], hi_l[1]]
        self.r.send("create_curve", name=name, splines=base, fill=False,
                    bevel_depth=src.style.get("thickness", 0.04) / 2, parent=fr.parent, material=mat,
                    metadata={"node": "anim", "anim_helper": True})
        pts = [p.tolist() for k_ in key for p in k_]
        self.r.send("shape_keys", object=name, keys=[{"name": "target", "co": pts}], relative=True)
        items = self._swap(self._objects(src.name), self._objects(dst.name), [name], t0, t1)
        items.append({"object": name, "path": "shape:target", "keys": [[self.frame(t0), 0.0], [self.frame(t1), 1.0]],
                      "interpolation": interp})
        items += self._alpha([name], [(t0, 1.0), (t1, 1.0)])
        self._keys(items)
        return {"method": "exact pointwise blend (1-s)·f + s·g", "helper": name}

    def _morph_curves(self, src, dst, t0, t1, interp):
        """Arclength-matched morph of general curves (single-piece curves)."""
        rs, rd = src.data["runs"], dst.data["runs"]
        if len(rs) != 1 or len(rd) != 1:
            self._fade_out([src.name], t0, t1, interp=interp)
            self._fade_in([dst.name], t0, t1, interp=interp)
            return {"method": "cross-fade (curves have several pieces)"}
        fr = self.c.frame(src.spec["axes"])
        A, B = _resample(np.asarray(rs[0], float), 400), _resample(np.asarray(rd[0], float), 400)
        name = self._helper_name(f"{src.name}->{dst.name}")
        self.r.send("create_curve", name=name, splines=[geo.polyline_spline(fr.to_local(A, "curve"))], fill=False,
                    bevel_depth=src.style.get("thickness", 0.04) / 2, parent=fr.parent,
                    material=self.r.material(src.style.get("color") or self.r.theme.text),
                    metadata={"node": "anim", "anim_helper": True})
        self.r.send("shape_keys", object=name, keys=[{"name": "target", "co": fr.to_local(B, "curve").tolist()}])
        items = self._swap(self._objects(src.name), self._objects(dst.name), [name], t0, t1)
        items.append({"object": name, "path": "shape:target", "keys": [[self.frame(t0), 0.0], [self.frame(t1), 1.0]],
                      "interpolation": interp})
        self._keys(items)
        return {"method": "arclength-matched morph", "helper": name}

    def _morph_text(self, src, dst, t0, t1, interp):
        """Move glyphs that exist in both formulas; fade the rest (TransformMatchingTex-like)."""
        ps, pd = self.r.placed.get(src.name), self.r.placed.get(dst.name)
        if ps is None or pd is None:
            raise MathRefusal("Both formulas must be drawn (visible) to morph between them.")
        gs = [g.transformed(ps["loc"][0], ps["loc"][1], 1.0) for g in ps["typeset"].glyphs]
        gd = [g.transformed(pd["loc"][0], pd["loc"][1], 1.0) for g in pd["typeset"].glyphs]
        used = set()
        pairs, fade_out, fade_in = [], [], []
        for j, g in enumerate(gd):
            cands = [i for i, h in enumerate(gs) if i not in used and h.key == g.key]
            if cands:
                # nearest in reading order (relative horizontal position)
                i = min(cands, key=lambda i: abs(_rel(gs[i], ps) - _rel(g, pd)))
                used.add(i)
                pairs.append((i, j))
            else:
                fade_in.append(j)
        fade_out = [i for i in range(len(gs)) if i not in used]
        z = LAYER_Z["label"] + 0.01
        helpers, items = [], []
        parent = ps["parent"]

        def glyph_obj(g, tag, color):
            cx = (g.bbox[0] + g.bbox[2]) / 2
            cy = (g.bbox[1] + g.bbox[3]) / 2
            local = g.transformed(-cx, -cy, 1.0)
            name = self._helper_name(f"{tag}")
            self.r.send("create_curve", name=name, splines=local.splines, fill=True, location=[cx, cy, z],
                        parent=parent, material=self.r.material(color), metadata={"node": "anim", "anim_helper": True})
            helpers.append(name)
            return name, (cx, cy)

        f0, f1 = self.frame(t0), self.frame(t1)
        cs = ps["color"] or self.r.theme.text
        cd = pd["color"] or self.r.theme.text
        for i, j in pairs:
            name, (x0, y0) = glyph_obj(gs[i], f"{src.name}~{dst.name}:m{i}", cs)
            x1 = (gd[j].bbox[0] + gd[j].bbox[2]) / 2
            y1 = (gd[j].bbox[1] + gd[j].bbox[3]) / 2
            items.append({"object": name, "path": "location", "interpolation": interp,
                          "keys": [[f0, [x0, y0, z]], [f1, [x1, y1, z]]]})
        for i in fade_out:
            name, _ = glyph_obj(gs[i], f"{src.name}~out{i}", cs)
            items += self._alpha([name], [(t0, 1.0), (t0 + 0.5 * (t1 - t0), 0.0)], interp)
        for j in fade_in:
            name, _ = glyph_obj(gd[j], f"{dst.name}~in{j}", cd)
            items += self._alpha([name], [(t0 + 0.5 * (t1 - t0), 0.0), (t1, 1.0)], interp)
        items += self._swap(self._objects(src.name), self._objects(dst.name), helpers, t0, t1)
        self._keys(items)
        return {"method": "glyph matching", "moved": len(pairs), "faded_out": len(fade_out), "faded_in": len(fade_in)}

    def _apply_matrix(self, targets, t0, t1, lag=0.0, interp="BEZIER", keys: int = 12, **_):
        """Animate a matrix_transform node from the identity to its matrix."""
        if len(targets) != 1 or self.c.nodes[targets[0]].kind != "matrix_transform":
            raise MathRefusal("apply_matrix needs one matrix_transform object.")
        node = self.c.nodes[targets[0]]
        fr = self.c.frame(node.spec["axes"])
        A = np.array(node.data["matrix"].tolist(), float)
        mats = [(1 - s) * np.eye(2) + s * A for s in np.linspace(0, 1, keys + 1)]
        helpers, items = [], []
        # grid lines: fixed set of source lines, each a 2-point spline per key (clipped exactly per key)
        step = node.spec.get("grid_step", 1)
        span = max(abs(fr.lo[0]), abs(fr.hi[0]), abs(fr.lo[1]), abs(fr.hi[1]))
        try:
            reach = span * max(np.linalg.norm(np.linalg.inv(M), 2) for M in mats) * 1.5
        except np.linalg.LinAlgError:
            reach = span * 3
        kmax = int(min(np.ceil(reach / step), 60))
        from .kinds_calculus import clip_line

        src_lines = [(np.array([i * step, 0.0]), np.array([0.0, 1.0])) for i in range(-kmax, kmax + 1)] + \
                    [(np.array([0.0, i * step]), np.array([1.0, 0.0])) for i in range(-kmax, kmax + 1)]
        per_key = []
        for M in mats:
            segs = []
            for P0, D in src_lines:
                seg = clip_line(M @ P0, M @ D, fr) if np.linalg.norm(M @ D) > 1e-12 else None
                if seg is None:
                    c0 = np.clip(M @ P0, fr.lo[:2], fr.hi[:2])
                    seg = np.array([c0, c0])  # collapsed (invisible)
                segs.append(seg)
            per_key.append(segs)
        grid_name = self._helper_name(f"{node.name}:grid")
        base = [geo.polyline_spline(fr.to_local(s, "grid")) for s in per_key[0]]
        self.r.send("create_curve", name=grid_name, splines=base, fill=False, bevel_depth=0.01, parent=fr.parent,
                    material=self.r.material(node.style.get("color") or self.r.theme.palette[0]),
                    metadata={"node": "anim", "anim_helper": True})
        self.r.send("shape_keys", object=grid_name, relative=False,
                    keys=[{"name": f"s{k}", "co": [p.tolist() for s in segs for p in fr.to_local(s, "grid")]}
                          for k, segs in enumerate(per_key[1:], start=1)])
        helpers.append(grid_name)
        # arrows (basis + applied vectors): meshes with exact geometry at each key
        arrows = [("i", np.zeros(2), np.array([1.0, 0.0]), self.r.theme.palette[2], 0.055),
                  ("j", np.zeros(2), np.array([0.0, 1.0]), self.r.theme.palette[3], 0.055)]
        for ref in node.spec.get("apply_to", []) or []:
            tgt = self.c.nodes[ref]
            if "vector" in tgt.data:
                arrows.append((ref, np.array([fnum(v) for v in tgt.data["tail"]]),
                               np.array([fnum(v) for v in tgt.data["head"]]), tgt.style.get("color") or "white", 0.05))
        for tag, tail, head, color, th in arrows:
            shapes = []
            for M in mats:
                A_ = fr.to_local(M @ tail, "vector")[0]
                B_ = fr.to_local(M @ head, "vector")[0]
                if np.linalg.norm(B_ - A_) < 1e-9:
                    B_ = A_ + 1e-6
                v, f = geo.arrow_mesh(A_, B_, th, style="flat")
                shapes.append((v, f))
            name = self._helper_name(f"{node.name}:{tag}")
            self.r.send("create_mesh", name=name, vertices=shapes[0][0], faces=shapes[0][1], parent=fr.parent,
                        material=self.r.material(color), metadata={"node": "anim", "anim_helper": True})
            self.r.send("shape_keys", object=name, relative=False,
                        keys=[{"name": f"s{k}", "co": v} for k, (v, _f) in enumerate(shapes[1:], start=1)])
            helpers.append(name)
        f0, f1 = self.frame(t0), self.frame(t1)
        for h in helpers:
            items.append({"object": h, "path": "shape_eval_time", "keys": [[f0, 0.0], [f1, 10.0 * keys]],
                          "interpolation": interp})
        items += self._swap([], self._objects(node.name), helpers, t0, t1)
        # keep the helpers visible from the start (they show the identity state before t0)
        items += [{"object": h, "path": "visible", "keys": [[1, 1], [f1, 0]], "include_children": False}
                  for h in helpers]
        self._keys(items)
        return {"method": f"absolute shape keys at {keys + 1} exact intermediate maps", "helpers": helpers}

    # ------------------------------------------------------------------ camera

    def _camera_move(self, targets, t0, t1, interp="BEZIER", center=None, width=None, margin=0.1, **_):
        """Pan/zoom the (orthographic 2D or perspective 3D) camera to targets or to center/width."""
        info = self.r.send("get_scene_info")
        cam = info.get("camera")
        if cam is None:
            raise MathRefusal("No camera; call setup_scene first.")
        if targets:
            b = self.r.send("bounds", names=[o for t in targets for o in self._objects(t) if not o.endswith(":halo")])
            lo = np.min([v["min"] for v in b.values()], axis=0)
            hi = np.max([v["max"] for v in b.values()], axis=0)
            center = ((lo + hi) / 2).tolist()
            size = hi - lo
            aspect = info["resolution"][0] / info["resolution"][1]
            width = max(size[0], size[1] * aspect) * (1 + 2 * margin)
        if center is None:
            raise MathRefusal("camera_move needs targets or a center.")
        f0, f1 = self.frame(t0), self.frame(t1)
        items = []
        loc0 = cam["location"]
        if cam["type"] == "ORTHO":
            loc1 = [float(center[0]), float(center[1]), loc0[2]]
            items.append({"object": cam["name"], "path": "location", "keys": [[f0, loc0], [f1, loc1]],
                          "interpolation": interp})
            if width:
                items.append({"object": cam["name"], "path": "data.ortho_scale",
                              "keys": [[f0, cam["ortho_scale"]], [f1, float(width)]], "interpolation": interp})
        else:
            c0 = np.asarray(loc0, float)
            ctr = np.asarray(center + [0] * (3 - len(center)), float)
            direction = c0 - ctr
            dist = np.linalg.norm(direction)
            if width:
                dist = float(width) / (2 * math.tan(math.radians(39.6) / 2))
            loc1 = (ctr + direction / np.linalg.norm(direction) * dist).tolist()
            items.append({"object": cam["name"], "path": "location", "keys": [[f0, loc0], [f1, loc1]],
                          "interpolation": interp})
        self._keys(items)
        return {"camera": cam["name"], "to_center": center, "to_width": width}

    def _camera_orbit(self, targets, t0, t1, interp="BEZIER", degrees: float = 90.0, center=None, **_):
        piv = self.r.send("camera_pivot", center=list(center or [0, 0, 0]))
        f0, f1 = self.frame(t0), self.frame(t1)
        self._keys([{"object": piv["pivot"], "path": "rotation_euler", "index": 2,
                     "keys": [[f0, 0.0], [f1, math.radians(degrees)]], "interpolation": interp}])
        return {"pivot": piv["pivot"], "degrees": degrees}


# --------------------------------------------------------------------------- utilities


def _num(expr):
    from .kinds_core import numeric

    return numeric(expr, [X])


def _union_samples(f, g, a, b, fr):
    t1, P1, b1 = smp.adaptive_sample(lambda t: np.stack([t, f(t)], 1), a, b, fr.scale[:2], 1e-3, fr.lo[:2], fr.hi[:2])
    t2, P2, b2 = smp.adaptive_sample(lambda t: np.stack([t, g(t)], 1), a, b, fr.scale[:2], 1e-3, fr.lo[:2], fr.hi[:2])
    ts_ = np.unique(np.concatenate([t1, t2]))
    brk = np.zeros(len(ts_) - 1, dtype=bool)
    for tt, bb in ((t1, b1), (t2, b2)):
        for i in np.nonzero(bb)[0]:
            lo_, hi_ = tt[i], tt[i + 1]
            j = np.searchsorted(ts_, lo_)
            k = np.searchsorted(ts_, hi_)
            brk[j:k] = True
    return ts_, brk


def _resample(P: np.ndarray, n: int) -> np.ndarray:
    d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    s = np.linspace(0, d[-1], n)
    return np.stack([np.interp(s, d, P[:, k]) for k in range(P.shape[1])], 1)


def _rel(g: ts.Glyph, placed: dict) -> float:
    t = placed["typeset"]
    return ((g.bbox[0] + g.bbox[2]) / 2 - placed["loc"][0] - t.bbox[0]) / max(t.width, 1e-9)

