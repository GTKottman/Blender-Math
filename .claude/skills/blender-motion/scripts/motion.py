"""blender_motion: easing, springs, paths and motion analysis for Blender.

Send this whole file through the Blender MCP server's code-execution tool
(e.g. `execute_blender_code`). It registers itself as the module
`blender_motion`, so later calls only need:

    import blender_motion as m
    print(m.animate("Cube", "location", (0, 0, 0), (4, 0, 0), 1, 48, "smootherstep"))

Every public Blender function returns a JSON-serialisable dict so the
result can be printed back to the agent.

Without bpy (plain Python) the math half still works, which is how the
formulas are tested: `python3 motion.py` runs the self-checks.
"""

import json
import math

try:
    import bpy
except ImportError:  # pure-math use outside Blender
    bpy = None


# ---------------------------------------------------------------------------
# Easing functions: f(0) = 0, f(1) = 1 (except where noted), t in [0, 1]
# ---------------------------------------------------------------------------

def _clamp01(t):
    return 0.0 if t < 0.0 else 1.0 if t > 1.0 else t


def linear(t):
    return _clamp01(t)


def smoothstep(t):
    """3t^2 - 2t^3: zero velocity at both ends, acceleration jumps to +/-6."""
    t = _clamp01(t)
    return t * t * (3.0 - 2.0 * t)


def smootherstep(t):
    """6t^5 - 15t^4 + 10t^3: zero velocity AND acceleration at both ends."""
    t = _clamp01(t)
    return t * t * t * (t * (6.0 * t - 15.0) + 10.0)


def ease_in(t, power=3.0):
    return _clamp01(t) ** power


def ease_out(t, power=3.0):
    return 1.0 - (1.0 - _clamp01(t)) ** power


def ease_in_out(t, power=3.0):
    t = _clamp01(t)
    if t < 0.5:
        return 0.5 * (2.0 * t) ** power
    return 1.0 - 0.5 * (2.0 * (1.0 - t)) ** power


def back(t, overshoot=1.70158):
    """Ease-out that overshoots the target by ~10% (default) and returns."""
    u = _clamp01(t) - 1.0
    return 1.0 + u * u * ((overshoot + 1.0) * u + overshoot)


def anticipate(t, overshoot=1.70158):
    """Ease-in that first pulls back below 0 (wind-up) before moving."""
    t = _clamp01(t)
    return t * t * ((overshoot + 1.0) * t - overshoot)


def _bezier_1d(s, p1, p2):
    # Cubic Bezier with endpoints 0 and 1.
    u = 1.0 - s
    return 3.0 * u * u * s * p1 + 3.0 * u * s * s * p2 + s * s * s


def _bezier_1d_ds(s, p1, p2):
    u = 1.0 - s
    return 3.0 * u * u * p1 + 6.0 * u * s * (p2 - p1) + 3.0 * s * s * (1.0 - p2)


def cubic_bezier(t, x1=0.42, y1=0.0, x2=0.58, y2=1.0):
    """CSS-style timing curve through (0,0), (x1,y1), (x2,y2), (1,1).

    x1, x2 must be in [0, 1] (time must move forward); y1, y2 may leave
    [0, 1] to anticipate or overshoot.
    """
    t = _clamp01(t)
    if t in (0.0, 1.0):
        return t
    s = t
    for _ in range(8):  # Newton
        err = _bezier_1d(s, x1, x2) - t
        if abs(err) < 1e-9:
            return _bezier_1d(s, y1, y2)
        d = _bezier_1d_ds(s, x1, x2)
        if abs(d) < 1e-9:
            break
        s -= err / d
    lo, hi, s = 0.0, 1.0, t  # bisection fallback
    for _ in range(60):
        x = _bezier_1d(s, x1, x2)
        if abs(x - t) < 1e-9:
            break
        if x < t:
            lo = s
        else:
            hi = s
        s = 0.5 * (lo + hi)
    return _bezier_1d(s, y1, y2)


def spring(t, zeta=0.5, omega=12.0):
    """Unit step response of a damped spring, time normalised to the clip.

    omega is in radians per clip length. Does NOT land exactly on 1 at
    t=1: residual is about exp(-zeta*omega). Use spring_to() for physical
    springs measured in seconds.
    """
    return 1.0 + _spring_offset(_clamp01(t), -1.0, 0.0, omega, zeta)


EASINGS = {
    "linear": linear,
    "smoothstep": smoothstep,
    "smootherstep": smootherstep,
    "ease_in": ease_in,
    "ease_out": ease_out,
    "ease_in_out": ease_in_out,
    "back": back,
    "anticipate": anticipate,
    "cubic_bezier": cubic_bezier,
    "spring": spring,
}


def get_easing(name, **params):
    if callable(name):
        return name
    if name not in EASINGS:
        raise ValueError("unknown easing %r; choose from %s" % (name, sorted(EASINGS)))
    fn = EASINGS[name]
    return (lambda t: fn(t, **params)) if params else fn


def derivatives(f, t, h=1e-4):
    """(f, f', f'', f''') at t by central differences, stencil kept in [0,1]."""
    c = min(max(t, 2 * h), 1 - 2 * h)
    fm2, fm1, f0, fp1, fp2 = (f(c + k * h) for k in (-2, -1, 0, 1, 2))
    d1 = (fp1 - fm1) / (2 * h)
    d2 = (fp1 - 2 * f0 + fm1) / (h * h)
    d3 = (fp2 - 2 * fp1 + 2 * fm1 - fm2) / (2 * h ** 3)
    return f(t), d1, d2, d3


def easing_profile(name, samples=11, **params):
    """Table of x, v, a, j for a normalised easing plus its peaks.

    Multiply by (B-A)/T^n to get real units: v*(B-A)/T, a*(B-A)/T^2, ...
    """
    f = get_easing(name, **params)
    rows = []
    for i in range(samples):
        t = i / (samples - 1)
        x, v, a, j = derivatives(f, t)
        rows.append({"t": round(t, 3), "x": round(x, 4), "v": round(v, 3),
                     "a": round(a, 3), "j": round(j, 2)})
    dense = [derivatives(f, i / 400.0) for i in range(401)]
    return {
        "easing": name, "params": params, "rows": rows,
        "peak": {
            "v": round(max(abs(d[1]) for d in dense), 3),
            "a": round(max(abs(d[2]) for d in dense), 3),
            "j": round(max(abs(d[3]) for d in dense), 2),
        },
        "range": [round(min(d[0] for d in dense), 4), round(max(d[0] for d in dense), 4)],
    }


# ---------------------------------------------------------------------------
# Damped spring (physical units)
# ---------------------------------------------------------------------------

def _spring_offset(t, d0, v0, omega, zeta):
    """Displacement from target at time t for x'' + 2*zeta*omega*x' + omega^2*x = 0."""
    if zeta < 1.0:
        wd = omega * math.sqrt(1.0 - zeta * zeta)
        c2 = (v0 + zeta * omega * d0) / wd
        return math.exp(-zeta * omega * t) * (d0 * math.cos(wd * t) + c2 * math.sin(wd * t))
    if zeta == 1.0:
        return (d0 + (v0 + omega * d0) * t) * math.exp(-omega * t)
    r = omega * math.sqrt(zeta * zeta - 1.0)
    r1, r2 = -zeta * omega + r, -zeta * omega - r
    c2 = (v0 - r1 * d0) / (r2 - r1)
    return (d0 - c2) * math.exp(r1 * t) + c2 * math.exp(r2 * t)


def spring_metrics(frequency_hz, damping_ratio):
    """Overshoot fraction and 2% settling time for a spring released from rest."""
    omega = 2.0 * math.pi * frequency_hz
    z = damping_ratio
    overshoot = math.exp(-z * math.pi / math.sqrt(1.0 - z * z)) if z < 1.0 else 0.0
    settle = 4.0 / (z * omega) if z <= 1.0 else 4.0 / (omega * (z - math.sqrt(z * z - 1.0)))
    return {"omega": round(omega, 4), "overshoot": round(overshoot, 4),
            "settle_seconds": round(settle, 4)}


# ---------------------------------------------------------------------------
# Blender helpers
# ---------------------------------------------------------------------------

def _require_bpy():
    if bpy is None:
        raise RuntimeError("this function must run inside Blender")


def _resolve(target):
    """'Cube' -> object, 'Cube:data' -> its data (light, camera...), 'Mat@material'."""
    _require_bpy()
    if "@" in target:
        name, kind = target.split("@", 1)
        coll = {"material": bpy.data.materials, "world": bpy.data.worlds,
                "scene": bpy.data.scenes, "camera": bpy.data.cameras,
                "light": bpy.data.lights}[kind]
        return coll[name]
    name, _, sub = target.partition(":")
    obj = bpy.data.objects[name]
    return getattr(obj, sub) if sub else obj


def _fcurves(idb):
    ad = idb.animation_data
    if not ad or not ad.action:
        return []
    act = ad.action
    slot = getattr(ad, "action_slot", None)
    if slot is not None:  # Blender 4.4+ layered actions
        try:
            from bpy_extras.anim_utils import action_get_channelbag_for_slot
            bag = action_get_channelbag_for_slot(act, slot)
            if bag is not None:
                return bag.fcurves
        except ImportError:
            pass
    return getattr(act, "fcurves", [])


def _fcurve(idb, data_path, index):
    for fc in _fcurves(idb):
        if fc.data_path == data_path and fc.array_index == index:
            return fc
    return None


def _as_list(v):
    try:
        return [float(x) for x in v]
    except TypeError:
        return [float(v)]


def _ensure_fcurves(idb, data_path, frame, values):
    """keyframe_insert once per channel so the F-curve exists (handles slots)."""
    prop = idb.path_resolve(data_path)
    is_vec = not isinstance(prop, (int, float, bool))
    curves = []
    for i, val in enumerate(values):
        idx = i if is_vec else -1
        if is_vec:
            prop[i] = val
        else:
            _set_scalar(idb, data_path, val)
        idb.keyframe_insert(data_path, index=idx, frame=frame)
        curves.append(_fcurve(idb, data_path, max(idx, 0)))
    return curves


def _set_scalar(idb, data_path, val):
    if "." in data_path:
        owner_path, attr = data_path.rsplit(".", 1)
        setattr(idb.path_resolve(owner_path), attr, val)
    elif data_path.endswith("]"):
        exec("idb.%s = val" % data_path, {"idb": idb, "val": val})
    else:
        setattr(idb, data_path, val)


def _clear_range(fc, f0, f1):
    doomed = [k for k in fc.keyframe_points if f0 - 1e-6 <= k.co.x <= f1 + 1e-6]
    for k in reversed(doomed):
        fc.keyframe_points.remove(k, fast=True)


def _key_at(fc, frame):
    for k in fc.keyframe_points:
        if abs(k.co.x - frame) < 1e-4:
            return k
    return None


# Easings Blender can represent exactly with one Bezier segment
# (control points as CSS cubic-bezier x1, y1, x2, y2).
_NATIVE_BEZIER = {
    "linear": (1 / 3, 1 / 3, 2 / 3, 2 / 3),
    "smoothstep": (1 / 3, 0.0, 2 / 3, 1.0),
}


def animate(target, data_path, start, end, frame_start, frame_end,
            easing="smootherstep", step=1, mode="auto", **params):
    """Animate a property from start to end with x(t) = A + (B-A) f(t).

    mode="auto": linear, smoothstep and cubic_bezier become two editable
    Bezier keys whose handles reproduce f exactly; other easings are baked
    (one key every `step` frames). mode="bake" always bakes.
    """
    idb = _resolve(target)
    a, b = _as_list(start), _as_list(end)
    if len(a) != len(b):
        raise ValueError("start and end have different lengths")
    f0, f1 = float(frame_start), float(frame_end)
    if f1 <= f0:
        raise ValueError("frame_end must be after frame_start")

    native = None
    if mode == "auto":
        if easing == "cubic_bezier":
            native = (params.get("x1", 0.42), params.get("y1", 0.0),
                      params.get("x2", 0.58), params.get("y2", 1.0))
        elif easing in _NATIVE_BEZIER and not params:
            native = _NATIVE_BEZIER[easing]

    curves = _ensure_fcurves(idb, data_path, f0, a)
    outer = []  # handles owned by neighbouring segments, kept when chaining
    for fc in curves:
        k_in, k_out = _key_at(fc, f0), _key_at(fc, f1)
        before = any(k.co.x < f0 - 1e-4 for k in fc.keyframe_points)
        after = any(k.co.x > f1 + 1e-4 for k in fc.keyframe_points)
        outer.append((tuple(k_in.handle_left) if k_in and before else None,
                      tuple(k_out.handle_right) if k_out and after else None))
        _clear_range(fc, f0, f1)

    f = get_easing(easing, **params)
    span = f1 - f0
    for fc, va, vb, (keep_l, keep_r) in zip(curves, a, b, outer):
        dv = vb - va
        if native:
            x1, y1, x2, y2 = native
            fc.keyframe_points.insert(f0, va, options={"FAST"})
            fc.keyframe_points.insert(f1, vb, options={"FAST"})
            # Look keys up after both inserts: insert() may reallocate the
            # array and leave earlier references pointing at stale memory.
            k0, k1 = _key_at(fc, f0), _key_at(fc, f1)
            k0.interpolation = "BEZIER"
            # Both sides FREE: an auto handle on either side makes Blender
            # recompute the pair and discard the timing curve.
            for k in (k0, k1):
                k.handle_left_type = k.handle_right_type = "FREE"
            k0.handle_right = (f0 + x1 * span, va + y1 * dv)
            k1.handle_left = (f0 + x2 * span, va + y2 * dv)
            k0.handle_left = keep_l or (2 * f0 - k0.handle_right[0],
                                        2 * va - k0.handle_right[1])
            k1.handle_right = keep_r or (2 * f1 - k1.handle_left[0],
                                         2 * vb - k1.handle_left[1])
        else:
            frames = [f0 + i * step for i in range(int(span // step) + 1)]
            if frames[-1] < f1:
                frames.append(f1)
            for fr in frames:
                k = fc.keyframe_points.insert(fr, va + dv * f((fr - f0) / span),
                                              options={"FAST"})
                k.interpolation = "BEZIER"
                k.handle_left_type = k.handle_right_type = "AUTO_CLAMPED"
        fc.update()
    return {"target": target, "data_path": data_path, "easing": easing,
            "params": params, "frames": [f0, f1],
            "method": "native_bezier" if native else "baked(step=%s)" % step,
            "channels": len(curves)}


def spring_to(target, data_path, start, goal, frame_start,
              frequency_hz=2.0, damping_ratio=0.4, velocity=None,
              max_seconds=10.0, tolerance=1e-3):
    """Bake a damped spring from start toward goal, with optional initial velocity.

    damping_ratio < 1 overshoots and oscillates, = 1 settles fastest
    without overshoot, > 1 creeps in. Ends when both displacement and
    per-frame motion are under tolerance * initial displacement.
    """
    idb = _resolve(target)
    fps = _fps()
    a, g = _as_list(start), _as_list(goal)
    v0 = _as_list(velocity) if velocity is not None else [0.0] * len(a)
    omega = 2.0 * math.pi * frequency_hz
    scale = max(max(abs(x - y) for x, y in zip(a, g)),
                max(abs(v) for v in v0) / omega, 1e-6)
    tol = tolerance * scale

    frames, n = [], 0
    while True:
        t = n / fps
        vals = [gi + _spring_offset(t, ai - gi, vi, omega, damping_ratio)
                for ai, gi, vi in zip(a, g, v0)]
        frames.append(vals)
        nxt = [gi + _spring_offset(t + 1 / fps, ai - gi, vi, omega, damping_ratio)
               for ai, gi, vi in zip(a, g, v0)]
        settled = all(abs(x - gi) < tol and abs(y - x) < tol
                      for x, y, gi in zip(vals, nxt, g))
        if (n > 0 and settled) or t >= max_seconds:
            break
        n += 1
    frames[-1] = list(g)

    f0 = float(frame_start)
    curves = _ensure_fcurves(idb, data_path, f0, a)
    for fc in curves:
        _clear_range(fc, f0, f0 + len(frames) - 1)
    for ci, fc in enumerate(curves):
        for i, vals in enumerate(frames):
            k = fc.keyframe_points.insert(f0 + i, vals[ci], options={"FAST"})
            k.interpolation = "BEZIER"
            k.handle_left_type = k.handle_right_type = "AUTO_CLAMPED"
        fc.update()
    info = spring_metrics(frequency_hz, damping_ratio) if damping_ratio > 0 else {}
    info.update({"target": target, "data_path": data_path,
                 "frames": [f0, f0 + len(frames) - 1],
                 "settled": frames[-1] == list(g) and n / fps < max_seconds})
    return info


def set_interpolation(target, data_path=None, interpolation="BEZIER",
                      easing="AUTO", handle_type=None, frames=None, **extra):
    """Switch existing keys to one of Blender's built-in easings.

    interpolation: CONSTANT, LINEAR, BEZIER, SINE, QUAD, CUBIC, QUART,
    QUINT, EXPO, CIRC, BACK, BOUNCE, ELASTIC. easing: AUTO, EASE_IN,
    EASE_OUT, EASE_IN_OUT. extra: back, amplitude, period.
    """
    idb = _resolve(target)
    changed = 0
    for fc in _fcurves(idb):
        if data_path and fc.data_path != data_path:
            continue
        for k in fc.keyframe_points:
            if frames and not (frames[0] <= k.co.x <= frames[1]):
                continue
            k.interpolation = interpolation
            k.easing = easing
            if handle_type:
                k.handle_left_type = k.handle_right_type = handle_type
            for name, val in extra.items():
                setattr(k, name, val)
            changed += 1
        fc.update()
    return {"target": target, "keys_changed": changed}


def make_path(name, points, handles="AUTO", cyclic=False):
    """Create a Bezier curve object through `points` (the spatial path)."""
    _require_bpy()
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.use_path = True
    spline = curve.splines.new("BEZIER")
    spline.bezier_points.add(len(points) - 1)
    for bp, p in zip(spline.bezier_points, points):
        bp.co = p
        bp.handle_left_type = bp.handle_right_type = handles
    spline.use_cyclic_u = cyclic
    obj = bpy.data.objects.new(name, curve)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def animate_along_path(target, points, frame_start, frame_end,
                       easing="smootherstep", follow=True, path_name=None,
                       step=1, **params):
    """Separate WHERE (a Bezier path) from WHEN (an easing on progress).

    Creates a curve through `points`, adds a Follow Path constraint with a
    fixed location, and eases offset_factor 0 -> 1. Travel is by distance
    along the path, so speed on screen follows f'(t) directly.
    """
    obj = _resolve(target)
    path = make_path(path_name or (obj.name + "_path"), points)
    obj.location = (0.0, 0.0, 0.0)  # the constraint adds the path position
    con = obj.constraints.new("FOLLOW_PATH")
    con.target = path
    con.use_fixed_location = True
    con.use_curve_follow = follow
    if follow:
        con.forward_axis = "FORWARD_Y"
        con.up_axis = "UP_Z"
    dp = 'constraints["%s"].offset_factor' % con.name
    result = animate(target, dp, 0.0, 1.0, frame_start, frame_end,
                     easing=easing, step=step, **params)
    result.update({"path": path.name, "constraint": con.name})
    return result


def _fps():
    r = bpy.context.scene.render
    return r.fps / r.fps_base


def analyze(target, data_path="location", frame_start=None, frame_end=None,
            space="fcurve", substeps=1):
    """Measure position, velocity, acceleration and jerk of a channel.

    space="fcurve" samples the F-curves (fast; substeps > 1 adds sub-frame
    samples, useful for 2-key Bezier moves but noisy on baked per-frame keys).
    space="world" steps the scene and reads matrix_world (includes
    parents and constraints such as Follow Path; whole frames only).
    Units: per second, using the scene frame rate.
    """
    idb = _resolve(target)
    scene = bpy.context.scene
    fps = _fps()
    fs = scene.frame_start if frame_start is None else frame_start
    fe = scene.frame_end if frame_end is None else frame_end

    if space == "world":
        cur = scene.frame_current
        times, series = [], []
        for fr in range(int(fs), int(fe) + 1):
            scene.frame_set(fr)
            times.append(fr)
            series.append(list(idb.matrix_world.translation))
        scene.frame_set(cur)
        sub = 1
    else:
        curves = sorted((fc for fc in _fcurves(idb) if fc.data_path == data_path),
                        key=lambda fc: fc.array_index)
        if not curves:
            return {"error": "no F-curves for %s on %s" % (data_path, target)}
        sub = max(1, int(substeps))
        n = int(round((fe - fs) * sub))
        times = [fs + i / sub for i in range(n + 1)]
        series = [[fc.evaluate(t) for fc in curves] for t in times]
    return _kinematics(times, series, fps * sub)


def _kinematics(times, series, rate):
    """Finite-difference v, a, j per component plus speed; rate = samples/sec."""
    dt = 1.0 / rate
    n, dims = len(series), len(series[0])
    if n < 5:
        return {"error": "need at least 5 samples"}

    def diff(xs):
        return [(xs[i + 1] - xs[i]) / dt for i in range(len(xs) - 1)]

    out = {"samples": n, "frames": [times[0], times[-1]], "components": []}
    speeds = None
    for d in range(dims):
        x = [s[d] for s in series]
        v = diff(x)
        a = diff(v)
        j = diff(a)
        lo, hi = min(x[0], x[-1]), max(x[0], x[-1])
        over = max(max(x) - hi, lo - min(x), 0.0)
        peak_j = max(abs(q) for q in j)
        # A kink shows up as an isolated jerk sample far above its neighbours.
        spikes = [round(times[i + 2], 2) for i in range(2, len(j) - 2)
                  if abs(j[i]) > 0.1 * peak_j
                  and abs(j[i]) > 4 * max(abs(j[i - 2]), abs(j[i + 2]), 1e-9)]
        out["components"].append({
            "index": d,
            "start": round(x[0], 4), "end": round(x[-1], 4),
            "overshoot": round(over, 4),
            "v_start": round(v[0], 4), "v_end": round(v[-1], 4),
            "a_start": round(a[0], 4), "a_end": round(a[-1], 4),
            "peak_v": round(max(abs(q) for q in v), 4),
            "peak_a": round(max(abs(q) for q in a), 4),
            "peak_j": round(peak_j, 4),
            "accel_sign_changes": sum(1 for p, q in zip(a, a[1:])
                                      if p * q < 0 and abs(p) > 1e-6),
            "jerk_spikes_at_frames": spikes[:10],
        })
        speeds = ([vv * vv for vv in v] if speeds is None
                  else [s + vv * vv for s, vv in zip(speeds, v)])
    speeds = [math.sqrt(s) for s in speeds]
    peak_i = max(range(len(speeds)), key=lambda i: speeds[i])
    out["speed"] = {"peak": round(speeds[peak_i], 4),
                    "peak_at_frame": round(times[peak_i], 2),
                    "start": round(speeds[0], 4), "end": round(speeds[-1], 4)}
    return out


# ---------------------------------------------------------------------------
# Self-registration inside Blender, self-checks outside it
# ---------------------------------------------------------------------------

if bpy is not None:
    import sys as _sys
    import types as _types
    _mod = _types.ModuleType("blender_motion")
    _mod.__dict__.update({k: v for k, v in dict(globals()).items()
                          if not k.startswith("__")})
    _sys.modules["blender_motion"] = _mod
    print(json.dumps({"blender_motion": "loaded", "easings": sorted(EASINGS)}))
elif __name__ == "__main__":
    def close(x, y, tol=1e-3):
        assert abs(x - y) < tol, (x, y)

    for name in EASINGS:
        if name != "spring":
            close(get_easing(name)(0.0), 0.0)
            close(get_easing(name)(1.0), 1.0)
    for t in (0.1, 0.25, 0.5, 0.8):
        close(cubic_bezier(t, 1 / 3, 0.0, 2 / 3, 1.0), smoothstep(t), 1e-6)
        close(cubic_bezier(t, 1 / 3, 1 / 3, 2 / 3, 2 / 3), t, 1e-6)
    p = easing_profile("smoothstep")["peak"]
    close(p["v"], 1.5), close(p["a"], 6.0, 0.05)
    p = easing_profile("smootherstep")["peak"]
    close(p["v"], 1.875), close(p["a"], 10 / math.sqrt(3), 0.01)
    _, v, a, _ = derivatives(smootherstep, 0.0)
    close(v, 0.0), close(a, 0.0, 0.05)
    close(_spring_offset(0.0, 1.0, 0.0, 10.0, 0.3), 1.0)
    close(_spring_offset(0.0, 1.0, 0.0, 10.0, 1.0), 1.0)
    close(_spring_offset(0.0, 1.0, 0.0, 10.0, 2.0), 1.0)
    m = spring_metrics(2.0, 0.3)
    close(max(1.0 + _spring_offset(i / 2000.0, -1.0, 0.0, m["omega"], 0.3)
              for i in range(2000)) - 1.0, m["overshoot"])
    print("all checks passed")
