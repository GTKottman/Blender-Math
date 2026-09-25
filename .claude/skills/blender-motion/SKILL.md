---
name: blender-motion
description: Animate objects and properties in Blender through a Blender MCP server using motion math (easing curves, cubic Bézier timing, damped springs, motion paths) and verify the result by measuring velocity, acceleration and jerk. Use when asked to animate, keyframe, ease, time, smooth out, add bounce/overshoot/settle to, or move an object along a path in Blender, or to diagnose why a Blender animation looks linear, robotic, poppy or jittery.
---

# Blender motion

Good animation controls how position changes over time, not just where things
end up. Every move here is `x(t) = A + (B − A)·f(t)`. The job is picking the
timing function `f`, then checking the real curve's position, velocity,
acceleration and jerk (x, x′, x″, x‴) before calling it done.

Formulas, derivative tables, spring equations and a symptom → fix table are
in `references/math.md`. Read it when you're choosing parameters or reading
an analysis.

## Tools

This skill drives Blender through whatever Blender MCP server is connected.
Look for its tools (search for "blender"). You need:

- **code execution** (e.g. `execute_blender_code`): required, used for everything below
- **scene / object info** (e.g. `get_scene_info`, `get_object_info`): optional, inspection is also possible with code
- **viewport screenshot** (e.g. `get_viewport_screenshot`): optional, for a visual check

If no Blender MCP tools exist, say so and stop. Don't pretend to animate.

## Workflow

### 1. Inspect before touching anything

Get object names, current transforms, frame rate and frame range. Blender
keyframes are in frames, but motion quality is judged in seconds:

```python
import bpy, json
s = bpy.context.scene
print(json.dumps({"fps": s.render.fps / s.render.fps_base,
                  "range": [s.frame_start, s.frame_end],
                  "objects": {o.name: {"type": o.type, "loc": list(o.location),
                              "animated": bool(o.animation_data and o.animation_data.action)}
                              for o in s.objects}}))
```

### 2. Load the helper once per Blender session

Read `scripts/motion.py` (in this skill's directory) and send its full contents
as one code-execution call. It registers itself as the module `blender_motion`
and prints `{"blender_motion": "loaded", ...}`. Later calls only need
`import blender_motion as m`. If the import fails (Blender restarted), send
the file again.

Every function returns a dict. Always `print(json.dumps(...))` it so the
result comes back through MCP.

### 3. Pick the motion from the intent

| intent                                  | use                                                              |
|-----------------------------------------|------------------------------------------------------------------|
| move from rest to rest, general purpose | `smootherstep` (zero v and a at both ends)                       |
| same, but must stay editable as 2 keys  | `smoothstep` (exact as a Bézier) or `cubic_bezier`               |
| leave the frame / accelerate away       | `ease_in` (hard stop at end, which is fine when off-screen)      |
| arrive from off-screen / decelerate in  | `ease_out`                                                       |
| snappy UI-style arrival                 | `cubic_bezier` x1=0.16 y1=1 x2=0.3 y2=1                          |
| overshoot then settle (tweened)         | `back` or `cubic_bezier` with y2 > 1                             |
| wind-up before moving (anticipation)    | `anticipate` or `cubic_bezier` with y1 < 0                       |
| physical bounce, wobble, settle         | `spring_to` (ζ 0.3–0.5 bouncy, 0.7 subtle, 1.0 no overshoot)     |
| curved travel, arcs, fly-throughs       | `animate_along_path` (path = where, easing = when)               |
| constant speed (conveyor, rotor)        | `linear`, plus a Cycles modifier if it should loop               |

Duration drives feel as much as the curve does. Acceleration scales with
1/T², so a move that feels violent usually needs more frames, not a
different curve. Rough guide at 24 fps: small UI-scale moves take 8–15
frames, a character-scale move 18–36, a heavy object or camera 48+.

### 4. Apply

```python
import blender_motion as m, json
# rest -> rest move, 2 seconds at 24 fps
print(json.dumps(m.animate("Cube", "location", (0, 0, 0), (4, 0, 2), 1, 49, "smootherstep")))
# CSS-style timing curve, kept as two editable Bézier keys
print(json.dumps(m.animate("Cube", "rotation_euler", (0, 0, 0), (0, 0, 3.1416), 49, 73,
                           "cubic_bezier", x1=0.34, y1=1.56, x2=0.64, y2=1)))
# damped spring on scale (pass velocity=... to carry speed in from a previous move)
print(json.dumps(m.spring_to("Cube", "scale", (0.2,)*3, (1,)*3, 73,
                             frequency_hz=2.5, damping_ratio=0.4)))
# path for the arc, easing for the timing
print(json.dumps(m.animate_along_path("Ball", [(0,0,0), (2,3,1), (5,0,0)], 1, 72, "smootherstep")))
# non-object data: "Obj:data" (light energy, camera lens), "Mat@material"
print(json.dumps(m.animate("Lamp:data", "energy", 0, 1000, 1, 24, "ease_out")))
```

Rules that keep curves correct:

- `animate` replaces keys inside `[frame_start, frame_end]` on that channel
  only. Chaining moves end-to-start (A→B on 1–25, B→C on 25–49) is safe and
  keeps the neighbouring segment's handle.
- At a chained key, velocity must match unless a hard change is intended.
  Two rest-to-rest easings meet at v=0 (a hold). Mixing `linear` with
  anything leaves a kink, and `analyze` flags it under `jerk_spikes_at_frames`.
- Stagger groups of objects by offsetting `frame_start` 2–4 frames per object
  instead of moving them in lockstep. Overlap reads as organic.
- Animate one progress value (a path or a custom property driving several
  channels) rather than easing X, Y and Z separately, which bends the path.
- For rotations, animate `rotation_euler` only when the object's
  `rotation_mode` is Euler. Otherwise use the matching property.

For Blender's own Penner easings (BOUNCE, ELASTIC, BACK...) on existing keys:
`m.set_interpolation("Cube", "location", "ELASTIC", "EASE_OUT", amplitude=0.4, period=0.3)`.

### 5. Verify with numbers

Never finish on "keys inserted". Measure:

```python
print(json.dumps(m.analyze("Cube", "location", 1, 49)))                      # F-curves
print(json.dumps(m.analyze("Ball", frame_start=1, frame_end=72, space="world")))  # constraints / parents
```

Check against what was intended:

- **Rest-to-rest move**: `v_start` and `v_end` ≈ 0. For smootherstep,
  `a_start` and `a_end` are also small.
- **Ends where intended**: `end` equals the target. `overshoot` is 0 unless
  you asked for back or spring motion.
- **No kinks**: `jerk_spikes_at_frames` is empty, except at deliberate hard
  stops or impacts.
- **Plausible magnitudes**: `speed.peak` and `peak_a` suit the object's scale
  and weight. If not, change the duration first.
- **Springs**: `accel_sign_changes` roughly tracks the number of visible
  wobbles, and `settled` is true.

`m.easing_profile("smootherstep")` gives the normalized v, a and j table for
any easing, which is handy for comparing candidates before applying one.

If a viewport screenshot tool exists, set `scene.frame_current` to a few key
frames (start, 25%, peak speed, end) and capture them to confirm spacing and
arcs visually.

### 6. Report

Tell the user what moved, over which frames/seconds, which easing and why, and
the measured numbers that matter (e.g. "peak speed 3.75 m/s at frame 25,
starts and stops at rest, no overshoot"). If a check failed and you fixed it,
say what changed.

## Without the helper

If sending the helper file isn't possible, the same ideas in raw bpy:

- **smoothstep as native keys**: insert keys at both ends, set both handle
  types on both keys to `FREE`, then set `k0.handle_right = (f0 + D/3, v0)`
  and `k1.handle_left = (f0 + 2D/3, v1)`. Re-fetch keys after every
  `keyframe_points.insert()`, because insert can reallocate the array and
  stale references write nowhere.
- **Any other f**: bake one key per frame with
  `value = A + (B − A)·f((frame − f0)/(f1 − f0))`.
- **Blender 4.4+ layered actions**: F-curves live in
  `bpy_extras.anim_utils.action_get_channelbag_for_slot(action, slot).fcurves`,
  not `action.fcurves`. `obj.keyframe_insert(...)` works in every version.
