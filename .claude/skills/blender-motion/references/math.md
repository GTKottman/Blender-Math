# Motion math reference

All easings are written in normalized time `t ∈ [0,1]` with `f(0)=0`, `f(1)=1`.
A real move from A to B over T seconds is

    x(t) = A + (B − A)·f(t/T)

and every derivative picks up a factor of 1/T:

| quantity      | formula                       |
|---------------|-------------------------------|
| position      | `x = A + Δ·f(u)`, `u = t/T`   |
| velocity      | `v = Δ/T · f′(u)`             |
| acceleration  | `a = Δ/T² · f″(u)`            |
| jerk          | `j = Δ/T³ · f‴(u)`            |

So halving the duration doubles speed, quadruples acceleration and multiplies
jerk by eight. To keep a move under an acceleration budget `a_max`:

    T ≥ sqrt(|Δ| · peak|f″| / a_max)

## Polynomial easings

| easing        | f(t)                  | f′(t)                  | f″(t)                     | f‴(t)                  | peak v | peak a | ends |
|---------------|-----------------------|------------------------|---------------------------|------------------------|--------|--------|------|
| linear        | t                     | 1                      | 0                         | 0                      | 1      | 0 (∞ at ends) | v jumps 0→1: hard start/stop |
| smoothstep    | 3t² − 2t³             | 6t − 6t²               | 6 − 12t                   | −12                    | 1.5    | 6      | v=0, but a jumps 0→6 (jerk impulse) |
| smootherstep  | 6t⁵ − 15t⁴ + 10t³     | 30t²(t−1)²             | 60t − 180t² + 120t³       | 60 − 360t + 360t²      | 1.875  | 10/√3 ≈ 5.77 | v=0 and a=0; jerk 60 at ends |
| ease_in (pⁿ)  | tⁿ                    | n·tⁿ⁻¹                 | n(n−1)tⁿ⁻²                | …                      | n      | —      | soft start, hard stop |
| ease_out      | 1 − (1−t)ⁿ            | mirror of ease_in      |                           |                        | n      | —      | hard start, soft stop |

Rule of thumb: the smoother the ends, the higher the peak speed in the middle,
because the same distance has to be covered in less "effective" time.
smootherstep moves 25% faster at mid-point than smoothstep.

## Cubic Bézier timing curves

A timing curve is a 2-D cubic Bézier from (0,0) to (1,1) with control points
(x1,y1) and (x2,y2) — the CSS `cubic-bezier()` convention:

    X(s) = 3(1−s)²s·x1 + 3(1−s)s²·x2 + s³        (time)
    Y(s) = 3(1−s)²s·y1 + 3(1−s)s²·y2 + s³        (progress)

Solve `X(s) = t` for s (Newton, bisection fallback), then `f(t) = Y(s)`.

- `x1, x2 ∈ [0,1]` keeps time monotonic. `y` outside [0,1] makes anticipation
  (y1 < 0) or overshoot (y2 > 1).
- Start slope is `y1/x1`, end slope is `(1−y2)/(1−x2)`. Zero slope at an end = zero velocity there.
- `(1/3, 0, 2/3, 1)` is exactly smoothstep; `(1/3, 1/3, 2/3, 2/3)` is exactly linear.
- Common presets: ease `(0.25, 0.1, 0.25, 1)`, ease-in-out `(0.42, 0, 0.58, 1)`,
  snappy out `(0.16, 1, 0.3, 1)`, back-out `(0.34, 1.56, 0.64, 1)`.

**Blender mapping.** An F-curve Bézier segment between keys (f0, v0) and
(f1, v1) *is* this curve. With `D = f1 − f0`, `Δ = v1 − v0`:

    key0.handle_right = (f0 + x1·D, v0 + y1·Δ)
    key1.handle_left  = (f0 + x2·D, v0 + y2·Δ)

That's why `animate()` produces two editable keys for linear, smoothstep and
cubic_bezier. Smootherstep is degree 5 and cannot be one cubic segment, so it
is baked.

## Spatial paths vs. timing

Keep "where" and "when" separate:

- **Path**: a Bézier curve in space (`make_path`). Its shape controls arcs and direction.
- **Timing**: an easing on *progress along the path* (Follow Path
  `offset_factor` 0→1 with fixed location). Blender measures progress by
  distance along the curve, so on-screen speed is `path_length/T · f′(u)`.

Animating X, Y, Z separately with different easings bends the path. Easing
one progress value keeps the path you drew.

## Damped springs

    x″ + 2ζω·x′ + ω²·(x − target) = 0,   ω = 2π·frequency_hz

Let `d = x − target` with initial `d0`, `v0`:

- **Underdamped (ζ < 1)**, overshoots and rings. With `ωd = ω√(1−ζ²)`:
  `d(t) = e^(−ζωt) · (d0·cos ωd t + (v0 + ζω d0)/ωd · sin ωd t)`
- **Critically damped (ζ = 1)**, fastest settle with no overshoot:
  `d(t) = (d0 + (v0 + ω d0)·t) · e^(−ωt)`
- **Overdamped (ζ > 1)**, slow creep, no overshoot. `r₁,₂ = −ζω ± ω√(ζ²−1)`:
  `d(t) = c1·e^(r₁t) + c2·e^(r₂t)`, `c2 = (v0 − r₁d0)/(r₂ − r₁)`, `c1 = d0 − c2`

From rest:

- overshoot fraction = `exp(−ζπ / √(1−ζ²))`
- 2% settle time ≈ `4 / (ζω)`

| feel            | ζ        | frequency  | overshoot |
|-----------------|----------|------------|-----------|
| bouncy / toy    | 0.2–0.3  | 2–3 Hz     | 37–53%    |
| lively UI       | 0.4–0.5  | 2–4 Hz     | 16–25%    |
| subtle settle   | 0.7–0.8  | 1.5–3 Hz   | 1.5–5%    |
| heavy / precise | 1.0      | 1–2 Hz     | 0%        |

Higher frequency = stiffer and quicker. Lower ζ = more rings. An initial
velocity (`velocity=` in `spring_to`) is how you hand off motion from a
previous move without a speed discontinuity.

## Blender built-in interpolation (Penner easings)

`keyframe.interpolation` ∈ CONSTANT, LINEAR, BEZIER, SINE, QUAD, CUBIC, QUART,
QUINT, EXPO, CIRC, BACK, BOUNCE, ELASTIC. `keyframe.easing` ∈ AUTO, EASE_IN,
EASE_OUT, EASE_IN_OUT. BACK takes `back` (overshoot, default 1.70158); ELASTIC
takes `amplitude` and `period`. These apply to the segment *after* the key.
They're quick and editable, but they aren't tunable like a cubic Bézier or a
spring. Use `set_interpolation()` to apply them.

## Reading an analysis

`analyze()` reports per-component and overall speed numbers in units per second:

| symptom                                    | meaning                         | fix |
|--------------------------------------------|---------------------------------|-----|
| `v_start`/`v_end` ≠ 0 on a start/stop move | hard start/stop, "pops"         | ease that end |
| `a_start`/`a_end` large, v = 0             | jerk impulse (smoothstep-like)  | smootherstep or softer bezier |
| `jerk_spikes_at_frames` not empty          | kink between segments           | match velocity across keys; align handles |
| `overshoot` > 0 unexpectedly               | Bézier handles overshooting     | AUTO_CLAMPED handles or lower y2 |
| `accel_sign_changes` > 1 on a simple move  | wobble or ringing               | intended only for springs |
| `peak_a` too high for the object's weight  | move is too short               | lengthen T (see budget formula) |
