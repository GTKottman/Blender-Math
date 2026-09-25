# Animation

Animations are added at a **timeline cursor** (seconds). Each `animate` call starts at the cursor and
then moves the cursor forward, so you write the video as a script, top to bottom.

## Rules

1. **Build the whole scene first, then animate.** Objects exist from frame 1 unless an animation
   reveals them. `create`, `write`, `fade_in` and `grow` hide their target until the start of the
   animation.
2. Start with `set_timeline(fps=30, reset=True)`. `reset` removes old animation and shows every object
   statically again.
3. Pacing: 0.8–1.5 s per element, 1.5–2.5 s for morphs, and `wait` 0.5–1 s between ideas so viewers
   can read. Each text needs roughly 1 s plus 0.3 s per term.
4. Overlapping moves: `advance=False` makes the next call start at the same time. `lag=0.2`
   staggers several targets inside one call.
5. **Check before rendering the video.** Call `render_preview(frame=N)` at several frames (after each
   reveal, in the middle of morphs, at the end). Frame = 1 + seconds × fps.
6. `render_video(filepath=".../name.mp4")` renders the whole timeline. Use a small `resolution`
   (e.g. `[960, 540]`) for drafts. It can take minutes, so tell the user.
7. Changing the graph after animating is fine. Animations are replayed automatically on the redrawn
   objects, and `get_object` / `set_timeline` show the state.

## Actions

| action | targets | effect |
|---|---|---|
| `create` | any objects | Curves draw on, fills and meshes fade in, labels follow. |
| `write` | text, derivations, labels | Handwriting: outlines trace, then the fill appears. |
| `fade_in` / `fade_out` | any | Opacity animation. |
| `grow` | vectors, points | Scales up from the object's origin (a vector grows from its tail). |
| `indicate` | any | Short pulse to draw attention. |
| `transform` | `[source, target]` | Morph. Two functions on the same axes: the exact pointwise blend (1−s)·f + s·g. Text to text: matching symbols move, the rest fade. Other curves: arclength morph. Otherwise a cross-fade. |
| `apply_matrix` | `[matrix_transform]` | The plane deforms from the identity to the matrix; vectors ride along exactly. |
| `camera_move` | objects, or `center`/`width` | Pan and zoom to frame targets, or to a center and view width. |
| `camera_orbit` | none (`degrees`, `center`) | Orbits a 3D scene. |
| `wait` | none | Pause. |

`transform` hides the source and shows the target at the end. Create the target object before the
transform.

## Patterns

### Graph reveal with tangent and value
```python
clear_scene()
setup_scene(mode="2d")
define_function(name="f", expression="x^3/6 - x", label=True)
add_point(name="A", on="f", x="3")
add_calculus(kind="tangent", name="tA", function="f", point="A", label=True)
add_text(name="T", latex=r"f'(3) = \val{f'(A_x)}", at=[4.5, -3])
set_timeline(fps=30, reset=True)
animate(action="create", targets=["axes"], duration=1.0)
animate(action="create", targets=["f"], duration=1.5)
animate(action="grow", targets=["A"], duration=0.5)
animate(action="create", targets=["tA"], duration=1.0)
animate(action="write", targets=["T"], duration=1.2)
animate(action="indicate", targets=["T"], duration=0.8)
```

### "Change a parameter" morph (show a family)
The graph itself cannot be animated through a parameter sweep. Instead, define the start and end
states as separate functions and morph between them:
```python
clear_scene()
setup_scene(mode="2d")
define_function(name="f", expression="sin(x)", label=True)
define_function(name="g", expression="2*sin(x/2)", label=True)
set_timeline(fps=30, reset=True)
animate(action="create", targets=["axes", "f"], duration=1.5, lag=0.3)
animate(action="wait", duration=0.5)
animate(action="transform", targets=["f", "g"], duration=2.0)
```

### Equation morph
```python
clear_scene()
setup_scene(mode="2d")
add_text(name="e1", latex="(a+b)^2", at=[0, 1], size=1.0)
add_text(name="e2", latex="a^2 + 2ab + b^2", at=[0, 1], size=1.0)
set_timeline(fps=30, reset=True)
animate(action="write", targets=["e1"], duration=1.0)
animate(action="wait", duration=0.5)
animate(action="transform", targets=["e1", "e2"], duration=1.5)
```
Put both texts at the same `at` so they morph in place. Only e1 is visible before the transform.

### Linear map
```python
clear_scene()
setup_scene(mode="2d")
create_axes(x_range=[-7, 7], y_range=[-3.6, 3.6])
add_linear_algebra(kind="vector", name="v", components=["1", "2"])
add_linear_algebra(kind="matrix_transform", name="M", matrix=[["0", "-1"], ["1", "0"]], apply_to=["v"])
set_timeline(fps=30, reset=True)
animate(action="apply_matrix", targets=["M"], duration=2.5)
animate(action="camera_move", targets=["v"], duration=1.0)
```

### 3D orbit
```python
clear_scene()
setup_scene(mode="3d")
plot_surface(kind="graph", name="S", expression="x^2 - y^2")
frame_view()
set_timeline(fps=30, reset=True)
animate(action="fade_in", targets=["S"], duration=1.0)
animate(action="camera_orbit", degrees=90, center=[0, 0, 0], duration=4.0)
```
