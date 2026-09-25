---
name: blender-math
description: Build mathematically exact visuals and animations in Blender through the blender-math MCP server (tools such as define_function, add_point, add_calculus, add_geometry, add_text, animate, render_preview). Use whenever the user wants math shown in Blender — graphs of functions, tangents, areas, Riemann sums, geometry constructions, vectors and matrix transformations, vector fields, ODEs, phase portraits, splines/Bezier curves, 3D surfaces, LaTeX equations or derivations — or a math explainer video / 3Blue1Brown-style animation, even if they don't name the tools.
---

# Exact math in Blender (blender-math MCP)

The server turns math into Blender geometry **exactly**: every object is computed from its formula
(SymPy), typeset from real LaTeX, and refused if it would be wrong. Your job is to describe the math
as a **dependency graph of named objects** and to check the result visually. Never approximate by
hand, never draw math with `execute_blender_code`.

## Before you start

1. `blender_status` — confirms Blender is connected and which typesetter is active.
   - `connected: false` → tell the user to open Blender, enable the *Blender Math Bridge* add-on and
     press **Start Math MCP server** (3D View → N panel → Math MCP). Don't continue until it connects.
   - `typesetting_backend: "mathtext"` → no LaTeX installed: avoid `\begin{...}` environments
     (matrix, cases, aligned) and `\text{}`; use `\mathrm{}` instead. Suggest installing TeX Live if the
     user needs them.
2. `setup_scene(mode="2d"|"3d", theme="dark"|"light"|"chalkboard")` — once per scene. Dark is the
   default (math-video look); light suits print/textbook output. 3D scenes need `mode="3d"`.
3. If the user starts a fresh topic, `clear_scene()` and then `setup_scene(...)` again (clearing
   removes objects but keeps the camera and theme, so a 2D scene after a 3D one would otherwise render
   in perspective). Otherwise inspect with `list_objects()`: the graph is saved in the .blend file and
   may already contain objects.

## Mental model: the dependency graph

Every object has a **name** and a **definition** that can reference other objects. Changing one
object recomputes everything downstream.

```
set_parameter("a", "1/2")
define_function("f", "a*x^2 - 2", label=True)          # depends on a
add_point("A", on="f", x="1")                           # depends on f
add_calculus("tangent", name="tA", function="f", point="A", label=True)
add_text(r"f'(1) = \val{f'(A_x)}", at=[4, 3])           # live value
set_parameter("a", "2")                                 # f, A, tA and the text all update
```

Names usable inside expressions:

| Object | Use it as |
|---|---|
| parameter `a` | `a` |
| function `f` (also tangents, Taylor polynomials, splines, ODE solutions…) | `f(x)`, `f(2)`, derivatives `f'(x)`, `f''(0)` |
| point / vector `A` | `A_x`, `A_y` (`A_z`) |
| circle `c` | `c_r`, `c_x`, `c_y` |
| segment / polygon / angle / area / riemann_sum `S` | `S` (length, area, angle in radians, value) |

In `add_text` LaTeX, `\val{expr}` inserts the **exact** value (`\frac{\sqrt{2}}{2}`) and
`\approx{expr}` a decimal; both update live.

**Naming rules.** Letters/digits/underscore, starting with a letter. Reserved (refused): `x y z t
theta`, `E` `I` `e` `pi`, and SymPy function names such as `beta`, `gamma`, `re`, `im`, `sign`.
Use `alpha`, `phi`, `A`, `B`, `P1`, `f`, `g`, `h`… Omit `name` to get an automatic one.

## Coordinates and centering

- Math origin (0, 0) = Blender world origin; 1 math unit = 1 Blender unit (default axes).
  `create_axes(location=[x, y, 0], scale=...)` moves or stretches a coordinate system: world =
  location + scale · math. Use `location` to keep a small or offset range centered in the view.
- The 2D camera shows x ∈ [−8, 8], y ∈ [−4.5, 4.5]; default axes span x ∈ [−7, 7], y ∈ [−3.6, 3.6].
  Axes are created automatically on first use. Create them yourself (`create_axes`) for other ranges,
  a grid, `tick_style="pi"` for trig, or 3D (`z_range`).
- Keep the math inside the axes ranges: graphs, points and fills are **clipped** to the axes box, and
  points outside it are not drawn. Widen the axes (`update_object("axes", {"y_range": [...]})`) if
  something the user asked for is out of view.
- `add_text(..., at=[x, y])` uses world units unless `axes=` is given; `at="A"` auto-places next to
  point A. Titles go in a free corner, e.g. `at=[-4.5, 3]` (the top center is taken by the y-axis
  label).
- Layout helpers: `get_bounds([...])`, `align_object("T", "f", side="below")`,
  `frame_view(center_on="content"|"origin")`. Labels auto-avoid overlaps and re-flow after changes.

## Workflow (always)

1. Plan the objects and their dependencies (what should update when what changes).
2. Build them in dependency order (parameters → functions → points → derived objects → text).
3. `frame_view()` (2D: usually keep the default view; 3D: always frame).
4. **`render_preview()` and look at the image.** Check: labels legible and not overlapping, nothing
   cut off, colors distinguishable, the math reads correctly. Fix with `update_object` (move a text,
   change `label_dir`, `label=False`, colors) and preview again.
5. Report exact results to the user from the tool outputs (coordinates, slopes, areas) — quote the
   exact forms (`√2`, `π/4`) and say when a value is only numeric (`"exact": false`).
6. Animation (only if asked): see [references/animation.md](references/animation.md).

## Refusals are features

A tool error starting with **`REFUSED`** means the request was mathematically invalid and *nothing*
was drawn or changed. Read the reason, then fix the request or explain it to the user — never work
around it (no hand-drawn substitutes, no `execute_blender_code`). Typical cases:

| Refusal | What to do |
|---|---|
| not differentiable / vertical tangent / cusp | Tell the user no tangent exists there; offer one-sided behaviour or another x. |
| not defined / not real at x = … | Pick a point in the domain. |
| only N intersection(s); index k does not exist | `find_points(kind="intersections", of="f", with_object="g")` lists them all. |
| no root / local max … in [a, b] | Widen `x_range` or accept that none exists. |
| Step i is false … counterexample | The derivation is wrong; correct the algebra (use `verify_math`/`compute`). |
| not continuous on [a, b] | Area/Riemann sum across a pole is improper; split the interval. |
| unknown symbol(s) ['k'] | `set_parameter("k", …)` first. |
| Change rejected: X would become invalid | The update breaks a dependent object; change that object or delete it first. |
| reserved name | Rename (e.g. `beta` → `b1`). |

## Getting the math right before drawing

- `verify_math(lhs, rhs)` before displaying any claimed identity.
- `compute(...)` for derivatives, integrals, limits, series, solving — use its `latex` field in
  `add_text`, or better, reference objects with `\val{...}` so values stay live.
- `find_points(kind, of=...)` finds **all** roots / extrema / inflections / intersections in view,
  exact where possible, and creates labelled points.
- Show algebra with `add_derivation(steps=[...])` — each step is verified and shown exactly as written.
- Expression syntax: `x^2 + 2x`, `sin x`, `sqrt(1-x^2)`, `exp(-x^2/2)`, `abs(x)`, `floor(x)`,
  `cbrt(x)` (real cube root; `x^(1/3)` is undefined for x < 0), `Piecewise((x, x<0), (x^2, True))`,
  constants `pi`, `E`, `oo`. Write products of names with `*` or a space (`a*x`, not `ax`).
  Give exact inputs as strings: `"1/3"`, `"sqrt(2)"`, `"pi/4"` (not `0.333`).

## Choosing tools

| Goal | Tool |
|---|---|
| graph y = f(x) | `define_function` |
| special points | `add_point(of=..., keypoint=...)`, `find_points` |
| tangent, normal, secant, Riemann sum, area, Taylor, derivative/antiderivative graph, asymptotes | `add_calculus(kind=...)` |
| Euclidean constructions | `add_geometry(kind=...)` |
| vectors, matrices, eigenvectors | `add_linear_algebra(kind=...)` |
| vector/slope fields, ODEs, phase portraits, complex maps | `add_field(kind=...)` |
| parametric, polar, implicit curves | `plot_curve(kind=...)` |
| 3D surfaces | `plot_surface(kind="graph")` or `kind="parametric"` (after `setup_scene("3d")`) |
| interpolating splines, Bézier / de Casteljau, B-splines | `add_spline(kind=...)` |
| equations, titles, live values | `add_text`, `add_derivation` |
| change / inspect / remove | `update_object`, `get_object`, `list_objects`, `delete_object` |

Recipes with exact arguments for every topic: [references/recipes.md](references/recipes.md).

## Style

- Colors: leave `color` unset for the theme palette (distinct, readable), or `'#rrggbb'`, names like
  `math_blue`, `math_yellow`, `math_green`, `math_red`.
- `label=True` on functions shows `f(x) = …`, on points the name; `label="coords"` shows exact
  coordinates; `label="name_coords"` both; a string is custom LaTeX; `False` hides it.
- Style keys (via `style={...}` or `update_object(style=...)`): `thickness`, `dashed`, `opacity`,
  `fill_opacity`, `label_size`, `label_dir` (`n`, `ne`, `e`, … preferred label side), `visible`,
  `colormap`, `mesh_lines`, `unit` (angles: `deg`/`rad`), `show_points`, `show_polygon`.
- Keep scenes uncluttered: one idea per scene, ≤ ~6 labelled objects; use `label=False` generously.
