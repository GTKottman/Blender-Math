# Recipes

Every recipe is a sequence of tool calls, written as `tool(arg=value, ...)`. Each block starts from
`clear_scene()` followed by `setup_scene(...)`. `clear_scene` removes objects but keeps the camera and
theme, so a 2D scene after a 3D one would otherwise render in perspective. All recipes are tested
against the real add-on (see `scripts/check_skill_recipes.py`). Adapt names and numbers as needed.

## Contents
- Calculus: tangent line with live slope · extrema and roots · area and Riemann sums · Taylor polynomials · asymptotes
- Functions: families with a parameter · piecewise and domains · trig with π ticks
- Geometry: triangle centers · angles · lines and intersections
- Linear algebra: vectors and matrix maps · eigenvectors
- Fields and ODEs: slope field and solution · phase portrait · complex map
- Curves and surfaces: parametric, polar, implicit · 3D surface
- Splines: interpolation · Bézier with de Casteljau · B-spline and its basis
- Text: equations, derivations, layout

---

## Calculus

### Tangent line with live slope
```python
clear_scene()
setup_scene(mode="2d", theme="dark")
define_function(name="f", expression="x^3/6 - x", label=True)
add_point(name="A", on="f", x="3")
add_calculus(kind="tangent", name="tA", function="f", point="A", label=True)
add_text(latex=r"f'(3) = \val{f'(A_x)}", at=[4.5, -3])
update_object(name="A", definition={"x": "1"})
```
Moving A (`update_object`) moves the tangent and updates the text. The tangent's exact equation and slope
are in `get_object("tA")`.

### Extrema, roots, inflection points
```python
clear_scene()
setup_scene(mode="2d")
define_function(name="f", expression="x^3 - 3x", label=True)
find_points(kind="extrema", of="f")
find_points(kind="roots", of="f", label=False)
add_point(name="W", of="f", keypoint="inflection", label="name_coords")
```
`find_points` returns every point with exact coordinates (for example `(-1, 2)` and `(-sqrt(3), 0)`) and
creates labelled points (`label=False` skips the labels).

### Area between curves, Riemann sums
```python
clear_scene()
setup_scene(mode="2d")
define_function(name="f", expression="x^2/2", label=True)
define_function(name="g", expression="x + 4")
find_points(kind="intersections", of="f", with_object="g", prefix="I", label=False)
add_calculus(kind="area", name="S", upper="g", lower="f", a="I1_x", b="I2_x", label=True)
add_text(latex=r"\int_{-2}^{4} (g - f)\,dx = \val{S}", at=[-4, 3])
```
```python
clear_scene()
setup_scene(mode="2d")
define_function(name="f", expression="sin(x) + 2", label=True)
add_calculus(kind="riemann_sum", name="R", function="f", a="0", b="pi", n=6, method="midpoint")
add_text(latex=r"M_6 = \approx{R}", at=[-4, 3])
```
`get_object("R")` gives the exact sum, the exact integral and the error. Methods: left, right, midpoint,
trapezoid, upper, lower. The area is the geometric area, split where the curves cross. Its signed integral is
reported too.

### Taylor polynomials (approximation sequence)
```python
clear_scene()
setup_scene(mode="2d")
define_function(name="f", expression="sin(x)", label=True)
add_calculus(kind="taylor", name="T1", function="f", a="0", degree=1)
add_calculus(kind="taylor", name="T3", function="f", a="0", degree=3)
add_calculus(kind="taylor", name="T5", function="f", a="0", degree=5, label=True)
```

### Asymptotes
```python
clear_scene()
setup_scene(mode="2d")
define_function(name="h", expression="(x^2 + 1)/(x - 2)", label=True)
add_calculus(kind="asymptotes", function="h")
```

## Functions

### A family controlled by a parameter
```python
clear_scene()
setup_scene(mode="2d")
set_parameter(name="k", value="1", show=True)
define_function(name="f", expression="sin(k*x)", label=True)
set_parameter(name="k", value="2")
```
`show=True` displays `k = …`. For a *visual* change over time, use two functions and `animate("transform")`.
See animation.md.

### Piecewise functions and restricted domains
```python
clear_scene()
setup_scene(mode="2d")
define_function(name="p", expression="Piecewise((-x, x < 0), (x^2, True))", label=True)
define_function(name="q", expression="sqrt(4 - x^2)", domain=["-2", "2"])
```
Jumps and undefined regions are left as real gaps. No fake vertical segments are drawn.

### Trigonometry with π tick labels
```python
clear_scene()
setup_scene(mode="2d")
create_axes(x_range=["-2*pi", "2*pi"], y_range=[-2, 2], tick_style="pi", grid=True)
define_function(name="f", expression="sin(x)", label=True)
define_function(name="g", expression="cos(x)", label=True)
find_points(kind="intersections", of="f", with_object="g", prefix="X", label="coords")
```

## Geometry

### Triangle with circumcircle and incircle
```python
clear_scene()
setup_scene(mode="2d", theme="light")
add_point(name="A", coords=["-4", "-2.5"])
add_point(name="B", coords=["1", "-2.5"])
add_point(name="C", coords=["-2", "2.5"])
add_geometry(kind="polygon", name="tri", points=["A", "B", "C"])
add_geometry(kind="circle", name="cc", through=["A", "B", "C"])
add_geometry(kind="circle", name="ic", incircle=["A", "B", "C"])
add_point(name="O", center="cc", label="name_coords")
add_text(latex=r"R = \val{cc_r}", at=[4.5, 3])
```

### Angles, right angles, lengths
```python
clear_scene()
setup_scene(mode="2d")
add_point(name="A", coords=[0, 0])
add_point(name="B", coords=[4, 0])
add_point(name="C", coords=[0, 3])
add_geometry(kind="polygon", name="tri", points=["A", "B", "C"])
add_geometry(kind="angle", name="rt", points=["B", "A", "C"])
add_geometry(kind="angle", name="phi", points=["C", "B", "A"])
add_geometry(kind="segment", name="hyp", points=["B", "C"], label=True)
```
Right angles get the square mark automatically. Values that aren't nice show as `≈`.
Use `style={"unit": "rad"}` for radians.

### Lines and their intersections
```python
clear_scene()
setup_scene(mode="2d")
add_geometry(kind="line", name="l1", equation="2x + 3y = 6")
add_point(name="P", coords=[-3, -2])
add_geometry(kind="line", name="l2", perpendicular="l1", through="P")
add_point(name="F", intersection=["l1", "l2"], label="coords")
add_geometry(kind="segment", name="d", points=["P", "F"], label=True)
```

## Linear algebra

### Vectors and a matrix transformation
```python
clear_scene()
setup_scene(mode="2d")
create_axes(x_range=[-7, 7], y_range=[-3.6, 3.6], grid=True)
add_linear_algebra(kind="vector", name="v", components=["1", "2"], label=True)
add_linear_algebra(kind="matrix_transform", name="M", matrix=[["1", "1"], ["0", "1"]], apply_to=["v"], show_determinant=True)
```
`get_object("M")` gives the exact determinant and eigenvalues. Animate it with `animate(action="apply_matrix", targets=["M"])`.

### Eigenvectors
```python
clear_scene()
setup_scene(mode="2d")
add_linear_algebra(kind="eigenvectors", matrix=[["2", "1"], ["1", "2"]])
```
This is refused for matrices without real eigenvalues, such as rotations. Say so to the user.

## Fields and differential equations

### Slope field with a solution through a point
```python
clear_scene()
setup_scene(mode="2d")
add_field(kind="slope_field", rhs="x - y")
add_field(kind="ode_solution", name="y1", rhs="x - y", initial=["0", "1"], label=True)
```
The exact solution is used when one exists (`get_object` shows it). Otherwise the curve is a high-accuracy
numerical solution, and blow-ups are reported.

### Phase portrait with classified equilibria
```python
clear_scene()
setup_scene(mode="2d", theme="chalkboard")
add_field(kind="vector_field", field=["y", "-sin(x) - 0.3*y"])
add_field(kind="phase_portrait", name="pp", system=["y", "-sin(x) - 0.3*y"], initials=[[-3, 3], [3, -3], [0, 2.5]], t_range=[0, 25])
```

### Complex map w = f(z)
```python
clear_scene()
setup_scene(mode="2d")
add_field(kind="complex_map", expression="z^2", x_range=[-1.5, 1.5], y_range=[-1.5, 1.5], lines=9)
```

## Curves and surfaces

### Parametric, polar, implicit
```python
clear_scene()
setup_scene(mode="2d")
plot_curve(kind="parametric", name="lis", x="3*sin(3t)", y="2*sin(2t)", t_range=["0", "2*pi"])
plot_curve(kind="polar", name="rose", r="2*cos(3*theta)", theta_range=["0", "pi"])
plot_curve(kind="implicit", name="ell", equation="x^2/36 + y^2/9 = 1")
```

### 3D surface
```python
clear_scene()
setup_scene(mode="3d")
create_axes(name="ax3", x_range=[-3, 3], y_range=[-3, 3], z_range=[-1.5, 1.5])
plot_surface(kind="graph", name="S", expression="sin(sqrt(x^2 + y^2)*2)/(1 + x^2 + y^2)^0.3", mesh_lines=12)
plot_curve(kind="parametric", name="helix", x="2cos(t)", y="2sin(t)", z="t/6")
frame_view()
```
For a parametric surface such as a torus: `plot_surface(kind="parametric", x="(2+cos(v))cos(u)",
y="(2+cos(v))sin(u)", z="sin(v)", v_range=[0, "2*pi"])`.

## Splines

### Interpolating spline (exact piecewise cubic)
```python
clear_scene()
setup_scene(mode="2d")
add_spline(kind="interpolate", name="s", points=[[-6, 0], [-4, 2], [-2, 1], [0, 2.5], [2, 0]], type="natural")
add_point(name="Q", on="s", x="-3", label="coords")
```
`get_object("s")` gives the exact rational piecewise formula. The spline works like a function everywhere.

### Bézier curve with de Casteljau's construction
```python
clear_scene()
setup_scene(mode="2d")
add_spline(kind="bezier", name="bz", points=[[-3, -3], [-2, 2], [2, 3], [3, -2]], t="2/5")
```
Update `t` to move the construction point: `update_object(name="bz", definition={"t": "3/5"})`.

### B-spline and its basis functions
The basis plot uses its own small axes: `scale` stretches [0, 1] to 10 × 5 units and `location` puts
the axes' math origin at (−5, −2.5) so the plot is centered in the view.
```python
clear_scene()
setup_scene(mode="2d")
add_spline(kind="bspline", name="bs", points=[[-5, -3], [-4, 1], [-2, 3], [0, -1], [2, 2.5], [4, -2]], degree=3)
```
```python
clear_scene()
setup_scene(mode="2d")
create_axes(name="basis", x_range=[0, 1], y_range=[0, 1], scale=[10, 5], location=[-5, -2.5, 0])
add_spline(kind="bspline_basis", degree=3, count=6)
```

## Text

### Equations, derivations and layout
```python
clear_scene()
setup_scene(mode="2d")
define_function(name="f", expression="(x+1)^2 - 3", label=True)
add_derivation(name="D", steps=["(x+1)^2 - 3", "x^2 + 2x + 1 - 3", "x^2 + 2x - 2"], at=[-4, 2.5], size=0.5)
add_text(name="T", latex=r"\text{vertex at } (-1, -3)", at=[0, 0], size=0.5)
align_object(name="T", reference="D", side="below", gap=0.3)
```
If the typesetter is mathtext, write `\mathrm{vertex\ at}` instead of `\text{...}`.
