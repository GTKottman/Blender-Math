"""MCP server: exact, graph-based math in Blender.

Run with ``blender-math-mcp`` (stdio transport).  Environment variables:

* ``BLENDER_MATH_HOST`` / ``BLENDER_MATH_PORT`` - where the Blender add-on listens (default 127.0.0.1:9877).
* ``BLENDER_MATH_BACKEND`` - ``auto`` (default), ``latex`` or ``mathtext``.
* ``BLENDER_MATH_THEME`` - ``dark`` (default), ``light`` or ``chalkboard``.
"""

from __future__ import annotations

import base64
import functools
import os
from typing import Any, Literal

try:  # MCP Python SDK 2.x
    from mcp.server.mcpserver import Image
    from mcp.server.mcpserver import MCPServer as _Server
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError:  # MCP Python SDK 1.x
    from mcp.server.fastmcp import FastMCP as _Server
    from mcp.server.fastmcp import Image
    from mcp.server.fastmcp.exceptions import ToolError

import sympy as sp

from . import expressions as ex
from .blender_client import BlenderConnection, BlenderError
from .keypoints import MathRefusal
from .studio import Studio
from .typeset import TypesetError

INSTRUCTIONS = """\
Exact mathematics in Blender, organised as a DEPENDENCY GRAPH (like GeoGebra).

Every object has a name and a definition that may refer to other objects:
  set_parameter('a', 2); define_function('f', 'a*sin(x)'); add_point('A', on='f', x='pi/4');
  add_calculus('tangent', function='f', point='A'); add_text(latex="f'(\\pi/4) = \\val{f'(A_x)}")
Changing one object (update_object / set_parameter) recomputes everything that depends on it.

Correctness rules (the server enforces them):
* Nothing mathematically wrong is drawn. Invalid requests are REFUSED with the reason
  (e.g. tangent at a corner of |x|, intersection index that does not exist, false derivation step).
  Read the error and fix the request; never work around it by hand-drawing.
* Values are exact (SymPy) whenever possible: roots like sqrt(2), angles like pi/3. Results say
  "exact": false with the method when only a certified numeric value exists.
* Curves are exact Bezier splines (tangents from the true derivative; polynomials up to degree 3 exact),
  with true gaps at poles/jumps and exact clipping at the axes.

Coordinates: the default axes put math (0,0) at the world origin with 1 math unit = 1 Blender unit;
the 2D view shows x in [-8, 8], y in [-4.5, 4.5]. Axes are created automatically if missing.
Reference names in expressions: parameters `a`, functions `f(x)`, derivatives `f'(x)`, `f''(2)`,
point coordinates `A_x`, `A_y`, scalar results (lengths, areas, angles) by object name.
In LaTeX of add_text, `\\val{expr}` inserts the exact value, `\\approx{expr}` a decimal.

Workflow: setup_scene -> create objects -> frame_view -> render_preview (LOOK at it and fix overlaps)
-> optionally animate(...) -> render_video. Themes: dark (default), light, chalkboard.
Expression syntax: sin(x)/x, x^2 + 2x, sqrt(1-x^2), exp(-x^2), abs(x), floor(x), cbrt(x),
Piecewise((x, x<0), (x^2, True)); constants pi, E, I, oo. Reserved names: x y z t theta, E, I, pi...
Colors: 'auto' (theme palette), '#rrggbb', names (math_blue, math_yellow, red, ...).
"""

mcp = _Server("blender-math", instructions=INSTRUCTIONS)

_studio: Studio | None = None


def studio() -> Studio:
    global _studio
    if _studio is None:
        _studio = Studio(BlenderConnection(), backend=os.environ.get("BLENDER_MATH_BACKEND", "auto"),
                         theme=os.environ.get("BLENDER_MATH_THEME"))
    return _studio


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, sp.Basic):
        return str(v)
    if hasattr(v, "tolist") and not isinstance(v, (str, bytes)):
        return v.tolist()
    return v


def _tool(fn):
    """Register ``fn`` as a tool; turn refusals and failures into clear tool errors."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            out = fn(*args, **kwargs)
            return out if isinstance(out, Image) else _jsonable(out)
        except MathRefusal as exc:
            raise ToolError(f"REFUSED (nothing was drawn or changed): {exc}") from exc
        except (BlenderError, TypesetError, ex.ExpressionError, ValueError, KeyError, TypeError) as exc:
            raise ToolError(str(exc)) from exc

    return mcp.tool()(wrapper)


def _style(style: dict | None, **explicit) -> dict:
    out = dict(style or {})
    out.update({k: v for k, v in explicit.items() if v is not None})
    return out


Vec = list[float]
Num = float | str
Color = str | list[float]

# =============================================================================== status & math


@_tool
def blender_status() -> dict:
    """Connection check, active typesetting backend (latex = full LaTeX; mathtext = TeX subset), theme and
    number of objects in the construction."""
    return studio().status()


@_tool
def verify_math(lhs: str, rhs: str) -> dict:
    """Is lhs == rhs an identity? Returns 'proved', 'numerically_equal', 'not_equal' (with counterexample) or
    'unknown'. Use before showing any claimed equality. Example: verify_math('sin(2x)', '2 sin(x) cos(x)')."""
    return ex.verify_equal(ex.parse(lhs), ex.parse(rhs)).as_dict()


@_tool
def compute(
    operation: Literal["simplify", "expand", "factor", "cancel", "apart", "together", "trigsimp", "diff",
                       "integrate", "limit", "series", "solve", "evalf", "sum"],
    expression: str, variable: str | None = None, lower: str | None = None, upper: str | None = None,
    point: str | None = None, order: int = 1,
) -> dict:
    """Exact symbolic computation (SymPy); returns the result and LaTeX 'input = result' for add_text.
    diff: order; integrate: lower/upper for definite; limit: point; series: point + order (terms);
    solve: 'x^2 - 5x + 6 = 0' (real solutions); sum: lower/upper."""
    return ex.compute(operation, expression, variable, lower, upper, point, order)


@_tool
def expression_to_latex(expression: str) -> dict:
    """SymPy syntax (or 'lhs = rhs') -> exact LaTeX, exactly as written (no simplification)."""
    return {"latex": ex.display_latex(expression), "parsed": str(ex.parse_equation(expression))}


# =============================================================================== scene & layout


@_tool
def setup_scene(mode: Literal["2d", "3d"] = "2d", theme: Literal["dark", "light", "chalkboard"] | None = None,
                background: Color | None = None, view_width: float = 16.0, center: Vec = [0, 0, 0],  # noqa: B006
                resolution: list[int] = [1920, 1080], camera_location: Vec | None = None,  # noqa: B006
                engine: Literal["eevee", "cycles", "workbench"] = "eevee") -> dict:
    """Camera + background + color management. '2d': orthographic top view centered on `center`
    (default: the math origin) showing `view_width` units. '3d': perspective camera looking at center.
    Changing the theme recolors existing objects. Colors are exact (Standard view transform)."""
    return studio().setup_scene(mode, theme, background, view_width, center, resolution, camera_location, engine)


@_tool
def frame_view(names: list[str] | None = None, center_on: Literal["content", "origin"] = "content",
               margin: float = 0.06) -> dict:
    """Fit the camera to objects (default: everything). center_on='origin' keeps the math origin at the
    center of the picture. Labels are re-laid out for the new view."""
    return studio().frame_view(names, center_on, margin)


@_tool
def get_bounds(names: list[str]) -> dict:
    """World-space bounding box, center and size of objects (to reason about layout and centering)."""
    return studio().bounds(names)


@_tool
def align_object(name: str, reference: str, side: Literal["below", "above", "left", "right", "center"] = "below",
                 gap: float = 0.3, align: Literal["center", "left", "right", "top", "bottom"] = "center") -> dict:
    """Place a text/derivation next to another object using bounding boxes (e.g. an equation centered
    under a graph). Updates the text's definition, so it stays in the graph."""
    return studio().align(name, reference, side, gap, align)


@_tool
def render_preview(resolution: list[int] = [960, 540], frame: int | None = None,  # noqa: B006
                   engine: str | None = None) -> Image:
    """Render the camera view (optionally at an animation frame) and return the image. ALWAYS look at it
    to check legibility, overlaps and framing before finishing."""
    res = studio().r.send("render", resolution=resolution, frame=frame, engine=engine)
    return Image(data=base64.b64decode(res["png_base64"]), format="png")


@_tool
def execute_blender_code(code: str) -> dict:
    """Run Python in Blender (bpy). Escape hatch; disabled unless enabled in the add-on panel."""
    return studio().r.send("execute_code", code=code)


# =============================================================================== the graph


@_tool
def list_objects() -> list:
    """All objects of the construction with definitions, dependencies and exact values."""
    s = studio()
    s.c.ensure_loaded()
    return s.c.listing()


@_tool
def get_object(name: str) -> dict:
    """One object: definition, style, exact values (coordinates, slope, area, ...), dependencies and
    dependents, Blender object names."""
    s = studio()
    s.c.ensure_loaded()
    if name not in s.c.nodes:
        raise MathRefusal(f"No object named {name!r}.")
    return s.c.describe(name)


@_tool
def update_object(name: str, definition: dict | None = None, style: dict | None = None) -> dict:
    """Change an object's definition and/or style; all dependents are recomputed and redrawn.
    Transactional: if the change makes any dependent invalid, nothing changes and the reason is returned.
    Pass a key with null to remove it. Example: update_object('A', {'x': '3'})."""
    return studio().c.update(name, definition, style)


@_tool
def delete_object(name: str) -> dict:
    """Delete an object and everything that depends on it (returns the deleted names)."""
    return {"deleted": studio().c.delete(name)}


@_tool
def clear_scene() -> dict:
    """Delete every math object and the whole construction graph."""
    return studio().clear()


@_tool
def set_parameter(name: str, value: Num, min: Num | None = None, max: Num | None = None,  # noqa: A002
                  show: bool = False) -> dict:
    """Create or change a parameter (exact: '1/3', 'sqrt(2)', 'pi/4' are kept exact). Everything using it
    updates. show=True displays 'name = value'."""
    s = studio()
    s.c.ensure_loaded()
    spec = {"value": str(value), "min": None if min is None else str(min), "max": None if max is None else str(max)}
    if name in s.c.nodes:
        return s.c.update(name, spec, {"show": show} if show else None)
    return s.c.add("parameter", name, spec, {"show": show})


@_tool
def create_axes(name: str = "axes", x_range: list[Num] = [-7.5, 7.5], y_range: list[Num] = [-4, 4],  # noqa: B006
                z_range: list[Num] | None = None, scale: Num | list[Num] | None = 1, location: Vec = [0, 0, 0],  # noqa: B006
                tick_step: Num | list[Num] | None = None, tick_style: str | list[str] = "decimal",
                x_label: str | None = "x", y_label: str | None = "y", z_label: str | None = "z",
                grid: bool = False, tick_labels: bool = True, arrow_tips: bool = True,
                style: dict | None = None) -> dict:
    """Coordinate system (3D when z_range is given). Its math origin sits at `location` (default: the world
    origin) and world = location + scale * math. scale=1 (default): 1 math unit = 1 Blender unit; scale=None
    fits the view with equal aspect. tick_style='pi' labels x in multiples of pi. Other objects are drawn in
    these coordinates via axes=<name> (default: the most recent axes)."""
    spec = {"x_range": x_range, "y_range": y_range, "z_range": z_range, "scale": scale, "location": location,
            "tick_step": tick_step, "tick_style": tick_style, "x_label": x_label, "y_label": y_label,
            "z_label": z_label, "grid": grid, "tick_labels": tick_labels, "arrow_tips": arrow_tips}
    for k in ("x_label", "y_label", "z_label"):
        spec[k] = spec[k] or ""  # None/empty = no axis label
    return studio().add("axes", name, spec, style)


@_tool
def define_function(name: str, expression: str, domain: list[Num] | None = None, axes: str | None = None,
                    color: Color | None = None, label: bool | str | None = None, thickness: float | None = None,
                    style: dict | None = None) -> dict:
    """Define and graph f(x) (exact Bezier curve; breaks at poles/jumps; clipped exactly at the axes).
    The expression may use parameters and other functions: 'a*sin(x)', "f'(x)", 'g(x)^2'.
    label=True shows 'f(x) = ...'. Once defined, f can be used everywhere (points, tangents, areas ...)."""
    return studio().add("function", name, {"expression": expression, "domain": domain, "axes": axes},
                        _style(style, color=color, label=label, thickness=thickness))


@_tool
def add_point(name: str | None = None, coords: list[Num] | None = None, on: str | None = None,
              x: Num | None = None, angle: Num | None = None, intersection: list[str] | None = None,
              index: int = 0, of: str | None = None,
              keypoint: Literal["root", "local_max", "local_min", "extremum", "inflection", "y_intercept"] | None = None,
              midpoint: list[str | list[Num]] | None = None, center: str | None = None, axes: str | None = None,
              label: bool | str | None = None, color: Color | None = None, style: dict | None = None) -> dict:
    """A point, defined by ONE of:
      coords=[x, y] (or [x,y,z]; may use parameters) | on='f', x=... (on a graph) | on='circle', angle=...
      intersection=['f', 'g'], index=k (functions, lines, circles; exact) | of='f', keypoint='root'|'local_max'|
      'local_min'|'inflection'|'y_intercept', index=k | midpoint=['A', 'B'] | center='c1'.
    label: True (name), 'coords' (exact coordinates), 'name_coords', custom LaTeX, or False.
    Coordinates are available to expressions as <name>_x, <name>_y."""
    spec = {"coords": coords, "on": on, "x": x, "angle": angle, "intersection": intersection, "of": of,
            "keypoint": keypoint, "midpoint": midpoint, "center": center, "axes": axes}
    if intersection or keypoint:
        spec["index"] = index
    return studio().add("point", name, spec, _style(style, label=label, color=color))


@_tool
def find_points(kind: Literal["roots", "extrema", "inflections", "intersections"], of: str,
                with_object: str | None = None, x_range: list[Num] | None = None, create: bool = True,
                prefix: str | None = None, label: bool | str = "coords") -> dict:
    """Find ALL roots / local extrema / inflection points of a function, or all intersections of two objects,
    in the visible range (or x_range): exact first (SymPy, verified by substitution), else certified
    50-digit numerics (marked exact=false). create=True adds them as labeled point objects."""
    return studio().find_points(kind, of, with_object, x_range, create, prefix, label)


@_tool
def add_calculus(kind: Literal["tangent", "normal", "secant", "riemann_sum", "area", "taylor", "derivative",
                               "antiderivative", "asymptotes"],
                 name: str | None = None, function: str | None = None, x: Num | None = None,
                 point: str | None = None, x1: Num | None = None, x2: Num | None = None,
                 a: Num | None = None, b: Num | None = None, n: int | None = None,
                 method: Literal["left", "right", "midpoint", "trapezoid", "upper", "lower"] | None = None,
                 upper: str | None = None, lower: str | None = None, degree: int | None = None,
                 order: int | None = None, axes: str | None = None, color: Color | None = None,
                 label: bool | str | None = None, style: dict | None = None) -> dict:
    """Calculus objects (all exact, all live in the graph):
      tangent/normal: function + (x | point on it). Refused where not differentiable (corner, cusp, pole).
      secant: function, x1, x2.   riemann_sum: function, a, b, n, method -> exact sum vs exact integral.
      area: upper, lower (default '0'), a, b -> exact geometric area (splits at crossings) and signed integral.
      taylor: function, a (center), degree.   derivative: function, order.
      antiderivative: function, a (F(x) = integral from a to x; refused without closed form).
      asymptotes: function -> vertical/horizontal/oblique asymptotes as dashed lines.
    Function arguments may be a defined name ('f') or an expression ('x^2')."""
    spec = {"function": function, "x": x, "point": point, "x1": x1, "x2": x2, "a": a, "b": b, "n": n,
            "method": method, "upper": upper, "lower": lower, "degree": degree, "order": order, "axes": axes}
    return studio().add(kind, name, spec, _style(style, color=color, label=label))


@_tool
def add_geometry(kind: Literal["segment", "line", "ray", "circle", "polygon", "angle"], name: str | None = None,
                 points: list[str | list[Num]] | None = None, through: Any = None, point: str | list[Num] | None = None,
                 slope: Num | None = None, equation: str | None = None, perpendicular: str | None = None,
                 parallel: str | None = None, perpendicular_bisector: list[str | list[Num]] | None = None,
                 angle_bisector: list[str | list[Num]] | None = None, center: str | list[Num] | None = None,
                 radius: Num | None = None, incircle: list[str | list[Num]] | None = None,
                 from_point: str | list[Num] | None = None, directed: bool | None = None,
                 axes: str | None = None, color: Color | None = None, label: bool | str | None = None,
                 style: dict | None = None) -> dict:
    """Exact Euclidean geometry (sympy.geometry). Points may be point names or [x, y].
      segment: points=[A,B] (label=True shows exact length) | ray: from_point, through
      line: through=[A,B] | point+slope | equation='2x + 3y = 6' | perpendicular=<line>+through=<point> |
            parallel=<line>+through=<point> | perpendicular_bisector=[A,B] | angle_bisector=[A,B,C]
      circle: center+radius | center+through=<point> | through=[A,B,C] | incircle=[A,B,C]
      polygon: points (exact area & perimeter) | angle: points=[A,B,C] (at B; right angles get a square
      mark; style unit 'deg'|'rad'; directed=True measures counter-clockwise)."""
    spec = {"points": points, "through": through, "point": point, "slope": slope, "equation": equation,
            "perpendicular": perpendicular, "parallel": parallel, "perpendicular_bisector": perpendicular_bisector,
            "angle_bisector": angle_bisector, "center": center, "radius": radius, "incircle": incircle,
            "from": from_point, "directed": directed, "axes": axes}
    return studio().add(kind, name, spec, _style(style, color=color, label=label))


@_tool
def add_linear_algebra(kind: Literal["vector", "matrix_transform", "eigenvectors"], name: str | None = None,
                       components: list[Num] | None = None, tail: str | list[Num] | None = None,
                       from_point: str | None = None, to_point: str | None = None,
                       matrix: list[list[Num]] | str | None = None, apply_to: list[str] | None = None,
                       grid: bool | None = None, show_determinant: bool | None = None, axes: str | None = None,
                       color: Color | None = None, label: bool | str | None = None, style: dict | None = None) -> dict:
    """vector: components [x, y(, z)] (+ tail) or from_point/to_point; tip lands exactly on the head.
      matrix_transform: 2x2 matrix -> transformed grid, basis vectors i-hat/j-hat, images of apply_to vectors/
        polygons; exact determinant and eigenvalues. Animate it with animate('apply_matrix', [name]).
      eigenvectors: real eigen-lines of a 2x2 matrix labeled with exact eigenvalues (refused if none are real)."""
    spec = {"components": components, "tail": tail, "from": from_point, "to": to_point, "matrix": matrix,
            "apply_to": apply_to, "grid": grid, "show_determinant": show_determinant, "axes": axes}
    return studio().add(kind, name, spec, _style(style, color=color, label=label))


@_tool
def add_field(kind: Literal["vector_field", "slope_field", "ode_solution", "phase_portrait", "complex_map"],
              name: str | None = None, field: list[str] | None = None, rhs: str | None = None,
              initial: list[Num] | None = None, system: list[str] | None = None,
              initials: list[list[Num]] | None = None, t_range: list[Num] | None = None,
              expression: str | None = None, x_range: list[Num] | None = None, y_range: list[Num] | None = None,
              density: int | None = None, lines: int | None = None, axes: str | None = None,
              color: Color | None = None, label: bool | str | None = None, style: dict | None = None) -> dict:
    """Fields and differential equations:
      vector_field: field=['P(x,y)', 'Q(x,y)'] (arrows colored by magnitude).
      slope_field: rhs='f(x,y)' for y' = f(x, y).
      ode_solution: rhs + initial=[x0, y0]: exact solution via dsolve (verified with checkodesol, restricted to
        its interval of existence) or Dormand-Prince 5(4) at rtol 1e-10; blow-ups are detected.
      phase_portrait: system=['P','Q'], initials=[[x,y],...], t_range; ALL equilibria in view are found and
        classified exactly (saddle, node, spiral, center) from the Jacobian.
      complex_map: expression in z, x_range/y_range of the z-grid, lines -> images of grid lines under w=f(z)."""
    spec = {"field": field, "rhs": rhs, "initial": initial, "system": system, "initials": initials,
            "t_range": t_range, "expression": expression, "x_range": x_range, "y_range": y_range,
            "density": density, "lines": lines, "axes": axes}
    return studio().add(kind, name, spec, _style(style, color=color, label=label))


@_tool
def plot_curve(kind: Literal["parametric", "polar", "implicit"], name: str | None = None, x: str | None = None,
               y: str | None = None, z: str | None = None, t_range: list[Num] | None = None, r: str | None = None,
               theta_range: list[Num] | None = None, equation: str | None = None, axes: str | None = None,
               color: Color | None = None, label: bool | str | None = None, thickness: float | None = None,
               style: dict | None = None) -> dict:
    """parametric: x(t), y(t)[, z(t)] over t_range (default [0, 2pi]); closed curves are joined exactly.
      polar: r(theta) over theta_range (negative r handled). implicit: equation F(x,y) = G(x,y) (points snapped
      onto the curve by Newton steps; poles excluded)."""
    spec = {"x": x, "y": y, "z": z, "t_range": t_range, "r": r, "theta_range": theta_range,
            "equation": equation, "axes": axes}
    return studio().add(kind, name, spec, _style(style, color=color, label=label, thickness=thickness))


@_tool
def plot_surface(kind: Literal["graph", "parametric"] = "graph", name: str | None = None,
                 expression: str | None = None, x: str | None = None, y: str | None = None, z: str | None = None,
                 x_range: list[Num] | None = None, y_range: list[Num] | None = None,
                 u_range: list[Num] | None = None, v_range: list[Num] | None = None, resolution: int | None = None,
                 axes: str | None = None, color: Color | None = None, colormap: str | None = None,
                 opacity: float | None = None, mesh_lines: int | None = None, style: dict | None = None) -> dict:
    """3D surfaces (3D axes are created if needed). graph: z = expression(x, y), colored by height by default;
    mesh_lines draws exact iso-lines. parametric: x(u,v), y(u,v), z(u,v) with seams welded.
    Undefined parts and parts outside the axes box are left out (never clamped)."""
    spec = {"expression": expression, "x": x, "y": y, "z": z, "x_range": x_range, "y_range": y_range,
            "u_range": u_range, "v_range": v_range, "resolution": resolution, "axes": axes}
    kind_ = "surface" if kind == "graph" else "parametric_surface"
    return studio().add(kind_, name, spec, _style(style, color=color, colormap=colormap, opacity=opacity,
                                                   mesh_lines=mesh_lines))


@_tool
def add_spline(kind: Literal["interpolate", "bezier", "bspline", "bspline_basis"], name: str | None = None,
               points: list[str | list[Num]] | None = None,
               type: Literal["natural", "clamped", "not-a-knot", "periodic", "lagrange"] | None = None,  # noqa: A002
               end_slopes: list[Num] | None = None, t: Num | None = None, degree: int | None = None,
               knots: list[Num] | None = None, count: int | None = None, axes: str | None = None,
               color: Color | None = None, label: bool | str | None = None, style: dict | None = None) -> dict:
    """Splines, drawn exactly (cubic pieces ARE Bezier segments):
      interpolate: cubic spline through points (natural | clamped with end_slopes | not-a-knot | periodic |
        lagrange); exact rational piecewise formula; usable as a function (name(x)).
      bezier: control points (any degree); t=... shows de Casteljau's construction at t (exact).
      bspline: control points, degree (3), knots (default clamped uniform); control polygon shown.
      bspline_basis: basis functions N_{i,p}(t) for degree + knots (or count) plotted on the axes."""
    kind_ = {"interpolate": "spline"}.get(kind, kind)
    spec = {"points": points, "type": type, "end_slopes": end_slopes, "t": t, "degree": degree, "knots": knots,
            "count": count, "axes": axes}
    return studio().add(kind_, name, spec, _style(style, color=color, label=label))


@_tool
def add_text(latex: str, name: str | None = None, at: list[Num] | str = [0, 3.6],  # noqa: B006
             size: float = 0.7, color: Color | None = None, mode: Literal["math", "text"] = "math",
             align_x: Literal["left", "center", "right"] = "center",
             align_y: Literal["bottom", "center", "top", "baseline"] = "center", axes: str | None = None,
             plane: Literal["xy", "xz", "yz"] = "xy", face_camera: bool = False, style: dict | None = None) -> dict:
    """Typeset LaTeX (real LaTeX glyph outlines). at=[x, y] in world units (or axes coords with axes=...) or a
    point name (auto-placed next to it, avoiding overlaps). Live values: '\\val{expr}' is replaced by the exact
    value and '\\approx{expr}' by a decimal, and the text updates when the graph changes, e.g.
    "f'(2) = \\val{f'(2)}", "A = \\val{S}". mode='text' allows words with inline $math$."""
    return studio().add("text", name, {"latex": latex, "at": at, "mode": mode, "axes": axes},
                        _style(style, size=size, color=color, align_x=align_x, align_y=align_y, plane=plane,
                               face_camera=face_camera or None))


@_tool
def add_derivation(steps: list[str], name: str | None = None, at: list[Num] = [0, 0], size: float = 0.6,  # noqa: B006
                   color: Color | None = None, align_x: Literal["left", "center", "right"] = "center") -> dict:
    """Chain lhs = step1 = step2 = ... with aligned '=' signs, each shown as written. Every step is verified;
    a false step is REFUSED with a counterexample and nothing is drawn."""
    return studio().add("derivation", name, {"steps": steps, "at": at},
                        _style(None, size=size, color=color, align_x=align_x))


# =============================================================================== animation


@_tool
def animate(action: Literal["create", "write", "fade_in", "fade_out", "grow", "indicate", "transform",
                            "apply_matrix", "camera_move", "camera_orbit", "wait"],
            targets: list[str] | None = None, duration: float = 1.0, start: float | None = None, lag: float = 0.0,
            easing: Literal["smooth", "linear"] = "smooth", advance: bool = True,
            center: list[float] | None = None, width: float | None = None, degrees: float | None = None) -> dict:
    """Add an animation at the timeline cursor (seconds), then advance it (advance=False to overlap the next).
      create: curves draw on, fills/meshes fade in, labels follow | write: handwriting for equations/text |
      fade_in/fade_out | grow (scale from its origin) | indicate (pulse) |
      transform [source, target]: functions on the same axes morph by exact pointwise blend (1-s)f+s g;
        equations morph glyph-by-glyph (matching symbols move); other curves by arclength |
      apply_matrix [matrix_transform]: plane morphs from identity to the matrix (exact intermediate maps) |
      camera_move: to targets (framed) or center/width | camera_orbit: degrees around center (3D) | wait.
    lag staggers multiple targets. Objects animated with create/write/fade_in are hidden until their start."""
    opts = {k: v for k, v in {"center": center, "width": width, "degrees": degrees}.items() if v is not None}
    return studio().anim.play(action, targets, duration, start, lag, easing, advance, **opts)


@_tool
def set_timeline(fps: int | None = None, reset: bool = False, cursor: float | None = None) -> dict:
    """Timeline settings. reset=True removes all animation (objects become static and visible). cursor sets
    the current time (seconds) for the next animate call."""
    s = studio()
    out = s.anim.configure(fps, reset)
    if cursor is not None:
        s.anim.cursor = float(cursor)
        out = s.anim.state()
    return out


@_tool
def render_video(filepath: str, resolution: list[int] = [1920, 1080], format: Literal["mp4", "png"] = "mp4",  # noqa: B006, A002
                 engine: str | None = None, samples: int | None = None) -> dict:
    """Render the animation (whole timeline) to an MP4 (H.264) file or a PNG sequence folder. Can take a while;
    check single frames first with render_preview(frame=...)."""
    s = studio()
    old = s.r.t.timeout if hasattr(s.r.t, "timeout") else None
    if old is not None:
        s.r.t.timeout = 3600
    try:
        return s.r.send("render_animation", filepath=filepath, resolution=resolution, format=format, engine=engine,
                        samples=samples, frame_start=1, frame_end=s.anim.frame(s.anim.end + 0.5))
    finally:
        if old is not None:
            s.r.t.timeout = old


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
