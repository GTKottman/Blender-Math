"""MCP server: exposes exact math typesetting and plotting in Blender as tools.

Run with ``blender-math-mcp`` (stdio transport).  Environment variables:

* ``BLENDER_MATH_HOST`` / ``BLENDER_MATH_PORT`` - where the Blender add-on listens
  (default ``127.0.0.1:9877``).
* ``BLENDER_MATH_BACKEND`` - ``auto`` (default), ``latex`` or ``mathtext``.
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

from . import expressions as ex
from .blender_client import BlenderConnection, BlenderError
from .studio import MathStudio
from .typeset import TypesetError

INSTRUCTIONS = """\
Draw mathematically exact content in Blender.

Principles (follow them for correct results):
* Never hand-draw math. Formulas are typeset from LaTeX into real font outlines
  (render_latex); graphs are computed from the formula itself (plot_*), with
  adaptive sampling, true gaps at discontinuities/poles and exact clipping.
* Prefer SymPy syntax tools (render_expression, compute, verify_math,
  render_derivation) when showing results: the LaTeX is generated from the
  parsed math, so what is shown is what was computed. Check claims with
  verify_math before displaying them.
* Typical 2D workflow: setup_scene(mode='2d') -> create_axes(...) ->
  plot_function(..., axes=<axes name>) -> render_latex(...) -> frame_view() ->
  render_preview() to look at the result and fix anything that overlaps.
* 3D: setup_scene(mode='3d') -> create_axes(..., z_range=[..]) -> plot_surface /
  plot_parametric_surface / plot_parametric_curve with axes=<name>.
* With axes=<name>, positions are math coordinates of those axes; otherwise
  they are Blender world units (a 2D scene shows about 16 x 9 units).
* Expression syntax: sin(x)/x, x^2 + 2x, sqrt(1-x^2), exp(-x^2), abs(x),
  floor(x), Piecewise((x, x<0), (x^2, True)); constants pi, E, I, oo.
* Colors: '#rrggbb', names (white, black, red, blue, math_blue, math_yellow,
  math_green, math_red, math_teal, math_purple, math_gold, ...) or [r,g,b] 0..1.
"""

mcp = _Server("blender-math", instructions=INSTRUCTIONS)

_studio: MathStudio | None = None


def studio() -> MathStudio:
    global _studio
    if _studio is None:
        _studio = MathStudio(BlenderConnection(), backend=os.environ.get("BLENDER_MATH_BACKEND", "auto"))
    return _studio


def _jsonable(v):
    """Convert numpy scalars/arrays (and tuples) into plain JSON types."""
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if hasattr(v, "tolist") and not isinstance(v, (str, bytes)):
        return v.tolist()
    return v


def _tool(fn):
    """Register ``fn`` as a tool; turn expected failures into clean tool errors."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return _jsonable(fn(*args, **kwargs))
        except (BlenderError, TypesetError, ex.ExpressionError, ValueError, KeyError, TypeError) as exc:
            raise ToolError(str(exc)) from exc

    return mcp.tool()(wrapper)


Vec = list[float]
Align = Literal["left", "center", "right"]
VAlign = Literal["bottom", "center", "top", "baseline"]


# ------------------------------------------------------------------------ status / math only


@_tool
def blender_status() -> dict:
    """Check the Blender connection and which LaTeX backend is active.

    backend 'latex' = full LaTeX (amsmath: matrices, aligned, cases...);
    'mathtext' = built-in TeX subset (no environments / \\text), used when no
    LaTeX installation is found.
    """
    return studio().status()


@_tool
def verify_math(lhs: str, rhs: str) -> dict:
    """Check whether lhs == rhs is an identity (symbolically, then at random points).

    Returns status 'proved', 'numerically_equal', 'not_equal' (with a
    counterexample) or 'unknown'. Use before displaying any claimed equality.
    Example: verify_math('sin(2x)', '2 sin(x) cos(x)').
    """
    return ex.verify_equal(ex.parse(lhs), ex.parse(rhs)).as_dict()


@_tool
def compute(
    operation: Literal["simplify", "expand", "factor", "cancel", "apart", "together", "trigsimp", "diff",
                       "integrate", "limit", "series", "solve", "evalf", "sum"],
    expression: str,
    variable: str | None = None,
    lower: str | None = None,
    upper: str | None = None,
    point: str | None = None,
    order: int = 1,
) -> dict:
    """Do exact symbolic math with SymPy and get LaTeX ready for render_latex.

    - diff: order = derivative order.  integrate: lower/upper for definite.
    - limit: point (e.g. '0', 'oo').  series: point, order = number of terms.
    - solve: expression may be an equation 'x^2 - 5x + 6 = 0' (real solutions).
    - sum: lower/upper bounds.
    The returned 'latex' field shows "input = result" and can be passed to render_latex.
    """
    return ex.compute(operation, expression, variable, lower, upper, point, order)


@_tool
def expression_to_latex(expression: str) -> dict:
    """Convert a SymPy-syntax expression (or 'lhs = rhs') into exact LaTeX, kept as written."""
    return {"latex": ex.display_latex(expression), "parsed": str(ex.parse_equation(expression))}


# ------------------------------------------------------------------------ typesetting


@_tool
def render_latex(
    latex: str,
    name: str = "Equation",
    location: Vec = [0, 0, 0],  # noqa: B006 - JSON schema default
    size: float = 0.8,
    color: str | Vec = "white",
    align_x: Align = "center",
    align_y: VAlign = "center",
    plane: Literal["xy", "xz", "yz"] = "xy",
    rotation_deg: Vec | None = None,
    extrude: float = 0.0,
    mode: Literal["math", "text"] = "math",
    face_camera: bool = False,
    axes: str | None = None,
) -> dict:
    """Typeset LaTeX into exact glyph outlines (filled curves) in Blender.

    latex: math content without $ delimiters, e.g. '\\int_0^1 x^2\\,dx = \\frac{1}{3}'.
      With mode='text' it is text with inline $...$ math, e.g. 'Area $= \\pi r^2$'.
    size: height of 1 em in scene units.  location is the anchor point
      (align_x/align_y choose which point of the formula's box it is).
    plane: 'xy' (read from the top view, 2D scenes), 'xz' (upright, faces -Y).
    face_camera: keep the formula turned to the scene camera (3D scenes).
    extrude: >0 makes 3D letters.  axes: place at math coordinates of that axes.
    """
    return studio().render_latex(latex, name, location, size, color, align_x, align_y, plane, rotation_deg,
                                 extrude, mode, None, face_camera, axes)


@_tool
def render_expression(
    expression: str,
    lhs: str | None = None,
    simplify: bool = False,
    name: str = "Expression",
    location: Vec = [0, 0, 0],  # noqa: B006
    size: float = 0.8,
    color: str | Vec = "white",
    align_x: Align = "center",
    align_y: VAlign = "center",
    plane: Literal["xy", "xz", "yz"] = "xy",
    face_camera: bool = False,
    axes: str | None = None,
) -> dict:
    """Render a SymPy-syntax expression; the LaTeX is generated from the parsed math.

    'x^2/(x+1)' -> \\frac{x^{2}}{x + 1}.  'a = b' renders an equation.
    lhs: optional left side in LaTeX, e.g. 'f(x)'.  simplify: simplify first.
    """
    return studio().render_expression(expression, lhs=lhs, evaluate=simplify, name=name, location=location,
                                      size=size, color=color, align_x=align_x, align_y=align_y, plane=plane,
                                      face_camera=face_camera, axes=axes)


@_tool
def render_derivation(
    steps: list[str],
    verify: bool = True,
    name: str = "Derivation",
    location: Vec = [0, 0, 0],  # noqa: B006
    size: float = 0.7,
    color: str | Vec = "white",
    align_x: Align = "center",
    align_y: VAlign = "center",
    plane: Literal["xy", "xz", "yz"] = "xy",
    face_camera: bool = False,
) -> dict:
    """Render a chain of equal expressions with aligned '=' signs, verifying each step.

    steps: SymPy-syntax expressions, e.g. ['(x+1)^2', '(x+1)(x+1)', 'x^2 + 2x + 1'].
    If any step is provably wrong, nothing is drawn and the failing step is reported.
    """
    return studio().render_derivation(steps, verify, name, location, size, color, align_x, align_y, plane,
                                      face_camera=face_camera)


# ------------------------------------------------------------------------ graphs


@_tool
def create_axes(
    name: str = "Axes",
    x_range: Vec = [-5, 5],  # noqa: B006
    y_range: Vec = [-3, 3],  # noqa: B006
    z_range: Vec | None = None,
    scale: float | Vec | None = None,
    location: Vec = [0, 0, 0],  # noqa: B006
    tick_step: float | Vec | None = None,
    tick_style: Literal["decimal", "pi"] | list[Literal["decimal", "pi"]] = "decimal",
    x_label: str | None = "x",
    y_label: str | None = "y",
    z_label: str | None = "z",
    grid: bool = False,
    color: str | Vec = "white",
    grid_color: str | Vec = "#3a3a3a",
    thickness: float = 0.025,
    label_size: float = 0.4,
    tick_labels: bool = True,
) -> dict:
    """Create a coordinate system (2D, or 3D when z_range is given) with LaTeX tick labels.

    scale: scene units per math unit (number, or per-axis list). Default: equal
      aspect ratio, sized to fit the default 2D view (circles stay circles).
    tick_step: spacing of ticks (default: 'nice' 1/2/5 steps); tick_style='pi'
      labels the x axis in multiples of pi (\\frac{\\pi}{2}, \\pi, ...), ideal for trig
      (or give one style per axis, e.g. ['pi', 'pi']).
    Returns the axes name to pass as axes=... to plot and draw tools.
    """
    return studio().create_axes(name, x_range, y_range, z_range, scale, location, tick_step, tick_style,
                                x_label, y_label, z_label, grid, color, grid_color, thickness, label_size,
                                tick_labels)


@_tool
def plot_function(
    expression: str,
    axes: str | None = None,
    variable: str = "x",
    x_range: Vec | None = None,
    y_range: Vec | None = None,
    name: str | None = None,
    color: str | Vec = "math_blue",
    thickness: float = 0.04,
    label: str | bool | None = None,
) -> dict:
    """Graph y = f(x) exactly: adaptive sampling, breaks at poles/jumps, gaps where undefined,
    exact clipping to the axes (or to y_range).

    expression: SymPy syntax in `variable`, e.g. 'sin(x)/x', 'tan(x)', '1/(x-1)', 'floor(x)'.
    x_range defaults to the axes' x range.  label=True puts 'y = <formula>' at the curve end,
    or pass your own LaTeX string.
    """
    return studio().plot_function(expression, variable, x_range, y_range, axes, name, color, thickness, label)


@_tool
def plot_parametric_curve(
    x: str,
    y: str,
    z: str | None = None,
    t_range: list[float | str] = [0, "2*pi"],  # noqa: B006
    variable: str = "t",
    axes: str | None = None,
    name: str | None = None,
    color: str | Vec = "math_yellow",
    thickness: float = 0.04,
) -> dict:
    """Draw a parametric curve (x(t), y(t)[, z(t)]), e.g. x='cos(3t)', y='sin(2t)'.

    t_range entries may be expressions like '2*pi'. Closed curves are joined exactly.
    """
    return studio().plot_parametric_curve(x, y, z, variable, t_range, axes, name, color, thickness)


@_tool
def plot_implicit_curve(
    equation: str,
    axes: str | None = None,
    x_range: Vec | None = None,
    y_range: Vec | None = None,
    name: str | None = None,
    color: str | Vec = "math_green",
    thickness: float = 0.04,
    resolution: int = 400,
) -> dict:
    """Draw the curve F(x, y) = 0, e.g. 'x^2 + y^2 = 4', 'y^2 = x^3 - x'.

    Points are snapped onto the true curve with Newton iterations; spurious sign
    changes at poles are removed.
    """
    return studio().plot_implicit_curve(equation, x_range, y_range, axes, name, color, thickness, resolution)


@_tool
def plot_surface(
    expression: str,
    axes: str | None = None,
    x_range: Vec | None = None,
    y_range: Vec | None = None,
    name: str | None = None,
    resolution: int = 120,
    color: str | Vec = "math_blue",
    colormap: str | None = "viridis",
    opacity: float = 1.0,
    mesh_lines: int = 0,
) -> dict:
    """Graph the surface z = f(x, y), e.g. 'sin(sqrt(x^2+y^2))', 'x^2 - y^2'.

    colormap colors by height (any matplotlib name; None = solid `color`).
    Undefined points and parts outside the axes' z_range are left out (never clamped).
    mesh_lines: draw N+1 exact iso-lines in each direction on the surface.
    """
    return studio().plot_surface(expression, x_range, y_range, axes, name, resolution, color, colormap, opacity,
                                 mesh_lines=mesh_lines)


@_tool
def plot_parametric_surface(
    x: str,
    y: str,
    z: str,
    u_range: list[float | str] = [0, "2*pi"],  # noqa: B006
    v_range: list[float | str] = [0, "pi"],  # noqa: B006
    axes: str | None = None,
    name: str | None = None,
    resolution: int = 96,
    color: str | Vec = "math_teal",
    colormap: str | None = None,
    opacity: float = 1.0,
) -> dict:
    """Draw a parametric surface in u, v, e.g. a torus:
    x='(2+cos(v))cos(u)', y='(2+cos(v))sin(u)', z='sin(v)', v_range=[0,'2*pi'].
    Seams of closed surfaces are welded.
    """
    return studio().plot_parametric_surface(x, y, z, u_range, v_range, axes, name, resolution, color, colormap,
                                            opacity)


@_tool
def shade_region(
    upper: str,
    x_range: list[float | str],
    lower: str = "0",
    variable: str = "x",
    axes: str | None = None,
    name: str | None = None,
    color: str | Vec = "math_blue",
    opacity: float = 0.4,
) -> dict:
    """Fill the region between y=lower(x) and y=upper(x) for x in x_range
    (e.g. the area under a curve for an integral). Returns the exact signed area."""
    return studio().shade_region(upper, lower, x_range, variable, axes, name, color, opacity)


# ------------------------------------------------------------------------ primitives


@_tool
def draw_vector(
    end: Vec,
    start: Vec = [0, 0, 0],  # noqa: B006
    axes: str | None = None,
    name: str | None = None,
    color: str | Vec = "math_yellow",
    thickness: float = 0.05,
    label: str | None = None,
) -> dict:
    """Draw an arrow from start to end; the tip lands exactly on `end`.
    label: LaTeX, e.g. '\\vec{v}'."""
    return studio().draw_vector(end, start, axes, name, color, thickness, label)


@_tool
def draw_point(
    position: Vec,
    axes: str | None = None,
    name: str | None = None,
    color: str | Vec = "white",
    radius: float = 0.08,
    label: str | None = None,
) -> dict:
    """Mark a point (disk in 2D, sphere in 3D), optionally with a LaTeX label like 'P(1, 2)'."""
    return studio().draw_point(position, axes, name, color, radius, label)


@_tool
def draw_polyline(
    points: list[Vec],
    closed: bool = False,
    filled: bool = False,
    axes: str | None = None,
    name: str | None = None,
    color: str | Vec = "white",
    thickness: float = 0.03,
    opacity: float = 1.0,
) -> dict:
    """Draw straight segments through points (triangles, polygons, line segments).
    filled=True fills a planar polygon."""
    return studio().draw_polyline(points, closed, filled, axes, name, color, thickness, opacity)


# ------------------------------------------------------------------------ scene


@_tool
def setup_scene(
    mode: Literal["2d", "3d"] = "2d",
    background: str | Vec = "#000000",
    view_width: float = 16.0,
    center: Vec = [0, 0, 0],  # noqa: B006
    resolution: list[int] = [1920, 1080],  # noqa: B006
    camera_location: Vec | None = None,
    engine: Literal["eevee", "cycles", "workbench"] = "eevee",
) -> dict:
    """Prepare camera, background and color management for exact math visuals.

    '2d': orthographic top-down camera showing view_width units across (like a board).
    '3d': perspective camera looking at center (+ sun light).
    Sets the 'Standard' view transform so colors appear exactly as specified, and
    switches open 3D viewports to the camera view with material preview.
    """
    return studio().setup_scene(mode, background, view_width, center, resolution, camera_location, engine)


@_tool
def frame_view(names: list[str] | None = None, margin: float = 0.08) -> dict:
    """Fit the camera to the given objects (default: all math objects)."""
    return studio()._send("frame", names=names, margin=margin)


@_tool
def render_preview(resolution: list[int] = [960, 540], engine: str | None = None) -> Image:  # noqa: B006
    """Render the camera view and return it as an image so you can check the result
    (overlaps, legibility, framing) and correct it."""
    res = studio()._send("render", resolution=resolution, engine=engine)
    return Image(data=base64.b64decode(res["png_base64"]), format="png")


@_tool
def get_scene_info() -> dict:
    """List math objects (with their formulas/metadata), camera and render settings."""
    return studio()._send("get_scene_info")


@_tool
def delete_objects(names: list[str] | None = None, all_math: bool = False) -> dict:
    """Delete objects by name (children included), or all math objects with all_math=True."""
    if not names and not all_math:
        raise ValueError("Give names or set all_math=True")
    return studio()._send("delete", names=names or [], all_math=all_math)


@_tool
def transform_object(name: str, location: Vec | None = None, rotation_deg: Vec | None = None,
                     scale: float | Vec | None = None) -> dict:
    """Move / rotate (degrees, XYZ Euler) / scale an object. Moving an axes object moves its graphs too."""
    import math

    rot = [math.radians(a) for a in rotation_deg] if rotation_deg is not None else None
    return studio()._send("transform", name=name, location=location, rotation=rot, scale=scale)


@_tool
def set_color(name: str, color: str | Vec, style: Literal["flat", "shaded", "glossy", "metal"] = "flat",
              opacity: float = 1.0, include_children: bool = False) -> dict:
    """Change an object's material. 'flat' = exact unlit color (best for 2D);
    'shaded'/'glossy'/'metal' react to light (3D surfaces)."""
    s = studio()
    return s._send("set_material", name=name, material=s._material(color, style=style, opacity=opacity),
                   include_children=include_children)


@_tool
def execute_blender_code(code: str) -> dict:
    """Run Python code inside Blender (bpy available). Escape hatch for anything the
    other tools don't cover. Disabled unless enabled in the add-on's panel."""
    return studio()._send("execute_code", code=code)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()


__all__: list[Any] = ["mcp", "main"]
