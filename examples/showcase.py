"""Build a small live construction in a running Blender (no AI needed) to check the setup.

1. In Blender: enable the "Blender Math Bridge" add-on and press
   "Start Math MCP server" (3D View > Sidebar (N) > Math MCP).
2. Run:  python examples/showcase.py
"""

from blender_math_mcp.blender_client import BlenderConnection
from blender_math_mcp.studio import Studio

s = Studio(BlenderConnection())
print(s.status())
s.clear()
s.setup_scene("2d", theme="dark")

# A dependency graph: change `a` later and everything below updates.
s.add("parameter", "a", {"value": "1/6"})
s.add("function", "f", {"expression": "a*x^3 - x"}, {"label": True})
print(s.find_points("extrema", "f"))  # exact: (-sqrt(2), 2*sqrt(2)/3), (sqrt(2), -2*sqrt(2)/3)
s.add("point", "A", {"on": "f", "x": "3"})
s.add("tangent", "tg", {"function": "f", "point": "A"}, {"label": True})
s.add("area", "S", {"upper": "f", "a": "0", "b": "sqrt(6)"}, {"label": True})
s.add("text", "T", {"latex": r"f'(3) = \val{f'(A_x)}", "at": [4, -3]})
s.frame_view()

# Animate it and write a video next to this script.
s.anim.configure(fps=30, reset=True)
s.anim.play("create", ["axes"], 1.0)
s.anim.play("create", ["f"], 1.5)
s.anim.play("create", ["A", "tg"], 1.0, lag=0.3)
s.anim.play("write", ["T"], 1.0)
s.c.update("a", {"value": "1/4"})  # the whole graph recomputes exactly
print(s.c.describe("tg")["equation"])
print("Done - look through the camera (Numpad 0) and press Space to play.")
