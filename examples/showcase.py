"""Draw a small showcase in a running Blender (no AI needed) to check the setup.

1. In Blender: enable the "Blender Math Bridge" add-on and press
   "Start Math MCP server" (3D View > Sidebar (N) > Math MCP).
2. Run:  python examples/showcase.py
"""

from blender_math_mcp.blender_client import BlenderConnection
from blender_math_mcp.studio import MathStudio

s = MathStudio(BlenderConnection())
print(s.status())

s._send("delete", all_math=True)
s.setup_scene("2d", background="#101418")
ax = s.create_axes(x_range=[-6.5, 6.5], y_range=[-3, 3], tick_style="pi", grid=True)["axes"]
s.plot_function("tan(x)", axes=ax, color="math_blue", label=r"y = \tan x")
s.plot_function("sin(x)", axes=ax, color="math_yellow")
area = s.shade_region("sin(x)", x_range=[0, "pi"], axes=ax, color="math_yellow", opacity=0.3)
s.draw_point([1.5707963267948966, 1], axes=ax, label=r"\left(\frac{\pi}{2}, 1\right)")
s.render_latex(rf"\int_0^{{\pi}} \sin x\,dx = {area['area_latex']}", location=[-4.5, 3.3, 0], size=0.55,
               color="math_yellow")
s._send("frame", margin=0.04)
print("Done - look through the camera (Numpad 0).")
