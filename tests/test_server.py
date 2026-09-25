import asyncio

from blender_math_mcp.server import mcp

EXPECTED = {
    "blender_status", "verify_math", "compute", "expression_to_latex", "render_latex", "render_expression",
    "render_derivation", "create_axes", "plot_function", "plot_parametric_curve", "plot_implicit_curve",
    "plot_surface", "plot_parametric_surface", "shade_region", "draw_vector", "draw_point", "draw_polyline",
    "setup_scene", "frame_view", "render_preview", "get_scene_info", "delete_objects", "transform_object",
    "set_color", "execute_blender_code",
}


def test_all_tools_registered_with_docs():
    tools = asyncio.run(mcp.list_tools())
    assert {t.name for t in tools} == EXPECTED
    assert all(t.description and len(t.description) > 30 for t in tools)


def test_math_tool_runs_without_blender():
    r = asyncio.run(mcp.call_tool("verify_math", {"lhs": "sin(2x)", "rhs": "2 sin(x) cos(x)"}))
    assert '"proved"' in r.content[0].text
