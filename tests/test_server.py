import asyncio

from blender_math_mcp.server import mcp

EXPECTED = {
    "blender_status", "verify_math", "compute", "expression_to_latex", "setup_scene", "frame_view", "get_bounds",
    "align_object", "render_preview", "execute_blender_code", "list_objects", "get_object", "update_object",
    "delete_object", "clear_scene", "set_parameter", "create_axes", "define_function", "add_point", "find_points",
    "add_calculus", "add_geometry", "add_linear_algebra", "add_field", "plot_curve", "plot_surface", "add_spline",
    "add_text", "add_derivation", "animate", "set_timeline", "render_video",
}


def test_all_tools_registered_with_docs():
    tools = asyncio.run(mcp.list_tools())
    assert {t.name for t in tools} == EXPECTED
    assert all(t.description and len(t.description) > 40 for t in tools)


def test_math_tool_runs_without_blender():
    r = asyncio.run(mcp.call_tool("verify_math", {"lhs": "sin(2x)", "rhs": "2 sin(x) cos(x)"}))
    assert '"proved"' in r.content[0].text
