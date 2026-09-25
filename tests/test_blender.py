"""End-to-end tests of the construction graph with the real Blender add-on (needs ``bpy``)."""

import json
import math
import socket
import threading
import time

import numpy as np
import pytest

from blender_math_mcp.blender_client import BlenderConnection, BlenderError
from blender_math_mcp.keypoints import MathRefusal


def objects():
    import bpy

    return bpy.data.objects


# ----------------------------------------------------------------------------- graph semantics


def test_default_axes_put_math_origin_at_world_origin(studio):
    studio.add("function", "f", {"expression": "x^2"})
    ax = objects()["axes"]
    assert tuple(ax.location) == (0, 0, 0)
    assert studio.c.describe("axes")["scale"] == [1.0, 1.0, 1.0]


def test_dependents_update_when_parameter_changes(studio):
    studio.add("parameter", "a", {"value": "1"})
    studio.add("function", "f", {"expression": "a*x^2"})
    studio.add("point", "A", {"on": "f", "x": "2"})
    studio.add("tangent", "tg", {"function": "f", "point": "A"})
    assert studio.c.describe("tg")["equation"] == "y = 4 x - 4"
    out = studio.c.update("a", {"value": "1/2"})
    assert out["recomputed"] == ["f", "A", "tg"]
    assert studio.c.describe("A")["coordinates"] == ["2", "2"]
    assert studio.c.describe("tg")["equation"] == "y = 2 x - 2"


def test_update_is_transactional(studio):
    studio.add("function", "f", {"expression": "x^2 - 1"})
    studio.add("point", "R", {"of": "f", "keypoint": "root", "index": 1})
    with pytest.raises(MathRefusal, match="nothing was modified"):
        studio.c.update("f", {"expression": "x^2 + 1"})  # R would no longer exist
    assert studio.c.nodes["f"].spec["expression"] == "x^2 - 1"
    assert studio.c.describe("R")["coordinates"] == ["1", "0"]


@pytest.mark.parametrize("kind, spec, match", [
    ("tangent", {"function": "abs(x)", "x": "0"}, "not differentiable"),
    ("function", {"expression": "k*x"}, "unknown symbol"),
    ("point", {"on": "f", "x": "0"}, "not defined"),
    ("circle", {"through": [[0, 0], [1, 1], [2, 2]]}, "collinear"),
    ("eigenvectors", {"matrix": [["0", "-1"], ["1", "0"]]}, "no real eigenvalues"),
    ("derivation", {"steps": ["(a+b)^2", "a^2+b^2"]}, "false"),
    ("riemann_sum", {"function": "1/x", "a": "-1", "b": "1", "n": 4}, "not continuous"),
])
def test_refusals_draw_nothing(studio, kind, spec, match):
    studio.add("function", "f", {"expression": "1/x"})
    before = set(o.name for o in objects())
    with pytest.raises(MathRefusal, match=match):
        studio.add(kind, None, spec)
    assert set(o.name for o in objects()) == before


def test_self_reference_and_cycles_refused(studio):
    with pytest.raises(MathRefusal, match="itself"):
        studio.add("function", "g", {"expression": "g(x) + 1"})
    studio.add("parameter", "a", {"value": "1"})
    studio.add("parameter", "b", {"value": "a + 1"})
    with pytest.raises(MathRefusal, match="circular"):
        studio.c.update("a", {"value": "b"})


def test_delete_cascades(studio):
    studio.add("function", "f", {"expression": "x^2"})
    studio.add("point", "A", {"on": "f", "x": "1"})
    assert studio.c.delete("f") == ["f", "A"]
    assert "A" not in objects() and "f" not in objects()


def test_graph_persists_in_blend(studio, bpy_addon):
    from blender_math_mcp.studio import Studio
    from conftest import InProcessTransport

    studio.add("parameter", "a", {"value": "3"})
    studio.add("function", "f", {"expression": "a*x"})
    fresh = Studio(InProcessTransport(bpy_addon))
    fresh.c.ensure_loaded()
    assert set(fresh.c.nodes) >= {"a", "f", "axes"}
    assert fresh.c.describe("f")["expression"] == "3*x"


# ----------------------------------------------------------------------------- exact drawing


def _bezier_points(obj):
    pts = []
    for s in obj.data.splines:
        for bp in s.bezier_points:
            pts.append((tuple(bp.co), tuple(bp.handle_left), tuple(bp.handle_right)))
    return pts


def test_graph_is_exact_bezier_on_the_curve(studio):
    studio.add("function", "f", {"expression": "sin(x)"})
    obj = objects()["f"]
    assert obj.data.splines[0].type == "BEZIER"
    for co, _, _ in _bezier_points(obj):
        # every Bezier point (knots and exact subdivision points) is within 2% of the line width
        assert abs(math.sin(co[0]) - co[1]) < 0.02 * 0.04 + 1e-6
        assert co[2] == pytest.approx(0.02)  # 2D stacking layer


def test_cubic_polynomial_drawn_exactly(studio):
    studio.add("function", "f", {"expression": "x^3/8 - x"})
    obj = objects()["f"]
    s = obj.data.splines[0]
    pts = s.bezier_points
    for i in range(len(pts) - 1):
        p0, p1, p2, p3 = (np.array(v)[:2] for v in (pts[i].co, pts[i].handle_right, pts[i + 1].handle_left,
                                                     pts[i + 1].co))
        for u in np.linspace(0, 1, 9):
            b = (1 - u) ** 3 * p0 + 3 * (1 - u) ** 2 * u * p1 + 3 * (1 - u) * u ** 2 * p2 + u ** 3 * p3
            assert abs(b[1] - (b[0] ** 3 / 8 - b[0])) < 1e-5


def test_tan_is_broken_at_poles(studio):
    studio.add("function", "f", {"expression": "tan(x)"})
    assert len(objects()["f"].data.splines) == 5  # [-7, 7] contains 4 poles of tan


def test_intersection_point_exact(studio):
    studio.add("function", "f", {"expression": "x^2"})
    studio.add("line", "l", {"equation": "y = x + 2"})
    studio.add("point", "P", {"intersection": ["f", "l"], "index": 1})
    assert studio.c.describe("P")["coordinates"] == ["2", "4"]
    studio.add("circle", "c", {"center": [0, 0], "radius": "2"})
    studio.add("point", "Q", {"intersection": ["c", "l"], "index": 0})
    assert studio.c.describe("Q")["coordinates"] == ["-2", "0"]


def test_area_exact(studio):
    studio.add("function", "f", {"expression": "x^3 - x"})
    studio.add("area", "S", {"upper": "f", "a": "-1", "b": "1"})
    d = studio.c.describe("S")
    assert d["signed_integral"] == "0" and d["area"] == "1/2"


def test_riemann_sum_exact(studio):
    studio.add("riemann_sum", "R", {"function": "x^2", "a": "0", "b": "1", "n": 4, "method": "right"})
    d = studio.c.describe("R")
    assert d["sum"] == "15/32" and d["integral"] == "1/3"


def test_live_text_values(studio):
    studio.add("parameter", "a", {"value": "2"})
    studio.add("function", "f", {"expression": "a*x^2"})
    studio.add("text", "T", {"latex": r"f'(1) = \val{f'(1)}", "at": [0, 3]})
    assert studio.c.nodes["T"].data["latex"] == "f'(1) = 4"
    studio.c.update("a", {"value": "5"})
    assert studio.c.nodes["T"].data["latex"] == "f'(1) = 10"


def test_labels_do_not_overlap(studio):
    for i, x in enumerate(["-1", "-0.9", "-0.8", "0.8", "0.9", "1"]):
        studio.add("point", f"P{i}", {"coords": [x, "0"]})
    rects = studio.r._label_rects()
    for i, a in enumerate(rects):
        for b in rects[i + 1:]:
            overlap = min(a[2], b[2]) - max(a[0], b[0]) > 1e-6 and min(a[3], b[3]) - max(a[1], b[1]) > 1e-6
            assert not overlap


def test_phase_portrait_equilibria(studio):
    studio.add("phase_portrait", "pp", {"system": ["y", "-sin(x) - y/3"], "initials": [[1, 0]]})
    eq = studio.c.describe("pp")["equilibria"]
    assert [e["type"] for e in eq] == ["stable spiral", "saddle", "stable spiral", "saddle", "stable spiral"]


def test_matrix_and_eigen(studio):
    studio.add("matrix_transform", "M", {"matrix": [["2", "1"], ["1", "2"]]})
    d = studio.c.describe("M")
    assert d["determinant"] == "3"
    assert sorted(e["eigenvalue"] for e in d["eigen"]) == ["1", "3"]


def test_ode_exact_solution(studio):
    studio.add("ode_solution", "y1", {"rhs": "y", "initial": ["0", "1"]})
    assert studio.c.describe("y1")["solution"] == "exp(x)"


def test_surface_3d(studio):
    studio.add("surface", "S", {"expression": "x^2 - y^2", "resolution": 10})
    me = objects()["S"].data
    assert len(me.polygons) > 0 and "Col" in me.color_attributes
    assert studio.c.describe(studio.c.nodes["S"].spec["axes"])["dims"] == 3  # 3D axes created automatically


def test_align_text_below_graph(studio):
    studio.add("function", "f", {"expression": "sin(x)"})
    studio.add("text", "T", {"latex": r"y = \sin x", "at": [0, 0]})
    studio.align("T", "f", side="below", gap=0.3)
    b = studio.bounds(["T", "f"])
    assert b["T"]["max"][1] == pytest.approx(b["f"]["min"][1] - 0.3, abs=1e-3)
    assert b["T"]["center"][0] == pytest.approx(b["f"]["center"][0], abs=1e-3)


# ----------------------------------------------------------------------------- animation


def test_function_morph_is_pointwise_blend(studio):
    import bpy

    studio.add("function", "f", {"expression": "sin(x)"})
    studio.add("function", "g", {"expression": "x^2/4 - 2"})
    studio.anim.configure(fps=10, reset=True)
    r = studio.anim.play("transform", ["f", "g"], 1.0)
    helper = objects()[r["helper"]]
    key = helper.data.shape_keys.key_blocks["target"]
    basis = helper.data.shape_keys.key_blocks["Basis"]
    for kb_b, kb_t in list(zip(basis.data, key.data))[::25]:
        x = kb_b.co[0]
        assert kb_t.co[0] == pytest.approx(x, abs=1e-5)  # float32 storage
        assert kb_b.co[1] == pytest.approx(math.sin(x), abs=1e-5)
        assert kb_t.co[1] == pytest.approx(x * x / 4 - 2, abs=1e-5)
    bpy.context.scene.frame_set(1)
    assert objects()["f"].hide_render is True and helper.hide_render is False


def test_write_and_create_keyframes(studio):
    studio.add("function", "f", {"expression": "x"})
    studio.add("text", "T", {"latex": "x^2", "at": [0, 3]})
    studio.anim.configure(fps=10, reset=True)
    studio.anim.play("create", ["f"], 1.0)
    studio.anim.play("write", ["T"], 1.0)
    f = objects()["f"]
    assert f.data.animation_data and f.data.animation_data.action
    assert studio.anim.state()["end_frame"] == 21
    studio.anim.configure(reset=True)
    assert not any(o.name.startswith("anim") for o in objects())


def test_render_frame(studio):
    import bpy

    studio.add("function", "f", {"expression": "x"})
    bpy.context.scene.cycles.samples = 1
    r = studio.r.send("render", resolution=[64, 36], frame=1)
    assert r["width"] == 64 and r["png_base64"]


# ----------------------------------------------------------------------------- protocol


def test_socket_protocol(bpy_addon):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = bpy_addon.BridgeServer(port=port)
    server.start(use_timer=False)
    results = {}

    def client():
        conn = BlenderConnection(port=port, timeout=20)
        results["ping"] = conn.send("ping")
        try:
            conn.send("nope")
        except BlenderError as exc:
            results["error"] = str(exc)

    th = threading.Thread(target=client)
    th.start()
    deadline = time.time() + 20
    while th.is_alive() and time.time() < deadline:
        server.pump()
        time.sleep(0.01)
    server.stop()
    assert results["ping"]["addon_version"] == "0.2.0"
    assert "Unknown command" in results["error"]


def test_metadata_links_objects_to_nodes(studio):
    studio.add("function", "f", {"expression": "x", }, {"label": True})
    for name in studio.c.nodes["f"].objects:
        assert json.loads(objects()[name]["math_meta"])["node"] == "f"


def test_animation_survives_graph_updates(studio):
    import bpy

    studio.add("parameter", "a", {"value": "1"})
    studio.add("function", "f", {"expression": "a*x"})
    studio.anim.configure(fps=10, reset=True)
    studio.anim.play("create", ["f"], 1.0)
    studio.c.update("a", {"value": "2"})  # f is redrawn -> its draw-on keyframes are rebuilt
    f = objects()["f"]
    assert f.data.animation_data and f.data.animation_data.action
    bpy.context.scene.frame_set(1)
    assert f.data.bevel_factor_end == pytest.approx(0.0)
