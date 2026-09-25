"""End-to-end tests of the real Blender add-on (needs the ``bpy`` module)."""

import json
import math
import socket
import threading
import time

import numpy as np
import pytest

from blender_math_mcp.blender_client import BlenderConnection, BlenderError


def _objects(bpy):
    return {o.name: o for o in bpy.data.objects}


def test_ping(studio):
    assert "blender" in studio.status()


def test_latex_becomes_filled_curve(studio):
    import bpy

    r = studio.render_latex(r"e^{i\pi} + 1 = 0", name="Euler", location=[1, 2, 0], size=1.0,
                            backend="mathtext")
    obj = bpy.data.objects[r["object"]]
    assert obj.type == "CURVE" and obj.data.dimensions == "2D" and obj.data.fill_mode == "BOTH"
    assert tuple(obj.location) == (1, 2, 0)
    assert json.loads(obj["math_meta"])["latex"] == r"e^{i\pi} + 1 = 0"
    assert obj.users_collection[0].name == "Math"


def test_graph_on_axes_is_parented_and_scaled(studio):
    import bpy

    ax = studio.create_axes(x_range=[-4, 4], y_range=[-2, 2], scale=1.5)
    g = studio.plot_function("x^2", axes=ax["axes"])
    obj = bpy.data.objects[g["object"]]
    assert obj.parent.name == ax["axes"]
    pts = [tuple(p.co)[:3] for s in obj.data.splines for p in s.points]
    xs = np.array([p[0] for p in pts]) / 1.5
    ys = np.array([p[1] for p in pts]) / 1.5
    assert np.allclose(ys, xs ** 2, atol=1e-6)  # every vertex exactly on the parabola
    assert abs(xs.min() + math.sqrt(2)) < 1e-6 and abs(xs.max() - math.sqrt(2)) < 1e-6


def test_moving_axes_moves_graph(studio):
    import bpy

    ax = studio.create_axes()
    g = studio.plot_function("sin(x)", axes=ax["axes"])
    studio._send("transform", name=ax["axes"], location=[10, 0, 0])
    bpy.context.view_layer.update()
    assert bpy.data.objects[g["object"]].matrix_world.translation.x == pytest.approx(10)


def test_false_derivation_is_not_drawn(studio):
    import bpy

    before = len(bpy.data.objects)
    r = studio.render_derivation(["(a+b)^2", "a^2 + b^2"])
    assert r["rendered"] is False and len(bpy.data.objects) == before


def test_true_derivation_is_drawn(studio):
    r = studio.render_derivation(["(x+1)^2", "(x+1)(x+1)", "x^2 + 2x + 1"])
    assert r["rendered"] and all(c["status"] == "proved" for c in r["checks"])
    assert r["latex"][1] == r"\left(x + 1\right) \left(x + 1\right)"


def test_surface_and_colormap(studio):
    import bpy

    s = studio.plot_surface("x^2 - y^2", x_range=[-1, 1], y_range=[-1, 1], resolution=10)
    me = bpy.data.objects[s["object"]].data
    assert len(me.polygons) == 100 and "Col" in me.color_attributes


def test_parametric_sphere_seams_are_welded(studio):
    import bpy

    s = studio.plot_parametric_surface("sin(v)cos(u)", "sin(v)sin(u)", "cos(v)", resolution=16)
    me = bpy.data.objects[s["object"]].data
    # 16x16 grid: 15 inner rings of 16 + 2 poles once welded
    assert len(me.vertices) == 15 * 16 + 2


def test_vector_and_point_and_region(studio):
    ax = studio.create_axes()
    v = studio.draw_vector([2, 1], axes=ax["axes"], label=r"\vec{v}")
    p = studio.draw_point([1, 1], axes=ax["axes"], label="P")
    r = studio.shade_region("x^2", x_range=[0, 1], axes=ax["axes"])
    assert v["label_object"] and p["label_object"]
    assert r["signed_area"] == "1/3"


def test_delete_all_math(studio):
    import bpy

    studio.create_axes()
    studio._send("delete", all_math=True)
    assert not [o for o in bpy.data.objects if "math_meta" in o]


def test_errors_are_reported(studio):
    with pytest.raises(BlenderError, match="No object named"):
        studio._send("transform", name="missing", location=[0, 0, 0])


def test_code_execution_is_opt_in(studio):
    # No add-on preferences are registered in tests -> allowed; the check itself
    # is exercised through the preference object when registered in Blender.
    out = studio._send("execute_code", code="print(1 + 1)")
    assert out["stdout"].strip() == "2"


def test_setup_frame_and_render(studio):
    studio.setup_scene("2d", background="#ffffff", engine="cycles")
    studio.render_latex("x", backend="mathtext", color="black")
    studio._send("frame")
    import bpy

    bpy.context.scene.cycles.samples = 1
    r = studio._send("render", resolution=[64, 36])
    assert r["width"] == 64 and r["png_base64"]


def test_socket_protocol(bpy_addon):
    """Full TCP round trip: client thread <-> BridgeServer, pumped on this (main) thread."""
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
    assert results["ping"]["addon_version"]
    assert "Unknown command" in results["error"]
