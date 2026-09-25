import math

import numpy as np
import pytest

from blender_math_mcp import expressions as ex
from blender_math_mcp import sampling as smp


def graph(expr):
    f = ex.compile_numeric(ex.parse(expr), ["x"])
    return lambda t: np.stack([t, f(t)], axis=1)


def sample(expr, x=(-5, 5), y=(-3, 3), tol=1e-3):
    return smp.sample_curve(graph(expr), x[0], x[1], [1, 1], tol, [x[0], y[0]], [x[1], y[1]])


def test_tan_breaks_at_every_pole_and_clips_exactly():
    polys = sample("tan(x)")
    assert len(polys) == 3  # branches around -pi, 0, pi (pi/2-poles separate them)
    for p in polys:
        assert np.all(np.abs(p[:, 1]) <= 3 + 1e-12)
        # Each branch crosses the window edges at exactly tan(x) = +-3.
        assert abs(math.tan(p[0, 0]) + 3) < 1e-7 and abs(math.tan(p[-1, 0]) - 3) < 1e-7


def test_no_vertical_line_through_a_pole():
    for p in sample("1/x"):
        assert np.all(np.diff(p[:, 0]) > 0)
        assert not (p[:, 0].min() < 0 < p[:, 0].max())


def test_jump_discontinuities_are_breaks():
    polys = sample("floor(x)")
    assert len(polys) == 7  # floor = -3..3 visible
    for p in polys:
        assert np.ptp(p[:, 1]) == 0


def test_domain_edge_is_located_precisely():
    (p,) = sample("sqrt(x)")
    assert p[0, 0] < 1e-5 and p[0, 1] < 5e-3


def test_parabola_exits_at_exact_points():
    (p,) = sample("x^2")
    assert abs(p[0, 0] + math.sqrt(3)) < 1e-9 and abs(p[-1, 0] - math.sqrt(3)) < 1e-9


def test_steep_line_crossing_between_samples_is_found():
    (p,) = sample("1e6 * x")
    assert abs(p[0, 1] + 3) < 1e-9 and abs(p[-1, 1] - 3) < 1e-9


def test_smooth_curve_accuracy_within_tolerance():
    (p,) = sample("sin(x)", tol=1e-3)
    # Check the chord midpoints against the true curve.
    mid = 0.5 * (p[1:] + p[:-1])
    assert np.max(np.abs(np.sin(mid[:, 0]) - mid[:, 1])) < 2e-3


def test_implicit_circle_points_lie_on_the_circle():
    f = ex.compile_numeric(ex.parse("x^2 + y^2 - 4"), ["x", "y"])
    polys = smp.implicit_curve(f, (-3, 3), (-3, 3), resolution=200)
    assert len(polys) == 1
    r = np.hypot(polys[0][:, 0], polys[0][:, 1])
    assert np.max(np.abs(r - 2)) < 1e-9


def test_implicit_curve_ignores_poles():
    # 1/x = 0 has no solutions; the sign flip across x = 0 must not be drawn.
    f = ex.compile_numeric(ex.parse("1/x"), ["x", "y"])
    assert smp.implicit_curve(f, (-1, 1), (-1, 1), resolution=101) == []


def test_surface_drops_undefined_quads():
    f = ex.compile_numeric(ex.parse("sqrt(1 - x^2 - y^2)"), ["x", "y"])
    V, quads, _ = smp.grid_surface(lambda X, Y: np.stack([X, Y, f(X, Y)], -1), (-1, 1), (-1, 1), 20, 20)
    assert np.all(np.isfinite(V))
    assert 0 < len(quads) < 400


@pytest.mark.parametrize("value, step, style, label", [
    (math.pi / 2, math.pi / 2, "pi", r"\frac{\pi}{2}"),
    (-math.pi, math.pi / 2, "pi", r"-\pi"),
    (3 * math.pi / 2, math.pi / 2, "pi", r"\frac{3\pi}{2}"),
    (-2.5, 2.5, "decimal", "-2.5"),
    (0.1, 0.1, "decimal", "0.1"),
    (4.0, 2.0, "decimal", "4"),
])
def test_tick_labels(value, step, style, label):
    assert smp.tick_label(value, step, style) == label
