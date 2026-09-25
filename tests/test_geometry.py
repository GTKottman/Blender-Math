import numpy as np
import pytest

from blender_math_mcp import geometry as geo


def test_colors_are_linearised_exactly():
    assert geo.parse_color("#ffffff") == [1.0, 1.0, 1.0, 1.0]
    r, g, b, a = geo.parse_color("#808080")
    assert abs(r - 0.2158605) < 1e-6 and a == 1.0
    assert geo.parse_color("math_blue") == geo.parse_color("#58C4DD")
    assert geo.parse_color([1, 0, 0, 0.5]) == [1.0, 0.0, 0.0, 0.5]


def test_unknown_color_raises():
    with pytest.raises(ValueError):
        geo.parse_color("not-a-color")


@pytest.mark.parametrize("style", ["3d", "flat"])
def test_arrow_tip_is_exactly_the_end_point(style):
    v, f = geo.arrow_mesh([0, 0, 0], [1, 2, 0], 0.05, style=style)
    V = np.array(v)
    d = np.linalg.norm(V - np.array([1, 2, 0]), axis=1)
    assert d.min() == 0.0
    # The arrow never extends beyond its end point along its direction.
    u = np.array([1, 2, 0]) / np.sqrt(5)
    assert (V @ u).max() <= np.sqrt(5) + 1e-12


def test_bezier_circle_is_accurate():
    s = geo.bezier_circle([0, 0, 0], 2.0)
    co, hr, hl = map(np.array, (s["co"], s["hr"], s["hl"]))
    t = np.linspace(0, 1, 50)[:, None]
    for i in range(4):
        p0, p1, p2, p3 = co[i], hr[i], hl[(i + 1) % 4], co[(i + 1) % 4]
        pts = (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3
        r = np.linalg.norm(pts[:, :2], axis=1)
        assert np.max(np.abs(r - 2)) / 2 < 3e-4
