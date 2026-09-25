import sympy as sp
import pytest

from blender_math_mcp import keypoints as kp
from blender_math_mcp.expressions import parse

x = sp.Symbol("x", real=True)
y = sp.Symbol("y", real=True)


def pts(lst):
    return [(sp.simplify(k.x), sp.simplify(k.y), k.kind, k.exact) for k in lst]


def test_exact_roots():
    assert [k.x for k in kp.roots(parse("x^2 - 2"), x, -3, 3)] == [-sp.sqrt(2), sp.sqrt(2)]
    assert [k.x for k in kp.roots(parse("sin(x)"), x, -7, 7)] == [-2 * sp.pi, -sp.pi, 0, sp.pi, 2 * sp.pi]


def test_numeric_root_is_marked_and_accurate():
    (k,) = kp.roots(parse("cos(x) - x"), x, -3, 3)
    assert not k.exact
    assert abs(float(sp.cos(k.x) - k.x)) < 1e-25


def test_pole_is_not_a_root():
    assert kp.roots(parse("1/x"), x, -3, 3) == []


def test_double_root_found():
    assert [k.x for k in kp.roots(parse("(x-1)^2"), x, -3, 3)] == [1]


def test_extrema_classified():
    assert pts(kp.extrema(parse("x^3 - 3x"), x, -3, 3)) == [(-1, 2, "local_max", True), (1, -2, "local_min", True)]
    assert kp.extrema(parse("x^3"), x, -3, 3) == []  # stationary but not an extremum
    assert pts(kp.extrema(parse("abs(x)"), x, -3, 3)) == [(0, 0, "local_min", True)]  # corner


def test_inflections():
    assert [k.x for k in kp.inflections(parse("x^3 - 3x"), x, -3, 3)] == [0]
    assert kp.inflections(parse("x^4"), x, -3, 3) == []  # f'' = 0 but no sign change


def test_intersections_exact():
    got = pts(kp.intersections(parse("sin(x)"), parse("cos(x)"), x, -4, 4))
    assert got[1] == (sp.pi / 4, sp.sqrt(2) / 2, "intersection", True)
    assert len(got) == 3


@pytest.mark.parametrize("expr, x0, reason", [
    ("abs(x)", 0, "not differentiable"),
    ("x^(1/3)", 0, "vertical tangent"),
    ("sqrt(x)", 0, "vertical tangent"),
    ("1/x", 0, "not defined"),
    ("sqrt(x)", -1, "not real"),
])
def test_tangent_refusals(expr, x0, reason):
    with pytest.raises(kp.MathRefusal, match=reason):
        kp.require_differentiable(parse(expr), x, sp.Integer(x0))


def test_system_finds_all_periodic_solutions():
    sols = kp.solve_system_box(y, -sp.sin(x) - y / 3, x, y, (-7, 7, -3, 3))
    assert [s[0][0] for s in sols] == [-2 * sp.pi, -sp.pi, 0, sp.pi, 2 * sp.pi]
    assert all(s[1] for s in sols)
