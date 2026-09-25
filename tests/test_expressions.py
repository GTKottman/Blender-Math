import numpy as np
import pytest

from blender_math_mcp import expressions as ex


@pytest.mark.parametrize("text, expected", [
    ("x^2 + 2x + 1", "x**2 + 2*x + 1"),
    ("2 sin(x) cos(x)", "2*sin(x)*cos(x)"),
    ("sin x", "sin(x)"),
    ("e^x", "exp(x)"),
    ("2pi", "2*pi"),
])
def test_parse_calculator_syntax(text, expected):
    assert str(ex.parse(text)) == expected


@pytest.mark.parametrize("bad", ["__import__('os')", "x.__class__", "x.evalf()", "sin.func", "(x).subs(x, 1)",
                                 "lambda: 1", "exec('1')", "a; b"])
def test_parse_rejects_code(bad):
    with pytest.raises(ex.ExpressionError):
        ex.parse(bad)


def test_decimals_are_allowed():
    assert str(ex.parse("2.5 x + .5")) == "2.5*x + 0.5"


def test_implicit_product_symbols_are_the_same_symbol():
    # Regression: the x in "2x" must be the same symbol as the x in "sin(x)".
    e = ex.parse("sin(2x) - 2 sin(x) cos(x)")
    assert len(e.free_symbols) == 1


def test_numeric_is_nan_outside_the_real_domain():
    f = ex.compile_numeric(ex.parse("sqrt(x)"), ["x"])
    out = f(np.array([-1.0, 0.0, 4.0]))
    assert np.isnan(out[0]) and out[1] == 0 and out[2] == 2


def test_numeric_constant_broadcasts():
    f = ex.compile_numeric(ex.parse("3"), ["x"])
    assert f(np.zeros(4)).tolist() == [3, 3, 3, 3]


def test_numeric_rejects_unknown_parameters():
    with pytest.raises(ex.ExpressionError, match="unknown symbols"):
        ex.compile_numeric(ex.parse("a*x"), ["x"])


@pytest.mark.parametrize("lhs, rhs", [
    ("sin(2x)", "2 sin(x) cos(x)"),
    ("(x+1)^2", "x^2 + 2x + 1"),
    ("cos(theta)^2", "1 - sin(theta)^2"),
])
def test_verify_true_identities(lhs, rhs):
    assert ex.verify_equal(ex.parse(lhs), ex.parse(rhs)).status in ("proved", "numerically_equal")


def test_verify_false_identity_has_counterexample():
    v = ex.verify_equal(ex.parse("(x+1)^2"), ex.parse("x^2 + 1"))
    assert v.status == "not_equal"
    x = v.counterexample["x"]
    assert abs((x + 1) ** 2 - (x ** 2 + 1)) > 1e-6


def test_compute_definite_integral_is_exact():
    r = ex.compute("integrate", "exp(-x^2)", lower="-oo", upper="oo")
    assert r["result"] == "sqrt(pi)"
    assert r["latex"].endswith(r"= \sqrt{\pi}")


def test_compute_solve_equation():
    r = ex.compute("solve", "x^2 - 5x + 6 = 0")
    assert r["result"] == ["2", "3"]


def test_compute_diff_display_is_unambiguous():
    r = ex.compute("diff", "x sin(x)")
    assert r"\left(x \sin{\left(x \right)}\right)" in r["latex"]


@pytest.mark.parametrize("text, latex", [
    ("(x+1)(x+1)^2", r"\left(x + 1\right) \left(x + 1\right)^{2}"),  # not collapsed to ^3
    ("x - 1/x", r"x - \frac{1}{x}"),
    ("exp(-x^2/2)/sqrt(2 pi)", r"\frac{e^{- \frac{x^{2}}{2}}}{\sqrt{2 \pi}}"),
    ("sin(x)^2 + cos(x)^2 = 1", r"\sin^{2}{\left(x \right)} + \cos^{2}{\left(x \right)} = 1"),
])
def test_display_latex_keeps_the_written_form(text, latex):
    assert ex.display_latex(text) == latex
