import pytest

from blender_math_mcp import typeset as ts

BACKENDS = ["mathtext"] + (["latex"] if ts.latex_available() else [])


@pytest.mark.parametrize("backend", BACKENDS)
def test_typeset_produces_closed_bezier_outlines(backend):
    t = ts.typeset(r"\int_0^1 x^2\,dx = \frac{1}{3}", backend=backend)
    assert t.backend == backend
    assert len(t.splines) > 10
    for s in t.splines:
        assert s["cyclic"] and len(s["co"]) == len(s["hl"]) == len(s["hr"]) >= 2
    x0, y0, x1, y1 = t.bbox
    assert x1 - x0 > 3 and y0 < 0 < y1  # has depth below the baseline (limits, fraction)


def test_dollar_delimiters_are_optional():
    a = ts.typeset("$x^2$", backend="mathtext")
    b = ts.typeset("x^2", backend="mathtext")
    assert a.bbox == b.bbox


def test_mathtext_error_is_reported():
    with pytest.raises(ts.TypesetError):
        ts.typeset(r"\frac{1}", backend="mathtext")


def test_anchoring_centers_the_formula():
    t = ts.anchored(ts.typeset("x+y", backend="mathtext"), size=2.0)
    x0, y0, x1, y1 = t.bbox
    assert abs(x0 + x1) < 1e-9 and abs(y0 + y1) < 1e-9


def test_derivation_aligns_equals_signs():
    t = ts.stack_derivation([r"(a+b)^2", r"a^2+2ab+b^2", r"b^2+2ab+a^2"], backend="mathtext")
    assert len(t.source.split("=")) == 3


@pytest.mark.skipif(not ts.latex_available(), reason="LaTeX not installed")
def test_latex_matrix_column_spacing_is_correct():
    # Regression: matplotlib typesets at 100pt, which shrank \arraycolsep 10x.
    t = ts.typeset(r"\begin{pmatrix}1&2\end{pmatrix}")
    starts = sorted(min(p[0] for p in s["co"]) for s in t.splines)
    one, two = starts[1], starts[2]
    single = ts.typeset("12")
    s1 = sorted(min(p[0] for p in s["co"]) for s in single.splines)
    advance_digit = s1[1] - s1[0]  # 0.5 em
    gap = (two - one) - advance_digit  # should be 2 * \arraycolsep = 10pt = 1 em
    assert 0.9 < gap < 1.1


@pytest.mark.skipif(not ts.latex_available(), reason="LaTeX not installed")
def test_latex_environments():
    t = ts.typeset(r"f(x) = \begin{cases} x^2 & x \ge 0 \\ -x & x < 0 \end{cases}")
    assert t.backend == "latex" and t.height > 1.5
