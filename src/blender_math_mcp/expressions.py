"""Parsing, symbolic computation and numeric evaluation of math expressions.

Expressions are written in SymPy / calculator syntax, e.g. ``sin(x)/x``,
``x^2 + 2x + 1`` (``^`` and implicit multiplication are accepted),
``sqrt(1 - x**2)``, ``Piecewise((x, x < 0), (x**2, True))``.

The same parsed SymPy object is used both to produce the LaTeX that is shown
and to generate the numbers that are plotted, so the picture and the formula
can never disagree.
"""

from __future__ import annotations

import random
import re
import warnings
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)

# Names available inside expressions.  Everything else becomes a Symbol.
_ALLOWED = {
    name: getattr(sp, name)
    for name in (
        "sin cos tan cot sec csc asin acos atan atan2 acot asec acsc "
        "sinh cosh tanh coth sech csch asinh acosh atanh "
        "exp log ln sqrt cbrt root Abs sign floor ceiling frac factorial binomial gamma "
        "beta erf erfc zeta besselj bessely Min Max Piecewise Heaviside DiracDelta "
        "re im arg conjugate pi E I oo Rational Integer Float Symbol Function "
        "Sum Product Integral Derivative Limit Matrix Eq Ne Lt Le Gt Ge And Or Not "
        "diff integrate limit summation product series simplify expand factor "
        "binomial_coefficients legendre chebyshevt hermite laguerre"
    ).split()
    if hasattr(sp, name)
}
_ALLOWED.update({"e": sp.E, "abs": sp.Abs, "ln": sp.log, "inf": sp.oo, "infinity": sp.oo,
                 # real roots (x^(1/3) is the principal root, undefined for x < 0)
                 "cbrt": lambda a: sp.real_root(a, 3), "root": sp.real_root, "real_root": sp.real_root})

# parse_expr needs a handful of SymPy constructors in its global namespace.
_BASE_GLOBALS = {name: getattr(sp, name) for name in ("Symbol", "Integer", "Float", "Rational", "Function",
                                                     "Mul", "Add", "Pow")}


# No dunders, no attribute access (a '.' after a name/bracket), no statements.
_FORBIDDEN = re.compile(
    r"__|[A-Za-z_)\]]\s*\.\s*[A-Za-z_]"
    r"|\b(import|lambda|exec|eval|open|compile|globals|locals|getattr|setattr|vars|dir)\b|[;`]"
)


class ExpressionError(ValueError):
    """Raised for expressions that cannot be parsed or evaluated."""


def parse(text: str | int | float, variables: Sequence[str] = (), evaluate: bool = True,
          namespace: dict | None = None) -> sp.Expr:
    """Parse ``text`` into a SymPy expression (safely; no arbitrary code).

    ``namespace`` adds names (user functions as ``Lambda``, parameter values...).

    ``evaluate=False`` keeps the expression exactly as written (``(x+1)(x+1)^2``
    is not collapsed into ``(x+1)^3``); use it for display.
    """
    if isinstance(text, (int, float)):
        return sp.nsimplify(text) if isinstance(text, float) else sp.Integer(text)
    if not isinstance(text, str) or not text.strip():
        raise ExpressionError("Empty expression.")
    if _FORBIDDEN.search(text):
        raise ExpressionError(f"Expression {text!r} contains forbidden syntax.")
    local = dict(_ALLOWED)
    if namespace:
        local.update(namespace)
    for v in variables:
        local[v] = sp.Symbol(v, real=True)
    # Every free name is a real symbol (also inside implicit products like "2x";
    # multi-letter names such as theta stay whole instead of being split).
    for name in set(re.findall(r"[A-Za-z_]\w*", text)):
        if name not in local:
            local[name] = sp.Symbol(name, real=True)
    try:
        expr = parse_expr(text, local_dict=local, global_dict={"__builtins__": {}, **_BASE_GLOBALS},
                          transformations=_TRANSFORMS, evaluate=evaluate)
    except Exception as exc:
        raise ExpressionError(f"Could not parse {text!r}: {exc}") from exc
    if isinstance(expr, (tuple, list)):
        raise ExpressionError(f"{text!r} is a tuple; pass one expression at a time.")
    return expr


def is_reserved(name: str) -> bool:
    """Names that cannot be used for user objects (functions, constants, variables)."""
    return name in _ALLOWED or name in ("x", "y", "z", "t", "theta")


_EQ = re.compile(r"(?<![<>!=])=(?!=)")


def parse_equation(text: str, evaluate: bool = True, namespace: dict | None = None) -> sp.Basic:
    """Parse ``'lhs = rhs'`` into ``Eq(lhs, rhs)``; plain expressions are returned as-is."""
    if isinstance(text, str) and _EQ.search(text):
        lhs, rhs = _EQ.split(text, maxsplit=1)
        return sp.Eq(parse(lhs, evaluate=evaluate, namespace=namespace),
                     parse(rhs, evaluate=evaluate, namespace=namespace), evaluate=False)
    return parse(text, evaluate=evaluate, namespace=namespace)


def _tidy(e: sp.Basic) -> sp.Basic:
    """Clean an unevaluated tree for printing without changing its structure.

    Flattens nested products and folds their numeric factors into one leading
    coefficient, which removes printing artifacts such as ``x - 1 \\frac{1}{x}``
    or ``(-1) x^2``, while keeping e.g. ``(x+1)(x+1)^2`` as written.
    """
    if e.is_Atom or not e.args:
        return e
    args = [_tidy(a) for a in e.args]
    if isinstance(e, sp.Mul):
        flat = []
        for a in args:
            flat.extend(a.args if isinstance(a, sp.Mul) else [a])
        coeff = sp.Mul(*[a for a in flat if a.is_Number])
        rest = [a for a in flat if not a.is_Number]
        if not rest:
            return coeff
        return sp.Mul(*([] if coeff == 1 else [coeff]), *rest, evaluate=False)
    if isinstance(e, (sp.Add, sp.Pow)):
        return e.func(*args, evaluate=False)
    return e.func(*args)


def display_latex(text: str) -> str:
    """LaTeX for an expression/equation exactly as written (no simplification)."""
    e = parse_equation(text, evaluate=False)
    if isinstance(e, sp.Equality):
        return f"{sp.latex(_tidy(e.lhs), order='none')} = {sp.latex(_tidy(e.rhs), order='none')}"
    return sp.latex(_tidy(e), order="none")


def to_latex(expr: sp.Basic, **kw) -> str:
    return sp.latex(expr, **kw)


def free_symbol_names(expr: sp.Basic) -> list[str]:
    return sorted(s.name for s in expr.free_symbols)


def compile_numeric(expr: sp.Expr, variables: Sequence[str]) -> Callable[..., np.ndarray]:
    """Vectorised real-valued numeric function of ``expr``.

    Non-real results and domain errors become NaN (never silently wrong values).
    """
    syms = [sp.Symbol(v, real=True) for v in variables]
    # Unify symbols that were created without assumptions.
    subs = {s: sp.Symbol(s.name, real=True) for s in expr.free_symbols if s.name in variables}
    expr = expr.subs(subs)
    extra = sorted(s.name for s in expr.free_symbols if s.name not in variables)
    if extra:
        raise ExpressionError(
            f"Expression has unknown symbols {extra}; expected only {list(variables)}. "
            "Substitute numeric values for parameters first."
        )
    fn = sp.lambdify(syms, expr, modules=["numpy", "scipy"] if _has_scipy() else ["numpy"])

    def f(*args):
        shape = np.broadcast(*[np.asarray(a) for a in args]).shape
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            try:
                out = fn(*[np.asarray(a, dtype=complex) for a in args])
            except (TypeError, ValueError, ZeroDivisionError):
                # Fall back to pointwise evaluation for functions numpy can't vectorise.
                out = np.vectorize(_safe_scalar(fn), otypes=[complex])(*args)
        out = np.broadcast_to(np.asarray(out, dtype=complex), shape)
        real = out.real.copy()
        tol = 1e-9 * np.maximum(1.0, np.abs(out.real))
        real[np.abs(out.imag) > tol] = np.nan
        real[~np.isfinite(real)] = np.nan
        return real

    return f


def _safe_scalar(fn):
    def g(*a):
        try:
            return complex(fn(*a))
        except Exception:
            return complex(np.nan)

    return g


_SCIPY = None


def _has_scipy() -> bool:
    global _SCIPY
    if _SCIPY is None:
        try:
            import scipy  # noqa: F401  (availability probe)

            _SCIPY = True
        except ImportError:
            _SCIPY = False
    return _SCIPY


# --------------------------------------------------------------------------- symbolic tools


@dataclass
class Verification:
    status: str  # "proved", "numerically_equal", "not_equal", "unknown"
    detail: str
    counterexample: dict | None = None

    def as_dict(self):
        d = {"status": self.status, "detail": self.detail}
        if self.counterexample:
            d["counterexample"] = self.counterexample
        return d


def verify_equal(lhs: sp.Expr, rhs: sp.Expr, samples: int = 40, seed: int = 0) -> Verification:
    """Decide whether ``lhs == rhs`` identically (symbolically, then numerically)."""
    lhs, rhs = _unify_symbols(lhs, rhs)
    try:
        diff = sp.simplify(lhs - rhs)
        if diff == 0:
            return Verification("proved", "simplify(lhs - rhs) == 0")
        if diff.is_number and diff.is_zero is False:
            return Verification("not_equal", f"lhs - rhs simplifies to the nonzero constant {diff}")
        if sp.expand(sp.expand_trig(sp.expand_log(diff, force=True))) == 0:
            return Verification("proved", "expansion of lhs - rhs is 0")
    except Exception:  # simplify can fail on exotic input; fall through to numerics
        diff = lhs - rhs

    syms = sorted(diff.free_symbols, key=lambda s: s.name)
    rng = random.Random(seed)
    checked = 0
    for _ in range(samples * 3):
        point = {s: sp.Float(rng.uniform(-3, 3) if not s.is_positive else rng.uniform(0.1, 3)) for s in syms}
        try:
            a = complex(lhs.evalf(30, subs=point))
            b = complex(rhs.evalf(30, subs=point))
        except Exception:
            continue
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        checked += 1
        if abs(a - b) > 1e-9 * max(1.0, abs(a), abs(b)):
            return Verification(
                "not_equal",
                f"lhs = {a:.12g}, rhs = {b:.12g} at a sample point",
                {s.name: float(v) for s, v in point.items()},
            )
        if checked >= samples:
            break
    if checked == 0:
        return Verification("unknown", "could not simplify and no sample point evaluated")
    return Verification(
        "numerically_equal",
        f"not proved symbolically, but equal at {checked} random points (rel. tol 1e-9)",
    )


def _unify_symbols(*exprs):
    """Make same-named symbols identical (real) across expressions."""
    names = {s.name for e in exprs for s in e.free_symbols}
    real = {n: sp.Symbol(n, real=True) for n in names}
    return [e.subs({s: real[s.name] for s in e.free_symbols}) for e in exprs]


OPERATIONS = (
    "simplify", "expand", "factor", "cancel", "apart", "together", "trigsimp",
    "diff", "integrate", "limit", "series", "solve", "evalf", "sum",
)


def compute(operation: str, expression: str, variable: str | None = None,
            lower: str | None = None, upper: str | None = None,
            point: str | None = None, order: int = 1, digits: int = 15) -> dict:
    """Perform a symbolic operation and return the result with LaTeX for display."""
    op = operation.lower()
    if op not in OPERATIONS:
        raise ExpressionError(f"Unknown operation {operation!r}; choose from {', '.join(OPERATIONS)}")
    if op == "solve":
        expr = parse_equation(expression)
    else:
        expr = parse(expression)
    var = sp.Symbol(variable, real=True) if variable else _default_var(expr)
    if var is not None:
        expr = expr.subs({s: var for s in expr.free_symbols if s.name == var.name})

    display = None  # LaTeX for "input = result"
    if op in ("simplify", "expand", "factor", "cancel", "together", "trigsimp"):
        result = getattr(sp, op)(expr)
    elif op == "apart":
        result = sp.apart(expr, var)
    elif op == "diff":
        _need(var, op)
        result = sp.diff(expr, var, order)
        sup = f"^{{{order}}}" if order > 1 else ""
        body = sp.latex(expr)
        if isinstance(expr, (sp.Add, sp.Mul)):
            body = rf"\left({body}\right)"
        display = rf"\frac{{d{sup}}}{{d{sp.latex(var)}{sup}}}{body} = {sp.latex(result)}"
    elif op == "integrate":
        _need(var, op)
        if lower is not None and upper is not None:
            a, b = parse(lower), parse(upper)
            result = sp.integrate(expr, (var, a, b))
            lhs = sp.Integral(expr, (var, a, b))
        else:
            result = sp.integrate(expr, var)
            lhs = sp.Integral(expr, var)
            display = f"{sp.latex(lhs)} = {sp.latex(result)} + C"
        if display is None:
            display = f"{sp.latex(lhs)} = {sp.latex(result)}"
    elif op == "sum":
        _need(var, op)
        if lower is None or upper is None:
            raise ExpressionError("sum needs lower and upper bounds")
        a, b = parse(lower), parse(upper)
        result = sp.summation(expr, (var, a, b))
        display = f"{sp.latex(sp.Sum(expr, (var, a, b)))} = {sp.latex(result)}"
    elif op == "limit":
        _need(var, op)
        if point is None:
            raise ExpressionError("limit needs a point (e.g. '0' or 'oo')")
        p = parse(point)
        result = sp.limit(expr, var, p)
        display = f"{sp.latex(sp.Limit(expr, var, p))} = {sp.latex(result)}"
    elif op == "series":
        _need(var, op)
        p = parse(point) if point is not None else 0
        result = sp.series(expr, var, p, n=order if order > 1 else 6)
    elif op == "solve":
        eq = expr
        _need(var, op)
        result = sp.solve(eq, var, dict=False)
        sol_tex = ", ".join(sp.latex(r) for r in result) if result else r"\varnothing"
        display = f"{sp.latex(var)} \\in \\left\\{{ {sol_tex} \\right\\}}" if result else f"{sp.latex(var)} \\in \\varnothing"
        return {
            "operation": op,
            "input": str(expression),
            "result": [str(r) for r in result],
            "latex": display,
            "numeric": [_num(r, digits) for r in result],
        }
    else:  # evalf
        result = sp.N(expr, digits)

    if display is None:
        display = f"{sp.latex(expr)} = {sp.latex(result)}"
    return {
        "operation": op,
        "input": str(expression),
        "result": str(result),
        "result_latex": sp.latex(result),
        "latex": display,
        "numeric": _num(result, digits),
    }


def _num(v, digits):
    try:
        if getattr(v, "free_symbols", None):
            return None
        return str(sp.N(v, digits))
    except Exception:
        return None


def _default_var(expr) -> sp.Symbol | None:
    syms = sorted(expr.free_symbols, key=lambda s: s.name)
    for preferred in ("x", "t", "n", "z", "y"):
        for s in syms:
            if s.name == preferred:
                return s
    return syms[0] if syms else None


def _need(var, op):
    if var is None:
        raise ExpressionError(f"{op} needs a variable")
