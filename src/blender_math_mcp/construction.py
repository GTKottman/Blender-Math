"""The construction: a dependency graph of mathematical objects (GeoGebra-style).

Every object -- parameter, function, point, tangent, circle, label ... -- is a
*node* with a definition (``spec``) that may refer to other nodes by name:

    a = 2                          (parameter)
    f(x) = a*sin(x)                (function, depends on a)
    A = point on f at x = pi/4     (depends on f)
    t = tangent to f at A          (depends on f, A)
    label "f'(A_x) = {f'(A_x)}"    (depends on f, A)

Changing a node (``update``) recomputes all dependents in topological order.
Updates are *transactional*: every affected node is recomputed first; if any
of them becomes mathematically invalid (e.g. the intersection a point was
defined by no longer exists), nothing changes and the reason is returned.
Only then are the Blender objects of the affected nodes rebuilt.

The graph is stored in the .blend file, so it survives server restarts.

Names in expressions
--------------------
* parameters: ``a``        * functions: ``f(x)``, derivatives ``f'(x)``, ``f''(2)``
* points / vectors: ``A_x``, ``A_y`` (``A_z``)   * scalar results: ``s1`` (segment length),
  ``alpha`` (angle), ``R`` (Riemann sum value) ...
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import sympy as sp

from . import expressions as ex
from .keypoints import MathRefusal
from .render import Frame, Renderer
from .themes import get_theme

X = sp.Symbol("x", real=True)
Y = sp.Symbol("y", real=True)
T = sp.Symbol("t", real=True)

_PRIME = re.compile(r"\b([A-Za-z]\w*)('+)\s*\(")
_IDENT = re.compile(r"[A-Za-z_]\w*")
_NAME_OK = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
PRIME_SUFFIX = "_Dprime"


@dataclass
class Node:
    name: str
    kind: str
    spec: dict
    style: dict = field(default_factory=dict)
    deps: set = field(default_factory=set)
    data: dict = field(default_factory=dict)
    objects: list = field(default_factory=list)

    def copy(self) -> "Node":
        return Node(self.name, self.kind, dict(self.spec), dict(self.style), set(self.deps), dict(self.data),
                    list(self.objects))


class Kind:
    """Base class for node kinds.  Subclasses implement ``compute`` and ``draw``."""

    name = "base"
    needs_axes: int | None = None  # 2 or 3 if the node draws in an axes frame
    text_fields: tuple[str, ...] = ()  # spec fields holding expressions (dependency scan)
    ref_fields: tuple[str, ...] = ()  # spec fields holding node names (or lists of names)

    def refs(self, c: "Construction", spec: dict) -> set[str]:
        out = set()
        for f in self.text_fields:
            v = spec.get(f)
            for item in _flatten(v):
                if isinstance(item, str):
                    out |= c.env_refs(item)
        for f in self.ref_fields:
            for item in _flatten(spec.get(f)):
                if isinstance(item, str) and item in c.nodes:
                    out.add(item)
                elif isinstance(item, str) and f != "axes":
                    # a reference may also be an expression ("A" or "2*a")
                    out |= c.env_refs(item)
        if spec.get("axes"):
            out.add(spec["axes"])
        return out

    def compute(self, c: "Construction", node: Node) -> dict:  # pragma: no cover - abstract
        raise NotImplementedError

    def draw(self, c: "Construction", node: Node) -> list[str]:
        return []

    def provides(self, node: Node) -> dict:
        """Names this node contributes to expressions."""
        return {}

    def describe(self, node: Node) -> dict:
        return {}


def _flatten(v):
    if isinstance(v, (list, tuple)):
        for x in v:
            yield from _flatten(x)
    elif v is not None:
        yield v


KINDS: dict[str, Kind] = {}


def register(kind_cls):
    KINDS[kind_cls.name] = kind_cls()
    return kind_cls


class Construction:
    def __init__(self, renderer: Renderer):
        self.r = renderer
        self.nodes: dict[str, Node] = {}
        self.theme_name = renderer.theme.name
        self._color_index = 0
        self.loaded = False

    # ------------------------------------------------------------------ environment

    def namespace(self, exclude: str | None = None) -> dict:
        ns: dict[str, Any] = {}
        for n in self.nodes.values():
            if n.name == exclude or not n.data:
                continue
            ns.update(KINDS[n.kind].provides(n))
        return ns

    def _preprocess(self, text: str) -> str:
        return _PRIME.sub(lambda m: f"{m.group(1)}{PRIME_SUFFIX}{len(m.group(2))}(", text)

    def env_refs(self, text: str) -> set[str]:
        """Node names referenced by an expression string."""
        if not isinstance(text, str):
            return set()
        names = set()
        for ident in _IDENT.findall(self._preprocess(text)):
            base = ident.split(PRIME_SUFFIX)[0]
            if base in self.nodes:
                names.add(base)
            elif "_" in ident:
                head = ident.rsplit("_", 1)[0]
                if head in self.nodes:
                    names.add(head)
        return names

    def parse(self, text, variables=("x",), exclude: str | None = None, evaluate: bool = True) -> sp.Expr:
        """Parse with all node values available; results are exact SymPy expressions."""
        if isinstance(text, (int, float)):
            return ex.parse(text)
        ns = self.namespace(exclude)
        # derivative helpers for the functions actually referenced with primes
        s = self._preprocess(text)
        for m in re.finditer(rf"\b([A-Za-z]\w*){PRIME_SUFFIX}(\d+)\b", s):
            base, order = m.group(1), int(m.group(2))
            lam = ns.get(base)
            if not isinstance(lam, sp.Lambda):
                raise MathRefusal(f"{base!r} is not a function, so {base}{chr(39) * order} is undefined.")
            v = lam.variables[0]
            ns[f"{base}{PRIME_SUFFIX}{order}"] = sp.Lambda(v, sp.diff(lam.expr, v, order))
        try:
            e = ex.parse(s, variables, evaluate=evaluate, namespace=ns)
        except ex.ExpressionError as exc:
            raise MathRefusal(str(exc).replace(PRIME_SUFFIX + "1", "'").replace(PRIME_SUFFIX + "2", "''")) from exc
        if isinstance(e, sp.Lambda):
            raise MathRefusal(f"{text!r} is a function; call it, e.g. {text}(x).")
        # Unify symbols to real ones by name (variables x, y, t, ... inside Lambdas).
        e = e.subs({s_: sp.Symbol(s_.name, real=True) for s_ in e.free_symbols})
        return e

    def value(self, text, exclude=None) -> sp.Expr:
        """A constant (no free symbols) -- e.g. a bound or coordinate."""
        e = self.parse(text, (), exclude)
        free = sorted(s.name for s in e.free_symbols)
        if free:
            raise MathRefusal(f"{text!r} must be a number but depends on {free}. Define them as parameters "
                              "(set_parameter) or substitute values.")
        n = sp.N(e, 30)
        if not n.is_finite or n.is_real is False:
            raise MathRefusal(f"{text!r} = {e} is not a finite real number.")
        return sp.nsimplify(e) if e.is_Float else e

    def function_of(self, ref: str, var=X, exclude=None) -> sp.Expr:
        """Expression of a function-like node, or a free expression in ``var``."""
        node = self.nodes.get(ref)
        if node is not None:
            lam = KINDS[node.kind].provides(node).get(node.name)
            if not isinstance(lam, sp.Lambda):
                raise MathRefusal(f"{ref!r} ({node.kind}) is not a function of x.")
            return lam.expr.subs(lam.variables[0], var)
        e = self.parse(ref, (var.name,), exclude)
        extra = sorted(s.name for s in e.free_symbols if s != var)
        if extra:
            raise MathRefusal(f"{ref!r} depends on unknown symbols {extra}.")
        return e

    def point_of(self, ref, exclude=None) -> list[sp.Expr]:
        """Exact coordinates of a point node, or of a literal [x, y(, z)] list."""
        if isinstance(ref, str) and ref in self.nodes:
            n = self.nodes[ref]
            if "point" not in n.data:
                raise MathRefusal(f"{ref!r} is a {n.kind}, not a point.")
            return list(n.data["point"])
        if isinstance(ref, (list, tuple)) and 2 <= len(ref) <= 3:
            return [self.value(v, exclude) for v in ref]
        raise MathRefusal(f"{ref!r} is not a point (give a point name or [x, y]).")

    # ------------------------------------------------------------------ frames

    def axes_names(self, dims: int | None = None) -> list[str]:
        return [n.name for n in self.nodes.values() if n.kind == "axes" and (dims is None or n.data.get("dims") == dims)]

    def frame(self, axes: str | None) -> Frame:
        if not axes:
            return Frame(scale=np.ones(3), dims=2)
        node = self.nodes.get(axes)
        if node is None or node.kind != "axes":
            raise MathRefusal(f"{axes!r} is not an axes object.")
        d = node.data
        lo = np.array([d["x_range"][0], d["y_range"][0], (d.get("z_range") or [-1e9, 1e9])[0]], float)
        hi = np.array([d["x_range"][1], d["y_range"][1], (d.get("z_range") or [-1e9, 1e9])[1]], float)
        return Frame(np.asarray(d["scale"], float), parent=axes, lo=lo, hi=hi, dims=d["dims"],
                     origin=np.asarray(d["location"], float))

    def resolve_axes(self, spec: dict, dims: int) -> str:
        """Use the given axes, the most recent axes of that dimension, or create default axes."""
        if spec.get("axes"):
            if spec["axes"] not in self.nodes:
                raise MathRefusal(f"No axes named {spec['axes']!r}.")
            return spec["axes"]
        names = self.axes_names(dims) or (self.axes_names(3) if dims == 2 else [])
        if names:
            return names[-1]
        name = "axes" if "axes" not in self.nodes else f"axes{dims}d"
        if dims == 2:
            self.add("axes", name, {"x_range": [-7.5, 7.5], "y_range": [-4, 4], "scale": 1})
        else:
            self.add("axes", name, {"x_range": [-3, 3], "y_range": [-3, 3], "z_range": [-2, 2]})
        return name

    # ------------------------------------------------------------------ styles

    def auto_color(self) -> str:
        pal = self.r.theme.palette
        c = pal[self._color_index % len(pal)]
        self._color_index += 1
        return c

    def color(self, node: Node, key: str = "color") -> str:
        v = node.style.get(key)
        if v in (None, "auto"):
            return self.r.theme.text
        return v

    # ------------------------------------------------------------------ graph operations

    def add(self, kind: str, name: str | None, spec: dict, style: dict | None = None) -> dict:
        self.ensure_loaded()
        if kind not in KINDS:
            raise MathRefusal(f"Unknown kind {kind!r}.")
        k = KINDS[kind]
        name = name or self.auto_name(kind)
        if not _NAME_OK.match(name) or PRIME_SUFFIX in name:
            raise MathRefusal(f"Invalid name {name!r}: use letters, digits and '_' (starting with a letter).")
        if ex.is_reserved(name):
            raise MathRefusal(f"{name!r} is a reserved name (function/constant/variable); choose another.")
        if name in self.nodes:
            raise MathRefusal(f"{name!r} already exists; use update_object to change it or delete it first.")
        spec = {k_: v for k_, v in (spec or {}).items() if v is not None}
        style = {k_: v for k_, v in (style or {}).items() if v is not None}
        for f in k.text_fields:  # a definition may not refer to the object being defined
            for item in _flatten(spec.get(f)):
                if isinstance(item, str) and name in _IDENT.findall(self._preprocess(item).replace(PRIME_SUFFIX, " ")):
                    raise MathRefusal(f"{name} cannot be defined in terms of itself ({item!r}).")
        dims = k.needs_axes(self, spec) if callable(k.needs_axes) else k.needs_axes
        if dims and not spec.get("axes") and kind != "axes":
            spec["axes"] = self.resolve_axes(spec, dims)
        if style.get("color") in (None, "auto") and getattr(k, "colored", False):
            style["color"] = self.auto_color()
        node = Node(name, kind, spec, style)
        node.deps = k.refs(self, spec)
        missing = [d for d in node.deps if d not in self.nodes]
        if missing:
            raise MathRefusal(f"{name}: unknown objects {missing}.")
        self._commit([node])
        return self.describe(name)

    def update(self, name: str, spec: dict | None = None, style: dict | None = None) -> dict:
        self.ensure_loaded()
        if name not in self.nodes:
            raise MathRefusal(f"No object named {name!r}.")
        new = self.nodes[name].copy()
        for k_, v in (spec or {}).items():
            if v is None:
                new.spec.pop(k_, None)
            else:
                new.spec[k_] = v
        for k_, v in (style or {}).items():
            if v is None:
                new.style.pop(k_, None)
            else:
                new.style[k_] = v
        new.deps = KINDS[new.kind].refs(self, new.spec)
        if name in new.deps or new.deps & self.descendants(name):
            raise MathRefusal(f"{name} would depend on itself (circular definition).")
        affected = [new] + [self.nodes[n].copy() for n in self.topo_order(self.descendants(name))]
        self._commit(affected)
        out = self.describe(name)
        out["recomputed"] = [n.name for n in affected[1:]]
        return out

    def delete(self, name: str) -> list[str]:
        self.ensure_loaded()
        if name not in self.nodes:
            raise MathRefusal(f"No object named {name!r}.")
        doomed = [name] + self.topo_order(self.descendants(name))
        for n in reversed(doomed):
            node = self.nodes.pop(n)
            self.r.delete(node.objects, node=n)
            self.r.forget(n)
        self.r.relayout()
        self.save()
        return doomed

    def clear(self) -> None:
        self.r.send("delete", all_math=True)
        self.nodes.clear()
        self.r.occupancy.clear()
        self.r.labels.clear()
        self._color_index = 0
        self.loaded = True
        self.save()

    def descendants(self, name: str) -> set[str]:
        out, stack = set(), [name]
        while stack:
            cur = stack.pop()
            for n in self.nodes.values():
                if cur in n.deps and n.name not in out:
                    out.add(n.name)
                    stack.append(n.name)
        return out

    def topo_order(self, names) -> list[str]:
        order = [n for n in self.nodes if n in names]  # insertion order is a valid topo order
        return order

    def _commit(self, nodes: list[Node]) -> None:
        backup = {n.name: self.nodes.get(n.name) for n in nodes}
        failing = None
        try:
            for n in nodes:
                failing = n.name
                self.nodes[n.name] = n
                try:
                    n.data = KINDS[n.kind].compute(self, n)
                except MathRefusal:
                    raise
                except (ex.ExpressionError, ValueError, TypeError, ZeroDivisionError) as exc:
                    raise MathRefusal(f"{n.name}: {exc}") from exc
        except MathRefusal as exc:
            for name, old in backup.items():
                if old is None:
                    self.nodes.pop(name, None)
                else:
                    self.nodes[name] = old
            msg = str(exc)
            if len(nodes) > 1 or backup.get(nodes[0].name) is not None:
                who = "" if failing == nodes[0].name else f"{failing} (depends on {nodes[0].name}) would become invalid: "
                msg = f"Change rejected (nothing was modified): {who}{msg}"
            raise MathRefusal(msg) from exc
        for n in nodes:
            old = backup.get(n.name)
            if old is not None:
                self.r.delete(old.objects, node=n.name)
            self.r.forget(n.name)
            n.objects = [] if n.style.get("visible") is False else KINDS[n.kind].draw(self, n)
        self.r.relayout()
        self.save()

    def redraw_all(self) -> None:
        for n in self.nodes.values():
            self.r.delete(n.objects, node=n.name)
        self.r.occupancy.clear()
        self.r.labels.clear()
        for n in self.nodes.values():
            n.objects = [] if n.style.get("visible") is False else KINDS[n.kind].draw(self, n)
        self.r.relayout()

    # ------------------------------------------------------------------ naming / output

    def auto_name(self, kind: str) -> str:
        prefix = {"function": "f", "point": "P", "parameter": "a", "segment": "s", "line": "l", "circle": "c",
                  "polygon": "poly", "angle": "ang", "vector": "v", "text": "txt", "axes": "axes"}.get(kind, kind)
        seq = {"function": list("fghpqr"), "point": list("ABCDEFGHIJKLMNOPQRSTUVW")}.get(kind)
        if seq:
            for s in seq:
                if s not in self.nodes and not ex.is_reserved(s):
                    return s
        i = 1
        while f"{prefix}{i}" in self.nodes:
            i += 1
        return f"{prefix}{i}"

    def describe(self, name: str) -> dict:
        n = self.nodes[name]
        out = {"name": n.name, "kind": n.kind, "definition": n.spec, "style": n.style,
               "depends_on": sorted(n.deps), "dependents": sorted(self.descendants(name)),
               "objects": n.objects}
        out.update(_jsonable(KINDS[n.kind].describe(n)))
        return out

    def listing(self) -> list[dict]:
        return [{"name": n.name, "kind": n.kind, "definition": n.spec, "depends_on": sorted(n.deps),
                 **_jsonable(KINDS[n.kind].describe(n))} for n in self.nodes.values()]

    # ------------------------------------------------------------------ persistence

    def save(self) -> None:
        data = {"version": 1, "theme": self.r.theme.name, "color_index": self._color_index,
                "nodes": [{"name": n.name, "kind": n.kind, "spec": n.spec, "style": n.style}
                          for n in self.nodes.values()]}
        try:
            self.r.send("store", data=data)
        except Exception:  # noqa: BLE001 - persistence is best effort
            pass

    def ensure_loaded(self) -> None:
        """Load the graph stored in the .blend file the first time the server talks to Blender."""
        if self.loaded:
            return
        self.loaded = True
        try:
            stored = self.r.send("store").get("data")
        except Exception:  # noqa: BLE001
            return
        if not stored or not stored.get("nodes"):
            return
        self.r.theme = get_theme(stored.get("theme"))
        self._color_index = stored.get("color_index", 0)
        for item in stored["nodes"]:
            node = Node(item["name"], item["kind"], item["spec"], item.get("style", {}))
            node.deps = KINDS[node.kind].refs(self, node.spec)
            self.nodes[node.name] = node
            try:
                node.data = KINDS[node.kind].compute(self, node)
            except Exception:  # noqa: BLE001 - keep loading; the node stays undefined
                node.data = {}
        self.redraw_all()


def _jsonable(v):
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, sp.Basic):
        return str(v)
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


# --------------------------------------------------------------------------- shared helpers


def nice_latex(e: sp.Expr, digits: int = 6) -> str:
    """Exact LaTeX when it is readable; otherwise a decimal with \\approx."""
    try:
        if e.has(sp.CRootOf, sp.Float) or sp.count_ops(e) > 22 or len(sp.latex(e)) > 60:
            return r"\approx " + _num(e, digits)
    except Exception:  # noqa: BLE001
        pass
    return sp.latex(e)


def exact_or_approx(e: sp.Expr, digits: int = 6) -> tuple[str, bool]:
    s = nice_latex(e, digits)
    return s, not s.startswith(r"\approx")


def _num(e, digits):
    v = float(sp.N(e, 20))
    s = f"{v:.{digits}g}"
    return s


def coord_latex(P) -> str:
    parts = []
    for c in P:
        s = nice_latex(c)
        parts.append(s.replace(r"\approx ", "") if s.startswith(r"\approx") else s)
    approx = any(nice_latex(c).startswith(r"\approx") for c in P)
    body = r"\left(" + ", ".join(parts) + r"\right)"
    return (r"\approx " + body) if approx else body


def fnum(e) -> float:
    return float(sp.N(e, 20))


Compute = Callable[[Construction, Node], dict]
