import sympy 
import torch
import warnings
import torch.fx as fx
import torch.nn as nn
from typing import Any, Callable, Mapping, Sequence, Literal, Generator, Any
from dataclasses import dataclass

def _reduce_tree(g: fx.Graph, op, args: list[fx.Node]) -> fx.Node:
    if not args:
        raise ValueError("empty")
    while len(args) > 1:
        nxt = []
        it = iter(args)
        for a in it:
            b = next(it, None)
            if b is None:
                nxt.append(a)
            else:
                nxt.append(g.call_function(op, (a, b)))
        args = nxt
    return args[0]

def add_n_fx(g: fx.Graph, *args: fx.Node) -> fx.Node:
    return _reduce_tree(g, torch.add, list(args))

def mul_n_fx(g: fx.Graph, *args: fx.Node) -> fx.Node:
    return _reduce_tree(g, torch.mul, list(args))

def max_n_fx(g: fx.Graph, *args: fx.Node) -> fx.Node:
    return _reduce_tree(g, torch.maximum, list(args))

def min_n_fx(g: fx.Graph, *args: fx.Node) -> fx.Node:
    return _reduce_tree(g, torch.minimum, list(args))

GLOBAL_FUNC_LOOKUP: dict[type | Callable[..., Any], Callable[..., Any]] = {
    sympy.sin: torch.sin,
    sympy.cos: torch.cos,
    sympy.tan: torch.tan,
    sympy.asin: torch.asin,
    sympy.acos: torch.acos,
    sympy.atan: torch.atan,
    sympy.atan2: torch.atan2,
    sympy.exp: torch.exp,
    sympy.log: torch.log,
    sympy.sqrt: torch.sqrt,
    sympy.Abs: torch.abs,
    sympy.Pow: torch.pow,
    sympy.StrictGreaterThan: torch.gt,
    sympy.StrictLessThan: torch.lt,
    sympy.LessThan: torch.le,
    sympy.GreaterThan: torch.ge,
    sympy.And: torch.logical_and,
    sympy.Or: torch.logical_or,
    sympy.Not: torch.logical_not,
    sympy.Max: torch.max,
    sympy.Min: torch.min,
    sympy.MatAdd: torch.add,
    sympy.HadamardProduct: torch.mul,
    sympy.Trace: torch.trace,
    sympy.div: torch.div,
}


class _ConstRoot(nn.Module):
    """Holds constants as buffers/parameters so FX can refer to them via get_attr."""
    def __init__(self):
        super().__init__()


def _normalize_expressions(exprs) -> tuple:
    try:
        import symengine as _se
        if isinstance(exprs, _se.Expr):
            return (exprs,)
    except ImportError:
        pass
    if isinstance(exprs, sympy.Expr):
        return (exprs,)
    return tuple(exprs)


def _normalize_input_order(
    input_order: Sequence[str | sympy.Symbol] | None,
    required_syms: set[sympy.Symbol],
) -> list[sympy.Symbol]:
    if input_order is None:
        return sorted(required_syms, key=lambda s: s.name)

    ordered: list[sympy.Symbol] = []
    for item in input_order:
        if isinstance(item, sympy.Symbol):
            ordered.append(item)
        elif isinstance(item, str):
            ordered.append(sympy.Symbol(item))
        else:
            raise TypeError(f"input_order items must be str or sympy.Symbol, got {type(item)}")

    # validate duplicates by name
    names = [s.name for s in ordered]
    if len(names) != len(set(names)):
        raise ValueError(f"input_order has duplicates: {names}")

    ordered_set = set(ordered)
    missing = required_syms - ordered_set
    extra = ordered_set - required_syms
    if missing:
        raise ValueError(
            f"input_order is missing required symbols: "
            f"{[s.name for s in sorted(missing, key=lambda s: s.name)]}"
        )
    if extra:
        raise ValueError(
            f"input_order contains symbols not used in expressions: "
            f"{[s.name for s in sorted(extra, key=lambda s: s.name)]}"
        )

    return ordered


def _as_tensor_like(v: Any) -> torch.Tensor:
    if isinstance(v, torch.Tensor):
        return v
    return torch.as_tensor(v)

def _try_horner(e: sympy.Expr, ordered_inputs: list[sympy.Symbol]) -> sympy.Expr:
    # Horner is only “safe”/meaningful when e is polynomial in the inputs.
    # If it’s not, keep it as-is.
    try:
        if e.is_polynomial(*ordered_inputs):
            # horner() often works without expand(), which avoids blow-ups.
            return sympy.horner(e, *ordered_inputs)
    except Exception:
        pass
    return e

@dataclass(frozen=True)
class SymbolicSignature:
    input_symbols: list[str]
    sym2idx: dict[str, int]

class SymbolicFxModule(nn.Module):
    """
    Wrapper around a packed-input FX GraphModule.

    Accepts:
      - mod(x) where x[..., D]
      - mod(a, b, c) positional in input_order
      - mod(a=..., b=..., c=...) kwargs
    """
    def __init__(self, gm_packed: fx.GraphModule, sig: SymbolicSignature, *, keep_last_dim: bool):
        super().__init__()
        self.gm = gm_packed
        self.sig = sig
        self.keep_last_dim = keep_last_dim

    def compile(self, **kwargs):
        self.gm = torch.compile(self.gm, **kwargs)
        return self

    @property
    def input_symbols(self) -> list[str]:
        return self.sig.input_symbols

    @property
    def sym2idx(self) -> dict[str, int]:
        return self.sig.sym2idx

    def _pack_from_positional(self, args: tuple[Any, ...]) -> torch.Tensor:
        D = len(self.sig.input_symbols)
        if len(args) != D:
            raise TypeError(f"Expected {D} positional inputs ({self.sig.input_symbols}), got {len(args)}")

        if D == 0:
            return torch.empty(0)

        ts = [_as_tensor_like(a) for a in args]

        # Choose device/dtype from first tensor, but allow others to broadcast/cast naturally
        ref = ts[0]
        ts = [t.to(device=ref.device) if t.device != ref.device else t for t in ts]

        # Broadcast all to common shape (excluding the final stacking dim)
        bt = torch.broadcast_tensors(*ts)
        x = torch.stack(bt, dim=-1)  # [..., D]
        return x

    def _pack_from_kwargs(self, kwargs: dict[str, Any]) -> torch.Tensor:
        missing = [s for s in self.sig.input_symbols if s not in kwargs]
        extra = [k for k in kwargs.keys() if k not in self.sig.sym2idx]
        if missing:
            raise TypeError(f"Missing inputs: {missing}")
        if extra:
            warnings.warn(
                f"Inputs {extra} were ignored: not in input_symbols "
                f"{self.sig.input_symbols}. They may have been eliminated "
                f"via evidence conditioning or simplification.",
                UserWarning,
                stacklevel=3,
            )

        ordered = tuple(kwargs[name] for name in self.sig.input_symbols)
        return self._pack_from_positional(ordered)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        # Case 1: mod(x) with a single tensor argument and no kwargs
        if len(args) == 1 and not kwargs and isinstance(args[0], torch.Tensor):
            x = args[0]
            return self.gm(x)

        # Case 2: kwargs style
        if kwargs:
            if args:
                raise TypeError("Use either positional args OR keyword args OR a single packed tensor, not a mix.")
            x = self._pack_from_kwargs(kwargs)
            return self.gm(x)

        # Case 3: positional style (a, b, c, ...)
        x = self._pack_from_positional(args)
        return self.gm(x)
    
def symbolic_to_fx(
    expressions: sympy.Expr | Sequence[sympy.Expr],
    *,
    input_order: Sequence[str | sympy.Symbol] | None = None,
    extra_funcs: Mapping[type | Callable[..., Any], Callable[..., Any]] | None = None,
    do_cse: bool = True,
    trainable_floats: bool = False,
    float_dtype: torch.dtype | None = None,
    keep_last_dim: bool = False,
    try_horner: bool = False,
) -> SymbolicFxModule:
    """
    Build a module that supports BOTH:
      - packed input: mod(x) where x[..., D]
      - unpacked input: mod(a, b, c) or mod(a=..., b=..., c=...)

    input_order controls the mapping between symbols and x's last dimension.
    If None, defaults to alphabetical order by symbol name.

    keep_last_dim:
      - False (default): each symbol reads x[..., i] (drops last dim)
      - True: reads x[..., i:i+1] (keeps singleton last dim)
    """
    if float_dtype is None:
        float_dtype = torch.get_default_dtype()

    func_lookup = dict(GLOBAL_FUNC_LOOKUP)
    if extra_funcs:
        func_lookup.update(extra_funcs)

    expressions_t = _normalize_expressions(expressions)

    # Detect symengine expressions. Use symengine's faster CSE, then convert
    # only the compact CSE output (not the full tree) to sympy for FX building.
    _is_symengine = False
    try:
        import symengine as _se
        if any(isinstance(e, _se.Expr) for e in expressions_t):
            _is_symengine = True
    except ImportError:
        pass

    # .free_symbols works on both sympy and symengine; normalise to sympy.Symbol by name
    raw_free = set().union(*(e.free_symbols for e in expressions_t))
    required_syms: set[sympy.Symbol] = {sympy.Symbol(str(s)) for s in raw_free}
    ordered_inputs = _normalize_input_order(input_order, required_syms)

    if try_horner:
        expressions_t = tuple(_try_horner(sympy.sympify(e), ordered_inputs) for e in expressions_t)
        _is_symengine = False

    input_symbols = [s.name for s in ordered_inputs]
    sym2idx = {s.name: i for i, s in enumerate(ordered_inputs)}
    sig = SymbolicSignature(input_symbols=input_symbols, sym2idx=sym2idx)

    # CSE: use symengine's C++ CSE when available (30× faster), then convert
    # only the compact output to sympy for the FX graph builder.
    if do_cse:
        if _is_symengine:
            se_repls, se_reduced = _se.cse(list(expressions_t))
            repls   = [(sympy.Symbol(str(s)), sympy.sympify(e)) for s, e in se_repls]
            reduced = [sympy.sympify(e) for e in se_reduced]
        else:
            repls, reduced = sympy.cse(expressions_t, optimizations=[])
    else:
        repls, reduced = [], [sympy.sympify(e) if _is_symengine else e for e in expressions_t]

    root = _ConstRoot()
    g = fx.Graph()

    # single packed input
    x = g.placeholder("x")  # expects x[..., D]

    # Memo: SymPy expr -> fx.Node
    memo: dict[sympy.Basic, fx.Node] = {}

    # CSE temps: Symbol -> fx.Node
    temps: dict[sympy.Symbol, fx.Node] = {}

    const_count = 0

    def const_node(val: torch.Tensor, trainable: bool) -> fx.Node:
        nonlocal const_count
        name = f"c{const_count}"
        const_count += 1
        if trainable:
            setattr(root, name, nn.Parameter(val))
        else:
            root.register_buffer(name, val)
        return g.get_attr(name)

    def symbol_from_x(sym: sympy.Symbol) -> fx.Node:
        i = sym2idx[sym.name]
        dim = -1

        if keep_last_dim:
            # x[..., i:i+1]
            # torch.narrow(x, dim, start, length) is a torch op (GPU-friendly)
            return g.call_function(torch.narrow, (x, dim, i, 1))
        else:
            # x[..., i]
            # torch.select(x, dim, index) is also a torch op
            return g.call_function(torch.select, (x, dim, i))
        # if keep_last_dim:
        #     # x[..., i:i+1]  -> preserves a trailing singleton dimension
        #     return g.call_function(operator.getitem, (x, (Ellipsis, slice(i, i + 1)))) # type: ignore
        # else:
        #     # x[..., i] -> drops the last dimension
        #     return g.call_function(operator.getitem, (x, (Ellipsis, i))) # type: ignore

    def build(e: sympy.Expr) -> fx.Node:
        if e in memo:
            return memo[e]

        # CSE temporary symbol?
        if isinstance(e, sympy.Symbol) and e in temps:
            return temps[e]

        # Input symbol
        if isinstance(e, sympy.Symbol):
            n = symbol_from_x(e)
            memo[e] = n
            return n

        # Numeric constant
        if e.is_number:
            if e == sympy.I:
                n = const_node(torch.tensor(1j), trainable=False)
            elif isinstance(e, sympy.Integer):
                n = const_node(torch.tensor(int(e), dtype=float_dtype), trainable=False)
            elif isinstance(e, sympy.Rational):
                n = const_node(torch.tensor(float(e), dtype=float_dtype), trainable=False)
            elif isinstance(e, sympy.Float):
                n = const_node(torch.tensor(float(e), dtype=float_dtype), trainable=trainable_floats)
            elif isinstance(e, sympy.NumberSymbol):
                n = const_node(torch.tensor(float(e.evalf()), dtype=float_dtype), trainable=False)  # type: ignore
            elif e in (sympy.oo, -sympy.oo):
                ## e.g., log(0) when evidence makes the formula unsatisfiable
                n = const_node(torch.tensor(float(e), dtype=float_dtype), trainable=False)
            elif e in (sympy.zoo, sympy.nan):
                ## undefined values (e.g., the entropy of an unsatisfiable formula)
                n = const_node(torch.tensor(float("nan"), dtype=float_dtype), trainable=False)
            else:
                raise TypeError(f"Unsupported numeric type: {type(e)} ({e})")

            memo[e] = n
            return n

        # Operations
        if e.func is sympy.Add:
            args = [build(a) for a in e.args]  # type: ignore
            n = add_n_fx(g, *args)
        elif e.func is sympy.Mul:
            args = [build(a) for a in e.args]  # type: ignore
            n = mul_n_fx(g, *args)
        elif e.func in (sympy.Max, sympy.Min):
            ## n-ary max/min (torch.max/min only take two tensors)
            args = [build(a) for a in e.args]  # type: ignore
            n = (max_n_fx if e.func is sympy.Max else min_n_fx)(g, *args)
        else:
            torch_fn = func_lookup.get(e.func)
            if torch_fn is None:
                raise KeyError(f"No torch mapping for SymPy func {e.func} (expr={e})")
            args = tuple(build(a) for a in e.args)  # type: ignore
            n = g.call_function(torch_fn, args)

        memo[e] = n
        return n

    # Emit CSE assignments (temps)
    for sym, rhs in repls:
        assert isinstance(sym, sympy.Symbol)
        temps[sym] = build(rhs)

    # Emit outputs
    out_nodes = [build(e) for e in reduced]  # type: ignore
    if len(out_nodes) == 1:
        g.output(out_nodes[0])
    else:
        bt = g.call_function(torch.broadcast_tensors, tuple(out_nodes))
        out = g.call_function(torch.stack, (bt,), {"dim": -1})
        g.output(out)

    gm_packed = fx.GraphModule(root, g)

    # optional: expose signature on gm too (handy for debugging)
    gm_packed.input_symbols = input_symbols  # type: ignore[attr-defined]
    gm_packed.sym2idx = sym2idx  # type: ignore[attr-defined]

    return SymbolicFxModule(gm_packed, sig, keep_last_dim=keep_last_dim)