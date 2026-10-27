"""EXPERIMENTAL: tensorization of SDDs with KLay (https://github.com/ML-KULeuven/klay),
which evaluates circuits layer by layer with batched tensor operations.

Requires `pip install "symprog[klay]"` (Python >= 3.11).
"""
import torch
import torch.nn as nn

from .tensorization import SymbolicFxModule, SymbolicSignature

__all__ = ["klay_module", "KLAY_MODES"]

## inference modes supported with KLay: mode -> (KLay semiring, log-space inputs, smoothing)
KLAY_MODES = {
    "Prob": ("real", False, False),
    "LogProb": ("log", True, False),
    "Viterbi": ("mpe", False, True),
    "LogViterbi": ("tropical", True, True),
    "Fuzzy": ("godel", False, True),
}

class _KlayCircuit(nn.Module):
    """Maps packed inputs `x[..., D]` (probabilities, ordered as the module's input symbols)
    to the positive and negative literal weights of a KLay circuit module (ordered by SDD
    variable id, as `[batch, num_vars]`), and evaluates the circuit. Variables that are not
    inputs (e.g., auxiliary variables) get weights 1 and 0, so they do not affect the result.
    """
    def __init__(self, circuit: nn.Module, input_ids: list[int], num_vars: int, log_space: bool) -> None:
        super().__init__()
        self.circuit = circuit
        self.num_vars = num_vars
        self.log_space = log_space
        self.register_buffer("input_ids", torch.tensor(input_ids, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_shape = x.shape[:-1]
        flat = x.reshape(-1, x.shape[-1])
        pos = flat.new_ones(flat.shape[0], self.num_vars).index_copy(-1, self.input_ids, flat)
        if self.log_space:
            pos, neg = torch.log(pos), torch.log1p(-pos)
        else:
            neg = 1 - pos
        out = self.circuit(pos, neg)
        return out.reshape(batch_shape)

def _smooth_circuit(klay, mapper, formula, root_vars: set[int]):
    """Builds a *smooth* KLay circuit from an SDD (needed for max-based semirings, where
    a variable missing from a branch must contribute max(w, 1 - w) rather than 1). Each
    element of a decision node is smoothed over the variables of its vtree, and the root
    over `root_vars` (SDD variable ids)."""
    circuit = klay.Circuit()
    vtree_vars: dict[int, frozenset[int]] = {}
    converted: dict[int, tuple] = {}
    either: dict[int, object] = {}

    def variables(vtree) -> frozenset[int]:
        key = vtree.position()
        if key not in vtree_vars:
            if vtree.is_leaf():
                vtree_vars[key] = frozenset([vtree.var()])
            else:
                vtree_vars[key] = variables(vtree.left()) | variables(vtree.right())
        return vtree_vars[key]

    def smooth(node, covered, scope):
        missing = sorted(scope - covered)
        if not missing:
            return node
        for v in missing:
            if v not in either:
                either[v] = circuit.or_node([circuit.literal_node(v), circuit.literal_node(-v)])
        return circuit.and_node([node] + [either[v] for v in missing])

    def convert(node):
        """Returns the KLay node for an SDD node, and the variables it covers."""
        if node.id in converted:
            return converted[node.id]
        if node.is_true():
            result = (circuit.true_node(), frozenset())
        elif node.is_false():
            result = (circuit.false_node(), frozenset())
        elif node.is_literal():
            result = (circuit.literal_node(node.literal), frozenset([abs(node.literal)]))
        else:
            vtree = node.vtree()
            left, right = variables(vtree.left()), variables(vtree.right())
            children = []
            for prime, sub in node.elements():
                if prime.is_false() or sub.is_false():
                    continue
                prime_node, prime_vars = convert(prime)
                sub_node, sub_vars = convert(sub)
                children.append(circuit.and_node([
                    smooth(prime_node, prime_vars, left), smooth(sub_node, sub_vars, right)
                ]))
            node_ptr = circuit.or_node(children) if children else circuit.false_node()
            result = (node_ptr, left | right)
        converted[node.id] = result
        return result

    if formula.is_false():
        circuit.set_root(circuit.false_node())
    else:
        root, covered = convert(formula)
        circuit.set_root(smooth(root, covered, frozenset(root_vars)))
    return circuit

def klay_module(
    mapper, formula, input_symbols: list[str], mode: str = "Prob", compile: bool = False
) -> SymbolicFxModule:
    """Builds a KLay module computing inference `mode` (see `KLAY_MODES`) over an SDD
    `formula` from the SDD solver interface `mapper`. The module has the same calling
    conventions as modules from `Solver.tensorize`: `mod(x)` with `x[..., D]` ordered as
    `input_symbols`, `mod(a, b, ...)` or `mod(a=..., b=...)`, where inputs are probabilities.

    Arguments:
        mapper: The `SddMapper` that built `formula`.
        formula: The SDD node to compile.
        input_symbols: The names of the variables that are inputs of the module.
        mode: The inference mode.
        compile: Whether KLay should wrap its module with `torch.compile`.

    Raises:
        ImportError: If KLay is not installed.
        ValueError: If the mode is not supported.
    """
    try:
        import klay
        from klay.torch.layers import LogSumLayer, MaxLayer
    except ImportError as e:
        raise ImportError(
            "The KLay backend requires klay: pip install 'symprog[klay]' (Python >= 3.11)"
        ) from e
    if mode not in KLAY_MODES:
        raise ValueError(f"Mode '{mode}' is not supported by KLay (supported: {', '.join(KLAY_MODES)}).")
    semiring, log_space, smoothing = KLAY_MODES[mode]

    input_ids = [mapper[name].literal for name in input_symbols]
    if smoothing:
        circuit = _smooth_circuit(klay, mapper, formula, set(input_ids))
    else:
        circuit = klay.Circuit()
        circuit.add_sdd(formula)

    if semiring == "tropical":
        ## max-plus semiring: the log semiring with its log-sum-exp layers replaced by max layers
        module = circuit.to_torch_module(semiring="log", compile=False)
        for i, layer in enumerate(module.layers):
            if isinstance(layer, LogSumLayer):
                module.layers[i] = MaxLayer(layer.ix_in, layer.offsets, layer.ix_out, layer._eps)
        if compile:
            module = torch.compile(module)
    else:
        module = circuit.to_torch_module(semiring=semiring, compile=compile)

    signature = SymbolicSignature(
        input_symbols=list(input_symbols),
        sym2idx={name: i for i, name in enumerate(input_symbols)},
    )
    adapter = _KlayCircuit(module, [i - 1 for i in input_ids], mapper.manager.var_count(), log_space)
    return SymbolicFxModule(adapter, signature, keep_last_dim=False)
