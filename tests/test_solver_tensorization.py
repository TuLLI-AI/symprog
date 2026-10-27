# type: ignore
import pytest
import torch
from symprog.solvers import Solver

## tensorization is experimental (its warning is tested in test_tensorization.py)
pytestmark = pytest.mark.filterwarnings("ignore::symprog.ExperimentalWarning")

FORMULAS = [
    "Implies(a, b)",
    "And(Or(a, b), Implies(a, c))",
]
WEIGHTS = {"a": 0.9, "b": 0.64, "c": 0.34}


def manual_wmc(solver, weights):
    """Enumerate all models and compute WMC directly."""
    wmc = 0.0
    for model in solver.models():
        score = 1.0
        for var, val in model.items():
            w = weights[var]
            score *= w if val else (1.0 - w)
        wmc += score
    return wmc


# ── correctness: tensorized == inference() == manual enumeration ───────────────

def test_tensor_1():
    """Batch of constant weight vectors: each output matches solver inference."""
    s = Solver()
    s.add(FORMULAS)
    s.add_weights(WEIGHTS)

    mod = s.tensorize()
    a_t = torch.tensor([0.9]).repeat(10)
    b_t = torch.tensor([0.64]).repeat(10)
    c_t = torch.tensor([0.34]).repeat(10)

    batch = mod(a_t, b_t, c_t)
    assert torch.allclose(batch, s.query().inference.repeat(10))


def test_tensorize_single_point_matches_inference():
    """Single weight vector (kwargs): matches solver.inference()."""
    s = Solver()
    s.add(FORMULAS)
    s.add_weights(WEIGHTS)
    mod = s.tensorize()

    result = mod(a=torch.tensor(0.9), b=torch.tensor(0.64), c=torch.tensor(0.34))
    assert result.item() == pytest.approx(s.query().inference.item(), abs=1e-5)


def test_tensorize_single_point_matches_manual_wmc():
    """Single weight vector: matches model-enumeration oracle."""
    s = Solver()
    s.add(FORMULAS)
    s.add_weights(WEIGHTS)
    mod = s.tensorize()

    result = mod(a=torch.tensor(0.9), b=torch.tensor(0.64), c=torch.tensor(0.34))
    assert result.item() == pytest.approx(manual_wmc(s, WEIGHTS), abs=1e-5)


# ── input_symbols ordering ─────────────────────────────────────────────────────

def test_tensorize_input_symbols_alphabetical():
    """input_symbols defaults to alphabetical order of variable names."""
    s = Solver()
    s.add(FORMULAS)
    s.add_weights(WEIGHTS)
    mod = s.tensorize()
    assert mod.input_symbols == ["a", "b", "c"]


def test_tensorize_packed_input_order_follows_input_symbols():
    """Packed tensor x[..., i] maps to input_symbols[i] (alphabetical).

    Uses an asymmetric formula so swapping columns gives a detectably wrong answer.
    Implies(a, b)  =  1 - a*(1-b)  —  not symmetric in a and b.
    """
    s = Solver()
    s.add("Implies(a, b)")
    s.add_weights({"a": 0.9, "b": 0.64})
    mod = s.tensorize()

    assert mod.input_symbols == ["a", "b"]

    # P(Implies(a=0.9, b=0.64)) = 1 - 0.9*(1-0.64) = 0.676
    expected = 1 - 0.9 * (1 - 0.64)
    x_correct = torch.tensor([0.9, 0.64])
    x_swapped = torch.tensor([0.64, 0.9])

    assert mod(x_correct).item() == pytest.approx(expected, abs=1e-5)
    # Swapped gives 1 - 0.64*(1-0.9) = 0.936, not 0.676
    assert mod(x_swapped).item() != pytest.approx(expected, abs=1e-3)


# ── input interface: all three modes agree ─────────────────────────────────────

def test_tensorize_input_modes_agree():
    """Packed tensor, positional args, and kwargs all produce the same result."""
    s = Solver()
    s.add(FORMULAS)
    s.add_weights(WEIGHTS)
    mod = s.tensorize()

    # input_symbols is ['a', 'b', 'c']
    x_packed = torch.tensor([0.9, 0.64, 0.34])
    result_packed = mod(x_packed)
    result_positional = mod(torch.tensor(0.9), torch.tensor(0.64), torch.tensor(0.34))
    result_kwargs = mod(a=torch.tensor(0.9), b=torch.tensor(0.64), c=torch.tensor(0.34))

    assert torch.allclose(result_packed, result_positional, atol=1e-6)
    assert torch.allclose(result_packed, result_kwargs, atol=1e-6)


# ── batching ──────────────────────────────────────────────────────────────────

def test_tensorize_batch_varying_weights():
    """mod(x[B, D]) agrees with B individual mod(x[i]) calls for varied weight vectors."""
    s = Solver()
    s.add("Implies(a, b)")
    s.add_weights({"a": 0.9, "b": 0.64})
    mod = s.tensorize()

    torch.manual_seed(42)
    batch = torch.rand(20, 2)  # 20 different weight vectors
    batch_result = mod(batch)  # shape [20]

    for i in range(20):
        individual = mod(batch[i])
        assert batch_result[i].item() == pytest.approx(individual.item(), abs=1e-5)


# ── gradient flow ─────────────────────────────────────────────────────────────

def test_tensorize_gradient_flow():
    """Backward pass through the compiled module produces non-zero gradients."""
    s = Solver()
    s.add("And(a, b)")
    s.add_weights({"a": 0.5, "b": 0.5})
    mod = s.tensorize()

    x = torch.tensor([0.5, 0.5], requires_grad=True)
    out = mod(x)
    out.backward()

    assert x.grad is not None
    assert not torch.all(x.grad == 0)


# ── simplification equivalence ────────────────────────────────────────────────

def test_tensorize_simplify_matches_base():
    """simplify=True gives the same numerical result as the default."""
    s = Solver()
    s.add(FORMULAS)
    s.add_weights(WEIGHTS)

    x = torch.tensor([0.9, 0.64, 0.34])
    assert torch.allclose(s.tensorize()(x), s.tensorize(simplify=True)(x), atol=1e-5)


def test_tensorize_horner_matches_base():
    """try_horner=True gives the same numerical result as the default."""
    s = Solver()
    s.add(FORMULAS)
    s.add_weights(WEIGHTS)

    x = torch.tensor([0.9, 0.64, 0.34])
    assert torch.allclose(s.tensorize()(x), s.tensorize(try_horner=True)(x), atol=1e-5)


# ── edge cases ────────────────────────────────────────────────────────────────

def test_tensorize_single_variable():
    """Single-variable formula: output equals the variable's weight."""
    s = Solver()
    s.add("a")
    s.add_weights({"a": 0.7})
    mod = s.tensorize()

    assert mod.input_symbols == ["a"]
    assert mod(a=torch.tensor(0.7)).item() == pytest.approx(0.7, abs=1e-5)


def test_tensorize_tautology_is_constant_one():
    """Tautology reduces to the constant 1: no input symbols, any tensor input works.

    The polynomial for Or(a, Not(a)) reduces to 1 (the SDD collapses to TRUE),
    so free_symbols is empty and the compiled module ignores its input entirely.
    Calling with a named kwarg raises because there are no inputs to name.
    """
    s = Solver()
    s.add("Or(a, Not(a))")
    s.add_weights({"a": 0.3})
    mod = s.tensorize()

    assert mod.input_symbols == []
    assert mod(torch.empty(0)).item() == pytest.approx(1.0, abs=1e-5)

    # Extra kwargs warn rather than raise: the variable was folded out
    with pytest.warns(UserWarning, match="a"):
        result = mod(a=torch.tensor(0.3))
    assert result.item() == pytest.approx(1.0, abs=1e-5)


def test_tensorize_contradiction_is_constant_zero():
    """Contradiction reduces to the constant 0: same semantics as tautology."""
    s = Solver()
    s.add("And(a, Not(a))")
    s.add_weights({"a": 0.5})
    mod = s.tensorize()

    assert mod.input_symbols == []
    assert mod(torch.empty(0)).item() == pytest.approx(0.0, abs=1e-5)


# ── evidence-based constant folding (already works) ────────────────────────────

def test_tensorize_evidence_true_folds_variable():
    """evidence='b' (b=True) eliminates b from input_symbols via circuit conditioning."""
    s = Solver()
    s.add("And(a, b)")
    s.add_weights({"a": 0.9, "b": 0.64})
    mod = s.tensorize(evidence="b")
    assert "b" not in mod.input_symbols
    assert "a" in mod.input_symbols
    # And(a, True) = a
    assert mod(a=torch.tensor(0.9)).item() == pytest.approx(0.9, abs=1e-5)


def test_tensorize_evidence_false_folds_variable():
    """evidence='~b' (b=False) eliminates b from input_symbols via circuit conditioning.

    ~var is the only supported negative literal format; -var and Not(var) are silently ignored.
    """
    s = Solver()
    s.add("And(a, b)")
    s.add_weights({"a": 0.9, "b": 0.64})
    mod = s.tensorize(evidence="~b")
    assert "b" not in mod.input_symbols
    # And(a, False) = False → constant 0, no inputs
    assert mod.input_symbols == []
    assert mod(torch.empty(0)).item() == pytest.approx(0.0, abs=1e-5)


def test_tensorize_evidence_implies_folded():
    """evidence='a' on Implies(a,b): eliminates a, result is just b."""
    s = Solver()
    s.add("Implies(a, b)")
    s.add_weights({"a": 0.9, "b": 0.64})
    mod = s.tensorize(evidence="a")
    assert "a" not in mod.input_symbols
    assert mod.input_symbols == ["b"]
    assert mod(b=torch.tensor(0.64)).item() == pytest.approx(0.64, abs=1e-5)


# ── passing folded-out variables as kwargs ────────────────────────────────────

def test_tensorize_extra_kwarg_warns_not_raises():
    """Passing a kwarg for a variable that was folded out should warn, not raise.

    A natural calling convention is to always pass all variable weights; the module
    should tolerate extras that were eliminated via evidence or simplification.
    """
    s = Solver()
    s.add("Implies(a, b)")
    s.add_weights({"a": 0.9, "b": 0.64})
    mod = s.tensorize(evidence="a")
    assert "a" not in mod.input_symbols

    with pytest.warns(UserWarning, match="a"):
        result = mod(a=torch.tensor(0.9), b=torch.tensor(0.64))

    # Result should equal the no-extra-kwarg version
    expected = mod(b=torch.tensor(0.64))
    assert torch.allclose(result, expected)


def test_tensorize_extra_kwarg_constant_module_warns():
    """Extra kwargs on a constant module (all vars folded) warn and return the constant."""
    s = Solver()
    s.add("Or(a, Not(a))")
    s.add_weights({"a": 0.5})
    mod = s.tensorize()
    assert mod.input_symbols == []

    with pytest.warns(UserWarning, match="a"):
        result = mod(a=torch.tensor(0.5))

    assert result.item() == pytest.approx(1.0, abs=1e-5)


# NOTE: a fixed_weights= parameter (polynomial-level sympy.subs() for non-0/1 priors)
# was considered but not added. The two cases that motivated it are already handled:
#   - aux vars (_F, structural encoding variables): eliminated by wmc_interface
#   - Boolean observations: use evidence= above (circuit-level conditioning)
# For a variable with a fixed float prior, just pass it as a constant tensor input;
# the module evaluates correctly with no added API surface.
