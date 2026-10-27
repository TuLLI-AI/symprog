# type: ignore
"""Tests for solver backends (sdd, bdd, z3, sympy, pysat), the experimental JAX weight
backend, formula parsing, and solver-level regressions.
"""
import importlib.util
import warnings

import pytest

from symprog import Solver

FORMULAS = ["Implies(a, b)", "And(Or(a, b), Implies(a, c))"]
WEIGHTS = {"a": 0.9, "b": 0.64, "c": 0.34}
MODELS = [
    {"a": False, "b": True, "c": False},
    {"a": False, "b": True, "c": True},
    {"a": True, "b": True, "c": True},
]
SOLVER_BACKENDS = ["sdd", "bdd", "z3", "sympy", "pysat"]

def build(stype="sdd", formulas=FORMULAS, weights=WEIGHTS, container_type="pytorch"):
    s = Solver(_stype=stype)
    s.add(formulas)
    if weights is not None:
        s.add_weights(weights, container_type=container_type)
    return s

def _sorted_models(models):
    return sorted(sorted(m.items()) for m in models)

def _is_sat(out):
    return str(out.inference) in {"yes", "sat"}

## solver backends

@pytest.mark.parametrize("stype", SOLVER_BACKENDS)
def test_backend_models(stype):
    s = build(stype)
    assert _sorted_models(s.models()) == _sorted_models(MODELS)

@pytest.mark.parametrize("stype", SOLVER_BACKENDS)
def test_backend_models_unsatisfiable(stype):
    s = build(stype, formulas=["And(a, Not(a))"], weights=None)
    assert list(s.models()) == []

@pytest.mark.parametrize("stype", SOLVER_BACKENDS)
def test_backend_fast_count(stype):
    s = build(stype)
    assert int(s.query(mode="Count", fast=True).inference) == len(MODELS)

@pytest.mark.parametrize("stype", SOLVER_BACKENDS)
def test_backend_sat(stype):
    s = build(stype)
    assert _is_sat(s.query(mode="SAT"))
    s.add(["Not(b)"])
    assert not _is_sat(s.query(mode="SAT"))

@pytest.mark.parametrize("stype", ["bdd", "z3", "sympy", "pysat"])
def test_backend_without_wmc_raises(stype):
    s = build(stype)
    with pytest.raises(NotImplementedError):
        s.query(mode="Prob")

def test_sdd_models_include_irrelevant_variables():
    s = build(formulas=["c", "Not(e)", "Or(Or(Not(e), c), Not(f))"], weights=None)
    models = _sorted_models(s.models())
    assert models == _sorted_models([
        {"c": True, "e": False, "f": False},
        {"c": True, "e": False, "f": True},
    ])

def test_sdd_from_cnf_hides_tseitin_variables():
    s = Solver.from_cnf(formulas=FORMULAS, var_weights=WEIGHTS)
    assert _sorted_models(s.models()) == _sorted_models(MODELS)
    assert int(s.query(mode="Count", fast=True).inference) == len(MODELS)
    assert s.query(mode="Count").inference.item() == len(MODELS)
    assert s.query(mode="Prob").inference.item() == pytest.approx(0.25984, abs=1e-5)

## experimental JAX backend

HAS_JAX = importlib.util.find_spec("jax") is not None

@pytest.mark.skipif(HAS_JAX, reason="jax is installed")
def test_jax_backend_requires_jax():
    with pytest.raises(ImportError, match=r"symprog\[jax\]"):
        build(container_type="jax")

@pytest.mark.skipif(not HAS_JAX, reason="jax not installed")
@pytest.mark.parametrize("mode", ["Prob", "LogProb", "Count", "Viterbi", "LogViterbi", "Fuzzy", "Entropy"])
def test_jax_forward_inference_matches_pytorch(mode):
    with pytest.warns(UserWarning, match="experimental"):
        s_jax = build(container_type="jax")
    s_torch = build()
    assert float(s_jax.query(mode=mode).inference) == pytest.approx(
        s_torch.query(mode=mode).inference.item(), abs=1e-5
    )

@pytest.mark.skipif(not HAS_JAX, reason="jax not installed")
def test_jax_gradient_matches_pytorch():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = build(container_type="jax")
    assert [float(g) for g in s.gradient(mode="Prob", key="weights")] == pytest.approx(
        build().gradient(mode="Prob", key="weights").tolist(), abs=1e-5
    )

## formula parsing

@pytest.mark.parametrize("formula", [
    "And(Or(a, b), Implies(a, c), Implies(a, b))",
    "Or(a, b) & Implies(a, c) & Implies(a, b)",
    "(a | b) & (~a | c) & (~a | b)",
    "And(Or(a, b), Or(Not(a), c), Or(~a, b))",
])
def test_formula_syntax(formula):
    s = build(formulas=[formula])
    assert _sorted_models(s.models()) == _sorted_models(MODELS)

def test_xor():
    s = build(formulas=["Xor(a, b)"], weights={"a": 0.9, "b": 0.64})
    assert s.query().inference.item() == pytest.approx(0.9 * 0.36 + 0.1 * 0.64, abs=1e-5)

def test_docstring_conditional_example():
    s = Solver()
    s.add("And(Or(a, b), Implies(c, d))")
    s.add_weights({"a": 0.4, "b": 0.45, "c": 0.76, "d": 0.91})
    assert s.query(mode="Prob").inference.item() == pytest.approx(0.6242, abs=1e-4)
    s.add_conditional("Or(a, ~d)")
    ## P(F | G) by enumeration = 0.83364
    assert s.query(mode="Prob").inference.item() == pytest.approx(0.83364, abs=1e-4)

## solver regressions

def test_add_conditional_after_fast_count():
    s = build()
    s.query(mode="Count", fast=True)
    s.add_conditional("b")
    assert s.query().inference.item() == pytest.approx(0.25984 / 0.64, abs=1e-5)

def test_unknown_evidence_or_query_variable_raises():
    s = build()
    with pytest.raises(ValueError, match="Unknown variable"):
        s.query(evidence="z")
    with pytest.raises(ValueError, match="Unknown variable"):
        s.query(query="~z")

def test_evidence_set_not_mutated():
    s = build()
    s.add_conditional("b")
    evidence = {"a"}
    s.query(evidence=evidence)
    assert evidence == {"a"}

def test_sat_with_evidence():
    s = build()
    assert _is_sat(s.query(mode="SAT", evidence="a"))
    assert not _is_sat(s.query(mode="SAT", evidence="~b"))

def test_remove_conditional_restores_inference():
    s = build()
    s.add_conditional("Or(a, c)")
    s.query()
    s.remove_conditional()
    assert s.query().inference.item() == pytest.approx(0.25984, abs=1e-5)
    assert s.query(mode="Count").inference.item() == len(MODELS)
    assert int(s.query(mode="Count", fast=True).inference) == len(MODELS)
    assert _sorted_models(s.models()) == _sorted_models(MODELS)

def test_models_hide_auxiliary_variables():
    s = build()
    s.add_conditional("Or(a, c)")
    assert all("_F" not in m for m in s.models())

def test_solver_weights_is_weight_manager():
    s = build()
    assert s.weight_names == ["a", "b", "c"]
    assert s.weights.weights.tolist() == pytest.approx([0.9, 0.64, 0.34])
    assert repr(s.weights) == "PytorchWeights(a=0.9, b=0.64, c=0.34)"

def test_from_cnf_keeps_variables_of_tautological_clauses():
    """The CNF conversion drops tautological clauses such as `Or(a, Not(a))`, but `a` 
    remains a (free) variable of the formula."""
    s = Solver.from_cnf(formulas=["Or(a, Not(a))", "Or(b, c)"], var_weights={"a": 0.4, "b": 0.25, "c": 0.8})
    assert len(list(s.models())) == 6
    assert s.query(mode="Count").inference.item() == 6
    assert int(s.query(mode="Count", fast=True).inference) == 6
    ## a's distribution is unconstrained, so it adds its entropy H(0.4) to that of (b, c)
    import math
    h_a = -(0.4 * math.log(0.4) + 0.6 * math.log(0.6))
    p_bc = [0.25 * 0.8, 0.25 * 0.2, 0.75 * 0.8]
    z = sum(p_bc)
    h_bc = -sum(p / z * math.log(p / z) for p in p_bc)
    assert s.query(mode="Entropy").inference.item() == pytest.approx(h_a + h_bc, abs=1e-5)
