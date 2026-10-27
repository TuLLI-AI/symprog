# type: ignore
"""Tests for weight gradients.

Numerical gradients (PyTorch autograd, and JAX when installed) are checked against
central finite differences of the brute-force reference, and symbolic gradients (sympy,
symengine) are checked against PyTorch after substituting the weight values, following
the usage in the notebooks.
"""
import importlib.util
import math

import pytest
import sympy as sp
import torch

from symprog import Parser, Solver
from symprog.solvers.factory import BuildWeights

from .reference import CASES, _models, _model_prob, _sympy_formula, build, reference

## differentiable modes; Viterbi/LogViterbi/Fuzzy are piecewise smooth, and the test
## weights are chosen away from their kinks
SMOOTH_MODES = ["Prob", "LogProb", "Entropy"]
PIECEWISE_MODES = ["Viterbi", "LogViterbi", "Fuzzy"]
MODES = SMOOTH_MODES + PIECEWISE_MODES

GRAD_CASES = ["implies", "xor_chain", "irrelevant_var", "tautology_clause"]

HAS_JAX = importlib.util.find_spec("jax") is not None
NUMERIC_BACKENDS = ["pytorch"] + (["jax"] if HAS_JAX else [])

pytestmark = pytest.mark.filterwarnings("ignore:The JAX backend is experimental")

def _evidence_options(formulas, weights):
    """No evidence, and every single (satisfiable) evidence literal."""
    options = [None]
    for var in sorted(weights):
        for polarity in (True, False):
            if _models(formulas, sorted(weights), {var: polarity}):
                options.append(var if polarity else f"~{var}")
    return options

def _evidence_dict(literal):
    if literal is None:
        return None
    return {literal.lstrip("~"): not literal.startswith("~")}

def finite_differences(mode, formulas, weights, evidence=None, eps=1e-5):
    grads = []
    for var in sorted(weights):
        up = dict(weights, **{var: weights[var] + eps})
        down = dict(weights, **{var: weights[var] - eps})
        grads.append(
            (reference(mode, formulas, up, evidence) - reference(mode, formulas, down, evidence)) / (2 * eps)
        )
    return grads

def substitute(gradient, weights):
    """Substitutes weight values into a symbolic gradient (sympy Matrix or list of
    symengine expressions), as done in the notebooks."""
    values = []
    for expr in gradient:
        expr = sp.sympify(expr)
        values.append(float(expr.subs({x: weights[str(x)] for x in expr.free_symbols})))
    return values

def numeric_gradient(formulas, weights, backend="pytorch", **kwargs):
    grad = build(formulas, weights, backend).gradient(key="weights", **kwargs)
    return [float(g) for g in grad]

def torch_gradient(formulas, weights, **kwargs):
    return numeric_gradient(formulas, weights, "pytorch", **kwargs)

## numerical gradients against finite differences

@pytest.mark.parametrize("backend", NUMERIC_BACKENDS)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", GRAD_CASES)
def test_gradient_matches_finite_differences(case, mode, backend):
    formulas, weights = CASES[case]
    for evidence in _evidence_options(formulas, weights):
        kwargs = {"evidence": evidence} if evidence else {}
        grad = numeric_gradient(formulas, weights, backend, mode=mode, **kwargs)
        fd = finite_differences(mode, formulas, weights, _evidence_dict(evidence))
        assert all(math.isfinite(g) for g in grad), (evidence, grad)
        assert grad == pytest.approx(fd, abs=1e-3), evidence

@pytest.mark.parametrize("backend", NUMERIC_BACKENDS)
@pytest.mark.parametrize("mode", ["Prob", "LogProb"])
def test_gradient_of_query_given_evidence(mode, backend):
    formulas, weights = CASES["implies"]
    variables = sorted(weights)

    def conditional(w):
        num = sum(_model_prob(w, m) for m in _models(formulas, variables, {"b": True, "c": True}))
        den = sum(_model_prob(w, m) for m in _models(formulas, variables, {"b": True}))
        return num / den if mode == "Prob" else math.log(num / den)

    eps = 1e-5
    fd = [
        (conditional(dict(weights, **{v: weights[v] + eps})) - conditional(dict(weights, **{v: weights[v] - eps}))) / (2 * eps)
        for v in variables
    ]
    grad = numeric_gradient(formulas, weights, backend, mode=mode, query="c", evidence="b")
    assert grad == pytest.approx(fd, abs=1e-3)

@pytest.mark.parametrize("backend", NUMERIC_BACKENDS)
def test_gradient_with_conditional_formula(backend):
    formulas, weights = CASES["implies"]
    variables = sorted(weights)
    G = "Or(a, c)"

    def conditional(w):
        return (
            sum(_model_prob(w, m) for m in _models(formulas + [G], variables))
            / sum(_model_prob(w, m) for m in _models([G], variables))
        )

    eps = 1e-5
    fd = [
        (conditional(dict(weights, **{v: weights[v] + eps})) - conditional(dict(weights, **{v: weights[v] - eps}))) / (2 * eps)
        for v in variables
    ]
    s = build(formulas, weights, backend)
    s.add_conditional(G)
    assert [float(g) for g in s.gradient(key="weights")] == pytest.approx(fd, abs=1e-3)

## symbolic gradients against pytorch

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", GRAD_CASES)
def test_sympy_gradient_matches_torch(case, mode):
    formulas, weights = CASES[case]
    for evidence in _evidence_options(formulas, weights):
        kwargs = {"evidence": evidence} if evidence else {}
        sym_grad = build(formulas, weights).gradient(
            mode=mode, symbolic=True, key="weights", **kwargs
        )
        assert isinstance(sym_grad, sp.Matrix)
        assert substitute(sym_grad, weights) == pytest.approx(
            torch_gradient(formulas, weights, mode=mode, **kwargs), abs=1e-4
        ), evidence

@pytest.mark.parametrize("mode", SMOOTH_MODES)
@pytest.mark.parametrize("case", GRAD_CASES)
def test_symengine_gradient_matches_torch(case, mode):
    formulas, weights = CASES[case]
    for evidence in _evidence_options(formulas, weights):
        kwargs = {"evidence": evidence} if evidence else {}
        sym_grad = build(formulas, weights).gradient(
            mode=mode, symbolic=True, symbolic_backend="symengine", key="weights", **kwargs
        )
        assert substitute(sym_grad, weights) == pytest.approx(
            torch_gradient(formulas, weights, mode=mode, **kwargs), abs=1e-4
        ), evidence

@pytest.mark.parametrize("mode", PIECEWISE_MODES)
def test_symengine_gradient_of_max_min_not_supported(mode):
    formulas, weights = CASES["implies"]
    with pytest.raises(NotImplementedError, match="sympy"):
        build(formulas, weights).gradient(
            mode=mode, symbolic=True, symbolic_backend="symengine", key="weights"
        )

def test_symbolic_gradient_with_conditional_formula():
    formulas, weights = CASES["implies"]
    s = build(formulas, weights)
    s.add_conditional("Or(a, c)")
    sym_grad = s.gradient(symbolic=True, key="weights")
    s = build(formulas, weights)
    s.add_conditional("Or(a, c)")
    assert substitute(sym_grad, weights) == pytest.approx(s.gradient(key="weights").tolist(), abs=1e-4)

def test_notebook_symbolic_gradient_example():
    """Mirrors `notebooks/solver_tutorial.ipynb`: the symbolic gradient of `Prob`,
    with values substituted using positive real sympy symbols."""
    formulas, weights = CASES["implies"]
    s = build(formulas, weights)
    a, b, c = sp.symbols("a b c", real=True, positive=True)

    sym_grad = s.gradient(mode="Prob", symbolic=True, key="weights")
    assert sp.simplify(sym_grad - sp.Matrix([b * (c - 1), a * c - a + 1, a * b])) == sp.zeros(3, 1)

    substituted = sym_grad.subs({sp.Symbol(k, real=True, positive=True): v for k, v in weights.items()})
    assert [float(x) for x in substituted] == pytest.approx([-0.4224, 0.406, 0.576], abs=1e-6)
    assert s.gradient(mode="Prob", key="weights").tolist() == pytest.approx([-0.4224, 0.406, 0.576], abs=1e-5)

@pytest.mark.parametrize("case", GRAD_CASES)
def test_prob_gradient_is_evidence_difference(case):
    """WMC is multilinear in the weights, so dP(F)/dw_x = P(F | x) - P(F | ~x)
    (both numerically and symbolically)."""
    formulas, weights = CASES[case]
    s = build(formulas, weights)
    grad = s.gradient(mode="Prob", key="weights").tolist()
    sym_grad = build(formulas, weights).gradient(mode="Prob", symbolic=True, key="weights")
    for i, var in enumerate(sorted(weights)):
        diff = s.query(evidence=var).inference.item() - s.query(evidence=f"~{var}").inference.item()
        assert grad[i] == pytest.approx(diff, abs=1e-5)
        sym_diff = (
            s.query(evidence=var, symbolic=True).inference
            - s.query(evidence=f"~{var}", symbolic=True).inference
        )
        assert sp.simplify(sym_grad[i] - sym_diff) == 0

## boolean gradients

@pytest.mark.parametrize("case", ["implies", "xor_chain"])
def test_boolean_gradient_is_boolean_difference(case):
    """The Boolean derivative wrt x is F|x XOR F|~x."""
    formulas, weights = CASES[case]
    F = _sympy_formula(formulas)
    grads = build(formulas, weights).gradient(boolean=True, simplify=True, key="weights")
    for var, grad in zip(sorted(weights), grads):
        x = sp.Symbol(var)
        expected = sp.Xor(F.subs(x, True), F.subs(x, False))
        grad = grad.subs({s: sp.Symbol(str(s)) for s in grad.free_symbols})
        assert sp.satisfiable(sp.Not(sp.Equivalent(grad, expected))) is False, var

def test_notebook_boolean_gradient_example():
    formulas, weights = CASES["implies"]
    grads = build(formulas, weights).gradient(boolean=True, simplify=True, key="weights")
    assert [str(g) for g in grads] == ["b & ~c", "c | ~a", "a & b"]

## edge cases and API

@pytest.mark.parametrize("backend", NUMERIC_BACKENDS)
def test_gradient_is_zero_when_output_is_constant(backend):
    assert numeric_gradient(["a"], {"a": 0.3}, backend, evidence="a") == [0.0]

@pytest.mark.parametrize("backend", NUMERIC_BACKENDS)
def test_count_gradient_is_zero(backend):
    formulas, weights = CASES["implies"]
    assert numeric_gradient(formulas, weights, backend, mode="Count") == [0.0, 0.0, 0.0]

@pytest.mark.parametrize("backend", NUMERIC_BACKENDS)
def test_log_space_gradient_has_no_nans(backend):
    """Evidence introduces log(0) = -inf terms in log space, whose gradients must not be nan."""
    formulas, weights = CASES["implies"]
    for mode in ["LogProb", "LogViterbi"]:
        grad = numeric_gradient(formulas, weights, backend, mode=mode, evidence="~c")
        assert all(math.isfinite(g) for g in grad), (mode, grad)

def test_gradient_output_format():
    formulas, weights = CASES["implies"]
    s = build(formulas, weights)
    grads = s.gradient(mode="Prob")
    assert set(grads) == {"weights"}
    ## ordered by sorted variable names; repeated calls give the same result
    assert grads["weights"].tolist() == pytest.approx(s.gradient(mode="Prob", key="weights").tolist())
    assert grads["weights"].shape == (len(weights),)

def test_gradient_from_kwargs_with_requires_grad():
    formulas, weights = CASES["implies"]
    s = Solver.from_kwargs(var_list=sorted(weights), var_weights=weights, requires_grad=True)
    s.add(formulas)
    assert s.gradient(key="weights").tolist() == pytest.approx(torch_gradient(formulas, weights), abs=1e-6)

def test_gradient_not_supported_for_python_weights():
    formulas, weights = CASES["implies"]
    s = build(formulas, weights, container_type="python")
    with pytest.raises(NotImplementedError):
        s.gradient(key="weights")

def test_weight_manager_gradients():
    """Mirrors `notebooks/weight_interface.ipynb`: gradients directly from weight managers."""
    w = BuildWeights(manager="pytorch", weights=[0.5, 0.6], wmap={"a": 0, "b": 1}, requires_grad=True)
    f = w.mult(w.assignment("a"), w.log(w.assignment("b")))
    assert w.gradient(f, key="weights").tolist() == pytest.approx([math.log(0.6), 0.5 / 0.6], abs=1e-6)

    sym = BuildWeights(manager="sympy", weights=[0.5, 0.6], wmap={"a": 0, "b": 1}, symbolic=True)
    a, b = sym.assignment("a"), sym.assignment("b")
    f = sym.mult(sym.log(a), sym.log(b))
    assert sp.simplify(sym.gradient(f, key="weights") - sp.Matrix([sp.log(b) / a, sp.log(a) / b])) == sp.zeros(2, 1)

## language-level gradients

@Parser(language="constraint")
def constraint_prog():
    a: 0.9; b: 0.7; c: 0.65
    (not a) or b
    c or b
    (not b) or c

CONSTRAINT_FORMULAS = ["Or(Not(a), b)", "Or(c, b)", "Or(Not(b), c)"]
CONSTRAINT_WEIGHTS = {"a": 0.9, "b": 0.7, "c": 0.65}

def test_language_gradients():
    grads = constraint_prog.grad()
    assert grads["weights"].tolist() == pytest.approx(
        finite_differences("Prob", CONSTRAINT_FORMULAS, CONSTRAINT_WEIGHTS), abs=1e-3
    )
    assert constraint_prog.grad(mode="Entropy", key="weights").tolist() == pytest.approx(
        finite_differences("Entropy", CONSTRAINT_FORMULAS, CONSTRAINT_WEIGHTS), abs=1e-3
    )
    sym_grad = constraint_prog.grad(symbolic=True, key="weights")
    assert substitute(sym_grad, CONSTRAINT_WEIGHTS) == pytest.approx(grads["weights"].tolist(), abs=1e-4)

## inference as a function of the weights

@pytest.mark.parametrize("backend", NUMERIC_BACKENDS)
def test_as_function(backend):
    formulas, weights = CASES["xor_chain"]
    s = build(formulas, weights, backend)
    assert s.weight_names == sorted(weights)
    f = s.as_function(mode="Prob", evidence="b")
    other = {"a": 0.3, "b": 0.1, "c": 0.5, "d": 0.8}
    x = [other[v] for v in s.weight_names]
    if backend == "jax":
        import jax.numpy as jnp
        x = jnp.array(x)
    else:
        x = torch.tensor(x)
    assert float(f(x)) == pytest.approx(reference("Prob", formulas, other, {"b": True}), abs=1e-5)
    ## the solver's own weights are left unchanged
    assert float(s.query(evidence="b").inference) == pytest.approx(
        reference("Prob", formulas, weights, {"b": True}), abs=1e-5
    )

def test_as_function_torch_autograd():
    formulas, weights = CASES["implies"]
    s = build(formulas, weights)
    x = torch.tensor([weights[v] for v in s.weight_names], requires_grad=True)
    (grad,) = torch.autograd.grad(s.as_function(mode="LogProb")(x), x)
    assert grad.tolist() == pytest.approx(torch_gradient(formulas, weights, mode="LogProb"), abs=1e-5)

@pytest.mark.skipif(not HAS_JAX, reason="jax not installed")
@pytest.mark.parametrize("mode", ["Prob", "LogProb", "Entropy", "Viterbi", "LogViterbi", "Fuzzy"])
def test_jax_jit_and_vmap(mode):
    import jax
    import jax.numpy as jnp

    formulas, weights = CASES["xor_chain"]
    s = build(formulas, weights, "jax")
    f = s.as_function(mode=mode, evidence="~a")
    batch = [
        weights,
        {"a": 0.3, "b": 0.1, "c": 0.5, "d": 0.8},
        {"a": 0.6, "b": 0.45, "c": 0.25, "d": 0.35},
    ]
    x = jnp.array([[w[v] for v in s.weight_names] for w in batch])

    values = jax.jit(jax.vmap(f))(x)
    grads = jax.jit(jax.vmap(jax.grad(f)))(x)
    for i, w in enumerate(batch):
        assert float(values[i]) == pytest.approx(reference(mode, formulas, w, {"a": False}), abs=1e-5)
        assert [float(g) for g in grads[i]] == pytest.approx(
            finite_differences(mode, formulas, w, {"a": False}), abs=1e-3
        )

@pytest.mark.skipif(not HAS_JAX, reason="jax not installed")
def test_jax_gradient_requires_function():
    formulas, weights = CASES["implies"]
    s = build(formulas, weights, "jax")
    with pytest.raises(TypeError, match="as_function"):
        s._weights.gradient(s.query().inference)
