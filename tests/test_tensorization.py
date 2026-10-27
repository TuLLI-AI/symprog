# type: ignore
"""Tests for (experimental) tensorization: compiled modules from `Solver.tensorize`, with the
`fx` (torch.fx) and `klay` backends, checked against brute-force enumeration over batches
of random weights.
"""
import importlib.util
import itertools
import math
import random
import warnings

import pytest
import torch

import symprog
from symprog import Parser, Solver

from .reference import CASES, _models, _model_prob, reference

pytestmark = pytest.mark.filterwarnings("ignore::symprog.ExperimentalWarning")

MODES = ["Prob", "LogProb", "Entropy", "Viterbi", "LogViterbi", "Fuzzy"]
HAS_KLAY = importlib.util.find_spec("klay") is not None

## helpers

def build(formulas, weights):
    s = Solver()
    s.add(formulas)
    s.add_weights(weights)
    return s

def random_weights(names, shape=(5,), seed=0):
    generator = torch.Generator().manual_seed(seed)
    return 0.05 + 0.9 * torch.rand(*shape, len(names), generator=generator)

def evaluate(module, x, names):
    """Evaluates `module` on weights `x[..., len(names)]` (ordered as `names`), passing
    only the module's inputs by name."""
    if not module.input_symbols:
        return module()
    return module(**{n: x[..., names.index(n)] for n in module.input_symbols})

def expected(fn, x, names):
    """Applies `fn(weights_dict)` to each weight vector in the batch `x`."""
    flat = x.reshape(-1, len(names))
    values = [fn(dict(zip(names, row.tolist()))) for row in flat]
    return torch.tensor(values, dtype=torch.float64).reshape(x.shape[:-1])

def assert_close(out, target):
    out = out.to(torch.float64).expand_as(target)
    assert torch.allclose(out, target, atol=1e-4, equal_nan=True), (out, target)

def random_cnf(seed):
    """A random CNF (as Solver formula strings) over 3-5 variables, and random weights."""
    rng = random.Random(seed)
    names = sorted(rng.sample("abcdef", rng.randint(3, 5)))
    clauses = []
    for _ in range(rng.randint(1, 4)):
        literals = [v if rng.random() < 0.5 else f"Not({v})" for v in rng.sample(names, rng.randint(1, 3))]
        clauses.append(literals[0] if len(literals) == 1 else f"Or({', '.join(literals)})")
    names = sorted({v for v in names if any(v in c for c in clauses)})
    weights = {v: round(rng.uniform(0.05, 0.95), 3) for v in names}
    return clauses, weights

RANDOM_CNFS = [random_cnf(seed) for seed in range(12)]
SATISFIABLE_CNFS = [(f, w) for f, w in RANDOM_CNFS if _models(f, sorted(w))]

## fx backend against brute force

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", list(CASES))
def test_fx_matches_brute_force(case, mode):
    formulas, weights = CASES[case]
    names = sorted(weights)
    s = build(formulas, weights)
    x = random_weights(names)
    for evidence in [None, names[0], f"~{names[-1]}"]:
        evidence_dict = {evidence.lstrip("~"): not evidence.startswith("~")} if evidence else None
        out = evaluate(s.tensorize(mode=mode, evidence=evidence), x, names)
        assert_close(out, expected(lambda w: reference(mode, formulas, w, evidence_dict), x, names))

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("formulas, weights", SATISFIABLE_CNFS)
def test_fx_random_formulas(formulas, weights, mode):
    names = sorted(weights)
    s = build(formulas, weights)
    x = random_weights(names, seed=1)
    assert_close(evaluate(s.tensorize(mode=mode), x, names), expected(lambda w: reference(mode, formulas, w), x, names))

@pytest.mark.parametrize("mode", ["Prob", "LogProb"])
@pytest.mark.parametrize("case", ["implies", "xor_chain", "irrelevant_var"])
def test_fx_query_given_evidence(case, mode):
    """P(q | F, e), and P(q | F) without evidence."""
    formulas, weights = CASES[case]
    names = sorted(weights)
    s = build(formulas, weights)
    x = random_weights(names)

    def conditional(query, evidence, w):
        den = sum(_model_prob(w, m) for m in _models(formulas, names, evidence))
        num = sum(_model_prob(w, m) for m in _models(formulas, names, {**evidence, **query}))
        if den == 0:
            return math.nan  ## evidence inconsistent with F
        p = num / den
        return p if mode == "Prob" else (math.log(p) if p > 0 else -math.inf)

    for q, e in itertools.permutations(names, 2):
        out = evaluate(s.tensorize(mode=mode, query=q, evidence=e), x, names)
        assert_close(out, expected(lambda w: conditional({q: True}, {e: True}, w), x, names))
    for q in names:
        out = evaluate(s.tensorize(mode=mode, query=f"~{q}"), x, names)
        assert_close(out, expected(lambda w: conditional({q: False}, {}, w), x, names))

@pytest.mark.parametrize("case", ["implies", "xor_chain", "tautology_clause"])
def test_fx_conditional_formula(case):
    """P(F | G) for a conditional formula G."""
    formulas, weights = CASES[case]
    names = sorted(weights)
    G = f"Or({names[0]}, {names[-1]})"
    s = build(formulas, weights)
    s.add_conditional(G)
    x = random_weights(names)

    def conditional(w):
        return (
            sum(_model_prob(w, m) for m in _models(formulas + [G], names))
            / sum(_model_prob(w, m) for m in _models([G], names))
        )

    assert_close(evaluate(s.tensorize(), x, names), expected(conditional, x, names))

@pytest.mark.parametrize("options", [
    {"symbolic_backend": "sympy"},
    {"simplify": True},
    {"try_horner": True},
    {"simplify": True, "try_horner": True},
    {"incremental_simplify": True},
])
@pytest.mark.parametrize("mode", ["Prob", "LogProb", "Entropy", "Viterbi"])
@pytest.mark.parametrize("case", ["implies", "xor_chain"])
def test_fx_options(case, mode, options):
    formulas, weights = CASES[case]
    names = sorted(weights)
    s = build(formulas, weights)
    x = random_weights(names)
    out = evaluate(s.tensorize(mode=mode, **options), x, names)
    assert_close(out, expected(lambda w: reference(mode, formulas, w), x, names))

@pytest.mark.parametrize("mode", ["Prob", "LogProb", "Entropy", "Viterbi"])
@pytest.mark.parametrize("case", list(CASES))
def test_fx_from_cnf(case, mode):
    """Solvers built through CNF (with auxiliary Tseitin variables)."""
    formulas, weights = CASES[case]
    names = sorted(weights)
    s = Solver.from_cnf(formulas=formulas, var_weights=weights)
    x = random_weights(names)
    out = evaluate(s.tensorize(mode=mode), x, names)
    assert_close(out, expected(lambda w: reference(mode, formulas, w), x, names))

## shapes and inputs

@pytest.mark.parametrize("backend", ["fx"] + (["klay"] if HAS_KLAY else []))
def test_batch_shapes(backend):
    formulas, weights = CASES["implies"]
    names = sorted(weights)
    module = build(formulas, weights).tensorize(backend=backend)
    assert module.input_symbols == names
    for shape in [(), (4,), (2, 3)]:
        x = random_weights(names, shape=shape)
        target = expected(lambda w: reference("Prob", formulas, w), x, names)
        assert module(x).shape == x.shape[:-1]
        assert_close(module(x), target)
        assert_close(module(*x.unbind(-1)), target)
    ## per-variable inputs broadcast against each other
    a = torch.tensor([0.2, 0.5, 0.9])
    out = module(a=a, b=torch.tensor(0.64), c=torch.tensor(0.34))
    assert_close(out, torch.tensor([reference("Prob", formulas, {"a": v, "b": 0.64, "c": 0.34}) for v in a.tolist()], dtype=torch.float64))

def test_fx_inputs_are_the_expression_variables():
    """Variables that do not affect the result (here `f`) are not inputs of `fx` modules."""
    formulas, weights = CASES["irrelevant_var"]
    module = build(formulas, weights).tensorize()
    assert module.input_symbols == ["c", "e"]

## gradients

@pytest.mark.parametrize("mode", ["Prob", "LogProb", "Entropy"])
@pytest.mark.parametrize("case", ["implies", "xor_chain"])
def test_fx_gradients(case, mode):
    formulas, weights = CASES[case]
    names = sorted(weights)
    s = build(formulas, weights)
    module = s.tensorize(mode=mode)
    x = random_weights(names, shape=(3,)).requires_grad_()
    (grad,) = torch.autograd.grad(evaluate(module, x, names).sum(), x)

    eps = 1e-5
    for b in range(x.shape[0]):
        w = dict(zip(names, x[b].tolist()))
        for i, v in enumerate(names):
            fd = (
                reference(mode, formulas, {**w, v: w[v] + eps}) - reference(mode, formulas, {**w, v: w[v] - eps})
            ) / (2 * eps)
            assert grad[b, i].item() == pytest.approx(fd, abs=1e-3), (b, v)

## warnings and errors

def test_tensorize_is_experimental():
    formulas, weights = CASES["implies"]
    s = build(formulas, weights)
    with pytest.warns(symprog.ExperimentalWarning, match="experimental"):
        s.tensorize()

    @Parser(language="plp")
    def coins():
        heads1: 0.5
        heads2: 0.6
        twoHeads :- heads1 & heads2
        query(twoHeads)

    with pytest.warns(symprog.ExperimentalWarning):
        module = coins.tensorize()
    assert_close(module(heads1=torch.tensor(0.5), heads2=torch.tensor(0.6)), torch.tensor(0.3, dtype=torch.float64))

def test_experimental_warnings_can_be_silenced():
    formulas, weights = CASES["implies"]
    s = build(formulas, weights)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings("ignore", category=symprog.ExperimentalWarning)
        s.tensorize()

def test_unknown_backend_raises():
    formulas, weights = CASES["implies"]
    with pytest.raises(ValueError, match="backend"):
        build(formulas, weights).tensorize(backend="unknown")

## klay backend

klay_only = pytest.mark.skipif(not HAS_KLAY, reason="klay not installed")
KLAY_MODES = ["Prob", "LogProb", "Viterbi", "LogViterbi", "Fuzzy"]

@pytest.mark.skipif(HAS_KLAY, reason="klay is installed")
def test_klay_requires_klay():
    formulas, weights = CASES["implies"]
    with pytest.raises(ImportError, match=r"symprog\[klay\]"):
        build(formulas, weights).tensorize(backend="klay")

@klay_only
@pytest.mark.parametrize("mode", KLAY_MODES)
@pytest.mark.parametrize("formulas, weights", list(CASES.values()) + RANDOM_CNFS)
def test_klay_matches_brute_force(formulas, weights, mode):
    names = sorted(weights)
    s = build(formulas, weights)
    module = s.tensorize(mode=mode, backend="klay")
    assert module.input_symbols == names
    x = random_weights(names, shape=(2, 3))
    assert_close(module(x), expected(lambda w: reference(mode, formulas, w), x, names))

@klay_only
@pytest.mark.parametrize("mode", KLAY_MODES)
@pytest.mark.parametrize("case", list(CASES))
def test_klay_from_cnf(case, mode):
    formulas, weights = CASES[case]
    names = sorted(weights)
    s = Solver.from_cnf(formulas=formulas, var_weights=weights)
    module = s.tensorize(mode=mode, backend="klay")
    assert module.input_symbols == names
    x = random_weights(names)
    assert_close(module(x), expected(lambda w: reference(mode, formulas, w), x, names))

@klay_only
@pytest.mark.parametrize("formulas, weights", list(CASES.values()) + RANDOM_CNFS)
def test_klay_smoothed_circuits(formulas, weights):
    """Max-based modes compile a smoothed circuit (KLay does not smooth SDDs itself)."""
    import klay
    from symprog.solvers.klay_interface import _smooth_circuit

    s = build(formulas, weights)
    circuit = _smooth_circuit(klay, s._solver, s._formula_obj, {s._solver[v].literal for v in weights})
    result = klay.check_smooth(circuit)
    assert result.is_smooth and result.is_decomposable

@klay_only
def test_klay_log_space_is_stable():
    """LogProb and LogViterbi stay finite where probabilities underflow (P = 0.1^400)."""
    names = [f"x{i:03d}" for i in range(400)]
    s = build(["And(" + ", ".join(names) + ")"], {v: 0.1 for v in names})
    x = torch.full((2, len(names)), 0.1)
    assert s.tensorize(mode="Prob", backend="klay")(x).tolist() == [0.0, 0.0]
    for mode in ["LogProb", "LogViterbi"]:
        out = s.tensorize(mode=mode, backend="klay")(x)
        assert out.tolist() == pytest.approx([400 * math.log(0.1)] * 2, rel=1e-5)

@klay_only
@pytest.mark.parametrize("mode", ["Prob", "LogProb", "Viterbi", "LogViterbi"])
@pytest.mark.parametrize("case", ["implies", "xor_chain", "irrelevant_var"])
def test_klay_matches_fx_and_gradients(case, mode):
    formulas, weights = CASES[case]
    names = sorted(weights)
    s = build(formulas, weights)
    klay_module, fx_module = s.tensorize(mode=mode, backend="klay"), s.tensorize(mode=mode)

    x = random_weights(names, shape=(4,)).requires_grad_()
    klay_out, fx_out = klay_module(x), evaluate(fx_module, x, names)
    assert torch.allclose(klay_out, fx_out, atol=1e-5)

    (klay_grad,) = torch.autograd.grad(klay_out.sum(), x)
    (fx_grad,) = torch.autograd.grad(fx_out.sum(), x)
    assert torch.allclose(klay_grad, fx_grad, atol=1e-4)

@klay_only
@pytest.mark.parametrize("mode", KLAY_MODES)
@pytest.mark.parametrize("formulas", [["a", "Not(a)"], ["Or(a, Not(a))"]])
def test_klay_constant_formulas(formulas, mode):
    module = build(formulas, {"a": 0.3}).tensorize(mode=mode, backend="klay")
    x = torch.tensor([[0.3], [0.8]])
    assert_close(module(x), expected(lambda w: reference(mode, formulas, w), x, ["a"]))

@klay_only
@pytest.mark.parametrize("options", [
    {"mode": "Entropy"},
    {"mode": "Count"},
    {"evidence": "a"},
    {"query": "c"},
    {"conditional": "Or(a, c)"},
])
def test_klay_unsupported_options(options):
    formulas, weights = CASES["implies"]
    s = build(formulas, weights)
    with pytest.raises(NotImplementedError):
        s.tensorize(backend="klay", **options)
    ## the solver is unchanged
    assert s._conditional is None
    assert s.query().inference.item() == pytest.approx(0.25984, abs=1e-5)

@klay_only
def test_klay_unsupported_solvers_and_conditionals():
    formulas, weights = CASES["implies"]
    s = Solver(_stype="bdd")
    s.add(formulas)
    s.add_weights(weights)
    with pytest.raises(NotImplementedError, match="sdd"):
        s.tensorize(backend="klay")

    s = build(formulas, weights)
    s.add_conditional("Or(a, c)")
    with pytest.raises(NotImplementedError, match="conditional"):
        s.tensorize(backend="klay")
    ## after removing the conditional, the auxiliary variable `_F` is not an input
    s.remove_conditional()
    for mode in ["Prob", "Viterbi"]:
        module = s.tensorize(mode=mode, backend="klay")
        assert module.input_symbols == ["a", "b", "c"]
        assert_close(
            module(torch.tensor([0.9, 0.64, 0.34])),
            torch.tensor(reference(mode, formulas, weights), dtype=torch.float64),
        )
