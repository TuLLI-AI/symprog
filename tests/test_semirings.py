# type: ignore
"""Tests for the semiring implementations, checked against brute-force enumeration.

Each semiring (inference `mode`) is compared with a reference value computed by
enumerating all assignments to the formula variables, across weight backends,
with evidence, queries and conditional formulas.
"""
import itertools
import math
import warnings

import pytest
import sympy as sp

from symprog.solvers.semiring import SemiringRegistry

from .reference import CASES, _models, _model_prob, _sympy_formula, build, reference, value

WEIGHT_BACKENDS = ["pytorch", "fast_pytorch", "python"]

## modes compared numerically against brute force below
NUMERIC_MODES = ["Prob", "LogProb", "Count", "Viterbi", "LogViterbi", "Fuzzy", "Entropy"]

## every registered mode must be covered by a test in this module
TESTED_MODES = set(NUMERIC_MODES) | {"ViterbiAssignment", "Formula", "Product", "TestLukasiewicz"}

## coverage of the registry

def test_all_registered_semirings_are_tested():
    registered = set(SemiringRegistry._IMPLEMENTATIONS.keys())
    untested = registered - TESTED_MODES
    assert not untested, f"Registered semirings without tests: {untested}"

## numeric modes against brute force

@pytest.mark.parametrize("backend", WEIGHT_BACKENDS)
@pytest.mark.parametrize("mode", NUMERIC_MODES)
@pytest.mark.parametrize("case", list(CASES))
def test_mode_matches_brute_force(case, mode, backend):
    formulas, weights = CASES[case]
    s = build(formulas, weights, backend)
    expected = reference(mode, formulas, weights)
    assert value(s.query(mode=mode)) == pytest.approx(expected, abs=1e-5)

@pytest.mark.parametrize("mode", NUMERIC_MODES)
@pytest.mark.parametrize("case", list(CASES))
def test_mode_with_evidence(case, mode):
    formulas, weights = CASES[case]
    for var in sorted(weights):
        for polarity in (True, False):
            evidence = {var: polarity}
            if not _models(formulas, sorted(weights), evidence):
                continue
            s = build(formulas, weights)
            literal = var if polarity else f"~{var}"
            expected = reference(mode, formulas, weights, evidence)
            assert value(s.query(mode=mode, evidence=literal)) == pytest.approx(expected, abs=1e-5), literal

@pytest.mark.parametrize("mode", ["Prob", "LogProb"])
@pytest.mark.parametrize("case", ["implies", "xor_chain", "irrelevant_var"])
def test_query_given_evidence(case, mode):
    """P(q | F, e) for all pairs of distinct query/evidence literals."""
    formulas, weights = CASES[case]
    variables = sorted(weights)
    s = build(formulas, weights)
    for (qv, ev) in itertools.permutations(variables, 2):
        for qpol, epol in itertools.product((True, False), repeat=2):
            den = sum(_model_prob(weights, m) for m in _models(formulas, variables, {ev: epol}))
            if den == 0:
                continue
            num = sum(_model_prob(weights, m) for m in _models(formulas, variables, {ev: epol, qv: qpol}))
            expected = num / den if mode == "Prob" else (math.log(num / den) if num > 0 else -math.inf)
            out = s.query(
                mode=mode,
                query=qv if qpol else f"~{qv}",
                evidence=ev if epol else f"~{ev}",
            )
            assert value(out) == pytest.approx(expected, abs=1e-5)

@pytest.mark.parametrize("mode", ["Prob", "LogProb"])
@pytest.mark.parametrize("case", ["implies", "xor_chain", "irrelevant_var", "tautology_clause"])
def test_query_without_evidence(case, mode):
    """Without evidence, a query gives the conditional P(q | F) (not the joint P(F, q))."""
    formulas, weights = CASES[case]
    variables = sorted(weights)
    s = build(formulas, weights)
    Z = sum(_model_prob(weights, m) for m in _models(formulas, variables))
    for var in variables:
        for polarity in (True, False):
            joint = sum(_model_prob(weights, m) for m in _models(formulas, variables, {var: polarity}))
            expected = joint / Z if mode == "Prob" else (math.log(joint / Z) if joint > 0 else -math.inf)
            literal = var if polarity else f"~{var}"
            assert value(s.query(mode=mode, query=literal)) == pytest.approx(expected, abs=1e-5), literal

@pytest.mark.parametrize("case", ["implies", "irrelevant_var"])
def test_query_is_unchanged_by_entailed_evidence(case):
    """Evidence that F already entails does not change a query's answer.

    For `implies`, F entails b, so P(c | F) = P(c | F, b). Previously, a query without
    evidence returned the joint P(F, c), so adding the redundant evidence changed the answer.
    """
    formulas, weights = CASES[case]
    variables = sorted(weights)
    s = build(formulas, weights)
    checked = 0
    for entailed in variables:
        for polarity in (True, False):
            models = _models(formulas, variables)
            if not all(m[entailed] == polarity for m in models):
                continue
            checked += 1
            evidence = entailed if polarity else f"~{entailed}"
            for var in variables:
                if var == entailed:
                    continue
                assert value(s.query(query=var)) == pytest.approx(
                    value(s.query(query=var, evidence=evidence)), abs=1e-5
                ), (var, evidence)
    assert checked > 0, "F entails no literal in this case"

def test_query_without_evidence_example():
    """The example from the solver tutorial: P(c | F) = P(F, c) / P(F) = 0.2176 / 0.25984."""
    formulas, weights = CASES["implies"]
    s = build(formulas, weights)
    assert value(s.query(query="c")) == pytest.approx(0.2176 / 0.25984, abs=1e-5)
    assert value(s.query(query="c", evidence="b")) == pytest.approx(0.2176 / 0.25984, abs=1e-5)

@pytest.mark.parametrize("case", ["implies", "xor_chain"])
def test_conditional_formula(case):
    """With a conditional formula G, `Prob` gives P(F | G) and `Entropy` the entropy
    over models of F and G."""
    formulas, weights = CASES[case]
    variables = sorted(weights)
    G = "Or(a, b)"
    pG = sum(_model_prob(weights, m) for m in _models([G], variables))
    models = _models(formulas + [G], variables)
    probs = [_model_prob(weights, m) for m in models]

    s = build(formulas, weights)
    s.add_conditional(G)
    assert value(s.query(mode="Prob")) == pytest.approx(sum(probs) / pG, abs=1e-5)

    Z = sum(probs)
    entropy = -sum(p / Z * math.log(p / Z) for p in probs)
    assert value(s.query(mode="Entropy")) == pytest.approx(entropy, abs=1e-5)

## Viterbi assignments

@pytest.mark.parametrize("backend", WEIGHT_BACKENDS)
@pytest.mark.parametrize("case", list(CASES))
def test_viterbi_assignment(case, backend):
    formulas, weights = CASES[case]
    s = build(formulas, weights, backend)
    out = s.query(mode="ViterbiAssignment")
    ## assigns every variable, is a model, and achieves the Viterbi score
    assert set(out.variables) == set(weights)
    assert out.variables in _models(formulas, sorted(weights))
    assert _model_prob(weights, out.variables) == pytest.approx(
        reference("Viterbi", formulas, weights), abs=1e-5
    )
    assert value(out) == pytest.approx(reference("Viterbi", formulas, weights), abs=1e-5)

def test_viterbi_assignment_with_evidence():
    formulas, weights = CASES["implies"]
    s = build(formulas, weights)
    out = s.query(mode="ViterbiAssignment", evidence="~a")
    assert out.variables == {"a": False, "b": True, "c": False}

## Boolean formula semiring

@pytest.mark.parametrize("case", list(CASES))
def test_formula_semiring_is_equivalent(case):
    formulas, weights = CASES[case]
    s = build(formulas, weights)
    out = s.query(mode="Formula").inference
    F = _sympy_formula(formulas)
    assert sp.satisfiable(sp.Not(sp.Equivalent(out, F))) is False

def test_formula_semiring_with_evidence():
    formulas, weights = CASES["implies"]
    s = build(formulas, weights)
    out = s.query(mode="Formula", evidence="a").inference
    expected = sp.sympify("b & c")
    assert sp.satisfiable(sp.Not(sp.Equivalent(out, expected))) is False

## symbolic computation

@pytest.mark.parametrize("symbolic_backend", ["sympy", "symengine"])
@pytest.mark.parametrize("mode", ["Prob", "LogProb", "Viterbi", "LogViterbi", "Fuzzy", "Entropy"])
@pytest.mark.parametrize("case", ["implies", "xor_chain"])
def test_symbolic_matches_numeric(case, mode, symbolic_backend):
    formulas, weights = CASES[case]
    s = build(formulas, weights)
    expr = s.query(mode=mode, symbolic=True, symbolic_backend=symbolic_backend).inference
    if symbolic_backend == "symengine":
        expr = sp.sympify(expr)
    subs = {x: weights[str(x)] for x in expr.free_symbols}
    assert float(expr.subs(subs)) == pytest.approx(reference(mode, formulas, weights), abs=1e-5)

## edge cases

@pytest.mark.parametrize("deterministic_weight", [0.0, 1.0])
def test_entropy_with_deterministic_weights(deterministic_weight):
    """0 * log(0) is treated as 0 (no NaNs)."""
    formulas, weights = CASES["implies"]
    weights = dict(weights, a=deterministic_weight)
    s = build(formulas, weights)
    out = value(s.query(mode="Entropy"))
    assert not math.isnan(out)
    assert out == pytest.approx(reference("Entropy", formulas, weights), abs=1e-5)

def test_entropy_rejects_query():
    formulas, weights = CASES["implies"]
    s = build(formulas, weights)
    with pytest.raises(NotImplementedError):
        s.query(mode="Entropy", query="c")

def test_count_is_unweighted():
    formulas, _ = CASES["implies"]
    s1 = build(formulas, {"a": 0.9, "b": 0.64, "c": 0.34})
    s2 = build(formulas, {"a": 0.1, "b": 0.2, "c": 0.3})
    assert value(s1.query(mode="Count")) == value(s2.query(mode="Count")) == 3.0

@pytest.mark.parametrize("case", list(CASES))
def test_fast_count_matches_semiring_count(case):
    formulas, weights = CASES[case]
    s = build(formulas, weights)
    assert float(s.query(mode="Count", fast=True).inference) == value(s.query(mode="Count"))

def test_fast_count_with_evidence_falls_back_to_exact():
    formulas, weights = CASES["implies"]
    s = build(formulas, weights)
    assert value(s.query(mode="Count", fast=True, evidence="a")) == 1.0

def test_product_semiring_no_weights():
    formulas, weights = CASES["implies"]
    s = build(formulas, weights)
    prob = SemiringRegistry.build_instance(name="Prob", weight_mgr=s.weights)
    count = SemiringRegistry.build_instance(name="Count", weight_mgr=s.weights)
    assert prob.no_weights is False
    assert count.no_weights is True

## experimental `Product` (product t-norm / probabilistic sum)

def test_product_mode_is_experimental():
    formulas, weights = CASES["single_var"]
    s = build(formulas, weights)
    with pytest.warns(UserWarning, match="experimental"):
        out = s.query(mode="Product")
    ## exact when the formula has a single model
    assert value(out) == pytest.approx(0.3, abs=1e-5)

def test_product_mode_in_unit_interval():
    formulas, weights = CASES["implies"]
    s = build(formulas, weights)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = value(s.query(mode="Product"))
    assert 0.0 <= out <= 1.0

## user-defined semirings

LUKASIEWICZ = '''
@SemiringRegistry("TestLukasiewicz")
class Lukasiewicz(Semiring):
    def one(self): return self._mgr.one()
    def zero(self): return self._mgr.zero()
    def product(self, x, y): return self._mgr.max(0, x + y - 1)
    def sum(self, x, y): return self._mgr.max(x, y)
    def negation(self, x): return self._mgr.neg(x)
'''

def test_user_defined_semiring_can_be_redefined():
    """Re-running a registration (e.g., a notebook cell) replaces it without warnings."""
    from symprog.solvers.semiring import Semiring
    namespace = {"SemiringRegistry": SemiringRegistry, "Semiring": Semiring}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        exec(LUKASIEWICZ, namespace)
        exec(LUKASIEWICZ, namespace)
    assert isinstance(namespace["Lukasiewicz"], type)

    formulas, _ = CASES["implies"]
    weights = {"a": 0.95, "b": 0.9, "c": 0.8}
    s = build(formulas, weights)
    expected = max(
        max(0.0, sum((weights[v] if b else 1 - weights[v]) for v, b in m.items()) - (len(m) - 1))
        for m in _models(formulas, sorted(weights))
    )
    assert value(s.query(mode="TestLukasiewicz")) == pytest.approx(expected, abs=1e-5)

def test_registering_a_taken_name_warns():
    from symprog.solvers.semiring import Semiring
    with pytest.warns(UserWarning, match="already registered"):
        @SemiringRegistry("Prob")
        class NotProb(Semiring):
            pass
    assert SemiringRegistry._IMPLEMENTATIONS["Prob"][0].__name__ == "Prob"
