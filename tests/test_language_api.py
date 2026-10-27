# type: ignore
"""Tests for the `ProbabilisticProgram` API shared by the modeling languages
(inference, gradients, queries and evidence, and the generated sampling programs)."""
import pytest

from symprog import Parser, Solver

@Parser(language="plp")
def coins():
    heads1: 0.5
    heads2: 0.6
    twoHeads :- heads1 & heads2
    someHeads :- heads1
    someHeads :- heads2
    query(twoHeads)

@Parser(language="plp")
def two_queries():
    heads1: 0.5
    heads2: 0.6
    twoHeads :- heads1 & heads2
    someHeads :- heads1
    someHeads :- heads2
    query(twoHeads)
    query(someHeads)

@Parser(language="constraint")
def grounded():
    a: X; b: X; c: X
    s: {a: 0.9, b: 0.7, c: 0.2}
    f: {a: 0.5, b: 0.5, c: 0.5}
    Implies(s(X), f(X))
    Or(s(X))

def test_grad_uses_program_queries():
    ## P(twoHeads) = heads1 * heads2
    grad = coins.grad(key="weights")
    names = coins.solver.weight_names
    assert grad[names.index("heads1")].item() == pytest.approx(0.6, abs=1e-5)
    assert grad[names.index("heads2")].item() == pytest.approx(0.5, abs=1e-5)

def test_grad_uses_provided_evidence():
    ## P(twoHeads | heads1) = heads2
    grad = coins.grad(key="weights", evidence="heads1")
    names = coins.solver.weight_names
    assert grad[names.index("heads1")].item() == pytest.approx(0.0, abs=1e-5)
    assert grad[names.index("heads2")].item() == pytest.approx(1.0, abs=1e-5)

def test_multiple_queries_are_sorted():
    results = two_queries.inference()
    assert [r.details["marginal"] for r in results] == ["someHeads", "twoHeads"]
    assert [r.inference.item() for r in results] == pytest.approx([0.8, 0.3], abs=1e-5)
    assert len(two_queries.grad(key="weights")) == 2

def test_query_keyword():
    ## a provided `query` is added to the program's queries
    results = coins.inference(query="someHeads")
    assert [r.inference.item() for r in results] == pytest.approx([0.8, 0.3], abs=1e-5)

def test_grounded_program_samples_each_variable_once():
    lines = [l.strip() for l in grounded.prog_str.splitlines() if "np.random" in l]
    assert len(lines) == len(set(lines)) == 6

def test_solver_nary_xor_is_exactly_one():
    s = Solver()
    s.add(["Xor(a, b, c)"])
    s.add_weights({"a": 0.9, "b": 0.7, "c": 0.5})
    expected = 0.9 * 0.3 * 0.5 + 0.1 * 0.7 * 0.5 + 0.1 * 0.3 * 0.5
    assert s.query().inference.item() == pytest.approx(expected, abs=1e-5)
