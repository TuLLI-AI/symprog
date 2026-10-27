# type: ignore 
import pytest

from symprog.solvers import solver
from symprog.solvers.solver import Solver

def test_solver_add_string_formulas_and_formula_property():
    s = Solver()
    s.add("And(a, Or(b, c))")
    assert len(s.formulas) == 1
    # formula property should build a solver-native object
    f = s.formula
    assert f is not None

def test_solver_add_multiple_formulas_and_joint_conjunction():
    s = Solver()
    s.add(["a", "Or(b, c)", "Implies(d, a)"])
    assert len(s.formulas) == 3
    joint = s.formula
    assert joint is not None

def test_solver_pop_removes_last_formula():
    s = Solver()
    s.add(["a", "b"])
    s.pop()
    assert len(s.formulas) == 1

def test_solver_reset_clears_state():
    s = Solver()
    s.add("a")
    s.add_weights({"a": 0.5})
    s.reset()
    assert s.formulas == []
    #assert s.weights is None
    assert s._solver is None

# Conditionals
def test_solver_add_and_remove_conditional():
    s = Solver()
    s.add("And(a, b)")
    s.add_conditional("a")
    assert s._conditional is not None
    # internal indicator var _F should be tracked as aux
    assert "_F" in s._aux_vars
    s.remove_conditional()
    assert s._conditional is None
    # _F remains a (now unused) solver variable, so it stays auxiliary to be ignored in inference
    assert "_F" in s._aux_vars

# Variables and BoolOps helpers
def test_solver_variables_and_boolops_helpers():
    s = Solver()
    s.add("And(a, b)")
    a_var, b_var = s.Variables("a b")
    assert a_var is not None and b_var is not None
    And, Or = s.BoolOps("And Or")
    assert callable(And) and callable(Or)

# Weights processing
def test_solver_add_weights_with_dict():
    s = Solver()
    s.add("a")
    s.add_weights({"a": 0.7})
    #assert s.weights is not None
    # internal manager param exposure
    assert "a" in s._weights._wmap

def test_solver_add_weights_with_list_of_pairs():
    s = Solver()
    s.add("Or(a, b)")
    s.add_weights([("a", 0.5), ("b", 0.6)])
    #assert s.weights is not None
    assert set(s._weights._wmap.keys()) >= {"a", "b"}

def test_solver_add_weights_raises_without_formula():
    s = Solver()
    with pytest.raises(ValueError):
        s.add_weights({"a": 0.5})

def test_solver_add_weights_rejects_bad_list_shape():
    s = Solver()
    s.add("a")
    with pytest.raises(ValueError):
        s.add_weights([("a", 0.5, 0.1)])

# Inference: SAT, Count, Prob
def test_solver_sat_inference():
    s = Solver()
    s.add("And(a, b)")
    out = s.query(mode="SAT")
    assert getattr(out, "inference", None) == "yes"

def test_solver_unsat_inference():
    s = Solver()
    s.add("And(a, Not(a))")
    out = s.query(mode="SAT")
    assert getattr(out, "inference", None) == "no"

def test_solver_fast_count_inference():
    s = Solver()
    s.add("And(a, b)")
    out = s.query(mode="Count", fast=True)
    assert isinstance(out.inference, (int, float)) 
    assert out.inference == 1

def test_solver_fast_non_count_inference():
    s = Solver()
    s.add("And(a, Not(a))")
    out = s.query(mode="Count", fast=True)
    assert isinstance(out.inference, (int, float)) 
    assert out.inference == 0


def test_solver_prob_requires_weights():
    s = Solver()
    s.add("a")
    with pytest.raises(ValueError):
        s.query(mode="Prob")
# Evidence and conditionals behavior
def test_solver_evidence_string_parsing_no_conditional():
    s = Solver()
    s.add("Or(a, b)")
    s.add_weights([("a", 0.5), ("b", 0.5)])
    out = s.query(mode="Prob", evidence="a")
    assert hasattr(out, "inference")

def test_solver_conditional_changes_query_flow():
    s = Solver()
    s.add("And(a, b)")
    s.add_weights([("a", 0.5), ("b", 0.6)])
    s.add_conditional("a")
    out = s.query(mode="Prob")
    assert hasattr(out, "inference")

# def test_variable_adding():
#     s = Solver()
#     And, Or, Implies = s.BoolOps("And Or Implies")
#     s.add("And(a, b)")
#     x, y = s.Variables("x y")
#     assert x is not None and y is not None

#     s._solver.add_symbol("z")
#     z = s._solver.Symbol("z")    

#     formula = Implies(Or(x, y), z)
#     solver.inference(formula, mode="Prob")
    
### condtitional semantics

WEIGHTS = {
    "a" : 0.9, 
    "b" : 0.64, 
    "c" : 0.34
}

def prob(output) -> float:
    return output.inference.item()


def test_conditional_semantics_tautology():
    s = Solver()
    s.add("And(a, b)")
    s.add_conditional("Or(a,Not(a))")
    s.add_weights(WEIGHTS)

    assert prob(s.query()) == pytest.approx(0.9 * 0.64, abs=1e-5)

def test_conditional_semantics_atom():
    s = Solver()
    s.add("And(a, b)")
    s.add_conditional("a")
    s.add_weights(WEIGHTS)

    assert prob(s.query()) == pytest.approx((0.9 * 0.64) / 0.9, abs=1e-5)

def test_conditional_semantics_atom_evidence():
    s = Solver()
    s.add("And(a, b)")
    s.add_weights(WEIGHTS)
    assert prob(s.query(evidence="a")) == pytest.approx((0.9 * 0.64) / 0.9, abs=1e-5)


def test_conditional_semantics_self():
    s = Solver()
    s.add("a")
    s.add_conditional("a")
    s.add_weights(WEIGHTS)

    assert prob(s.query()) == pytest.approx(1.0, abs=1e-5)

def test_conditional_semantics_implication():
    s = Solver()
    s.add("Implies(a, b)")
    s.add_conditional("a")
    s.add_weights(WEIGHTS)
    assert prob(s.query()) == pytest.approx(0.64, abs=1e-5)

def test_conditional_semantics_implication_evidence():
    s = Solver()
    s.add("Implies(a, b)")
    s.add_weights(WEIGHTS)
    assert prob(s.query(evidence="a")) == pytest.approx(0.64, abs=1e-5)


def test_conditional_semantics_implication_2():
    s = Solver()
    s.add("Implies(a, b)")
    s.add_conditional("b")
    s.add_weights(WEIGHTS)

    assert prob(s.query()) == pytest.approx(1.0, abs=1e-5)

def test_conditional_semantics_implication_evidence_2():
    s = Solver()
    s.add("Implies(a, b)")
    s.add_weights(WEIGHTS)

    assert prob(s.query(evidence="b")) == pytest.approx(1.0, abs=1e-5)


def test_conditional_semantics_implication_implication_3():
    s = Solver()
    s.add("Implies(a, b)")
    s.add_conditional("Or(a, b)")
    s.add_weights(WEIGHTS)

    assert prob(s.query()) == pytest.approx(
        0.64 / (((0.64 * 0.9) + ((1 - 0.9) * 0.64) + ((1 - 0.64) * 0.9))),
        abs=1e-5
    )

def test_conditional_semantics_implication_xor():
    s = Solver()
    s.add("Implies(a, b)")
    s.add_conditional("And(Or(a, b), Not(And(a, b)))")
    s.add_weights(WEIGHTS)

    assert prob(s.query()) == pytest.approx(
        (0.64*(1 - 0.9)) / ((0.64*(1 - 0.9)) + ((1 - 0.64)*0.9)),
        abs=1e-5
    )

def test_conditional_semantics_implication_disjunction():
    s = Solver()
    s.add("Or(a, b)")
    s.add_weights(WEIGHTS)

    assert prob(s.query()) == pytest.approx(
        (0.64 + ((1 - 0.64) * 0.9) ),
        abs=1e-5
    )


# ---------------------------------------------------------------------------
# Conditional semantics — broader coverage
# ---------------------------------------------------------------------------

def test_conditional_negation_zero():
    """P(And(a,b) | Not(a)) = 0 — condition contradicts the constraint."""
    s = Solver()
    s.add("And(a, b)")
    s.add_conditional("Not(a)")
    s.add_weights(WEIGHTS)
    assert prob(s.query()) == pytest.approx(0.0, abs=1e-5)


def test_conditional_negation_of_self_zero():
    """P(a | Not(a)) = 0 — atom conditioned on its own negation."""
    s = Solver()
    s.add("a")
    s.add_conditional("Not(a)")
    s.add_weights(WEIGHTS)
    assert prob(s.query()) == pytest.approx(0.0, abs=1e-5)


def test_conditional_conjunction_subsumes_formula():
    """P(a | And(a, b)) = 1 — condition logically implies the formula."""
    s = Solver()
    s.add("a")
    s.add_conditional("And(a, b)")
    s.add_weights(WEIGHTS)
    assert prob(s.query()) == pytest.approx(1.0, abs=1e-5)


def test_conditional_independent_third_var():
    """P(And(a, b) | c) = P(a)*P(b) — conditioning on an independent var is a no-op."""
    s = Solver()
    s.add("And(a, b)")
    s.add_conditional("c")
    s.add_weights(WEIGHTS)
    assert prob(s.query()) == pytest.approx(0.9 * 0.64, abs=1e-5)


def test_conditional_three_var_formula():
    """P(And(a, c) | b) = P(a)*P(c) — three-variable independence check."""
    s = Solver()
    s.add("And(a, c)")
    s.add_conditional("b")
    s.add_weights(WEIGHTS)
    assert prob(s.query()) == pytest.approx(0.9 * 0.34, abs=1e-5)


def test_conditional_disjunction_condition():
    """P(And(a,b) | Or(a,b)) = P(a∧b) / P(a∨b)."""
    s = Solver()
    s.add("And(a, b)")
    s.add_conditional("Or(a, b)")
    s.add_weights(WEIGHTS)
    p_or = 1 - (1 - 0.9) * (1 - 0.64)
    expected = (0.9 * 0.64) / p_or
    assert prob(s.query()) == pytest.approx(expected, abs=1e-5)


def test_conditional_biconditional_condition():
    """P(a | Biconditional(a,b)) = P(a∧b) / (P(a∧b) + P(¬a∧¬b))."""
    s = Solver()
    s.add("a")
    s.add_conditional("Biconditional(a, b)")
    s.add_weights(WEIGHTS)
    p_bicond = 0.9 * 0.64 + 0.1 * 0.36
    expected = (0.9 * 0.64) / p_bicond
    assert prob(s.query()) == pytest.approx(expected, abs=1e-5)


def test_conditional_not_conjunction_condition():
    """P(a | Not(And(a,b))) = P(a∧¬b) / P(¬(a∧b))."""
    s = Solver()
    s.add("a")
    s.add_conditional("Not(And(a, b))")
    s.add_weights(WEIGHTS)
    p_not_and = 1 - 0.9 * 0.64
    p_a_and_not_b = 0.9 * (1 - 0.64)
    expected = p_a_and_not_b / p_not_and
    assert prob(s.query()) == pytest.approx(expected, abs=1e-5)


def test_conditional_remove_restores_probability():
    """remove_conditional() reverts to the unconditional probability."""
    s = Solver()
    s.add("And(a, b)")
    s.add_weights(WEIGHTS)
    s.add_conditional("a")
    assert prob(s.query()) == pytest.approx(0.64, abs=1e-5)
    s.remove_conditional()
    assert prob(s.query()) == pytest.approx(0.9 * 0.64, abs=1e-5)




def test_conditional_overwrite_warns_and_updates():
    """A second add_conditional() issues a warning and replaces the first."""
    s = Solver()
    s.add("And(a, b)")
    s.add_conditional("a")
    s.add_weights(WEIGHTS)
    with pytest.warns(UserWarning):
        s.add_conditional("b")
    # P(And(a,b) | b) = P(a) = 0.9
    assert prob(s.query()) == pytest.approx(0.9, abs=1e-5)


def test_conditional_before_formula_raises():
    """add_conditional() before any formula is added raises ValueError."""
    s = Solver()
    with pytest.raises(ValueError):
        s.add_conditional("a")

### direct marginal queries 

def test_conditional_direct_evidence():
    """Direct marginal probability without conditioning."""
    s = Solver()
    s.add("And(a, b)")
    s.add_weights(WEIGHTS)
    assert prob(s.query(evidence="a")) == pytest.approx(0.64, abs=1e-5)
    assert prob(s.query(evidence="b")) == pytest.approx(0.9, abs=1e-5)
    assert prob(s.query(evidence=set(["a","b"]))) == pytest.approx(1.0, abs=1e-5)

def test_conditional_direct_evidence_2():
    """Direct marginal probability without conditioning."""
    s = Solver()
    s.add("Implies(a, b)")
    s.add_weights(WEIGHTS)
    assert prob(s.query(evidence="a")) == pytest.approx(0.64, abs=1e-5)
    assert prob(s.query(evidence="b")) == pytest.approx(1.0, abs=1e-5)
    assert prob(s.query(evidence=set(["a","b"]))) == pytest.approx(1.0, abs=1e-5)

def test_direct_query_1():
    s = Solver()
    s.add("Implies(a, b)")
    s.add_weights(WEIGHTS)
    ## queries are conditioned on the formula: P(q | a -> b), with P(a -> b) = 1 - 0.9 * 0.36
    p_formula = 1 - 0.9 * (1 - 0.64)
    assert prob(s.query(query="a")) == pytest.approx(0.9 * 0.64 / p_formula, abs=1e-5)
    assert prob(s.query(query="b")) == pytest.approx(0.64 / p_formula, abs=1e-5) #<-- needs smoothness

def test_direct_query_2():
    s = Solver()
    s._aux_vars.add("f")
    s.add("Biconditional(f, Implies(a, b))")
    s.add_weights(WEIGHTS)
    assert prob(s.query(query="a")) == pytest.approx(0.9, abs=1e-5)
    assert prob(s.query(query="b")) == pytest.approx(0.64, abs=1e-5)

def test_direct_query_3():
    s = Solver()
    s._aux_vars.add("f")
    s.add("Biconditional(f, And(a, b))")
    s.add_weights(WEIGHTS)
    assert prob(s.query(query="a")) == pytest.approx(0.9, abs=1e-5)
    assert prob(s.query(query="b")) == pytest.approx(0.64, abs=1e-5)

def test_direct_query_evidence_1():
    s = Solver()
    s._aux_vars.add("f")
    s.add("Biconditional(f, Or(a, b))")
    s.add_weights(WEIGHTS)
    assert prob(s.query(query="b", evidence="f")) == pytest.approx(0.66390041, abs=1e-5)
    assert prob(s.query(query="a", evidence="f")) == pytest.approx(0.93360996, abs=1e-5)

def test_direct_query_evidence_2():
    s = Solver()
    s._aux_vars.add("f")
    s.add("Biconditional(f, And(a, b))")
    s.add_weights(WEIGHTS)
    assert prob(s.query(query="b", evidence="f")) == pytest.approx(1.0, abs=1e-5)
    assert prob(s.query(query="a", evidence="f")) == pytest.approx(1.0, abs=1e-5)

def test_direct_query_evidence_3():
    s = Solver()
    s._aux_vars.add("f")
    s.add("Biconditional(f, Or(a, b))")
    s.add_conditional("f")
    s.add_weights(WEIGHTS)
    assert prob(s.query(query="b")) == pytest.approx(0.66390041, abs=1e-5)

### probabilistic facts: https://arxiv.org/pdf/1312.4328
## https://dtai.cs.kuleuven.be/problog/editor.html#task=prob&hash=ade948eb219d97009b86ccd80d3207e1

def test_probabilistic_facts_1():
    s = Solver()
    s._aux_vars.add("a")
    s._aux_vars.add("c")
    s._aux_vars.add("cm")
    s._aux_vars.add("cj")
    formulas = [
        "Biconditional(a, Or(e, b))", 
        "Biconditional(cm, And(a,hm))",
        "Biconditional(cj, And(a,hj))",
        "Biconditional(c, Or(cj, cm))"     
    ]    

    pweights = {
        "b"  : 0.1, 
        "e"  : 0.2, 
        "hm" : 0.7,
        "hj" : 0.4, 
    }
    s.add(formulas)
    s.add_weights(pweights)

    assert prob(s.query(query="b", evidence="cm")) == pytest.approx(0.35714, abs=1e-5)
    assert prob(s.query(query="cj", evidence="cm")) == pytest.approx(0.4, abs=1e-5)
