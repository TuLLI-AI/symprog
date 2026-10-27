# type: ignore 
import pytest
import math

from symprog import Solver

FORMULAS = [
    "Implies(a, b)", # a -> b 
    "And(Or(a, b), Implies(a, c))",
]
WEIGHTS = {
    "a" : 0.9, 
    "b" : 0.64, 
    "c" : 0.34
}

def prob(inference) -> float:
    return inference.inference.item()

def manual_wmc(solver, w, log_prob=False):
    """Manual computation of weighted model count via model enumeration. 
    
    Arguments: 
        solver (Solver): 
            The solver object provided with loaded formulas.  
        w (dict[str, float]): 
            The weights for the variables. 
    """
    if solver.formulas is None or not w: 
        raise ValueError(
            'Must provide solver with loaded formulas and weights'
        )
    WMC = 0.0 
    ## enumerate through models of F, or i |= F
    for i in solver.models():
        row_score = 1.0 
        ## all variables in i with their assignment 
        for (v, bool_val) in i.items():
            v_weight = w[v]
            row_score *= (v_weight if bool_val else (1.0 - v_weight))
        WMC += row_score
    return math.log(WMC) if log_prob else WMC

def manual_conditional_wmc(solver, w, evidence):
    for model in solver.models():
        pass 




## basic inference  

def test_basic_inference_prob():
    solver = Solver()
    solver.add(FORMULAS)
    solver.add_weights(WEIGHTS)

    inference = solver.query()
    man_wmc = manual_wmc(solver, WEIGHTS)
    assert prob(inference) == pytest.approx(man_wmc, abs=1e-5)
    assert prob(inference) == pytest.approx(0.2598400, abs=1e-5)

def test_basic_inference_logprob():
    solver = Solver()
    solver.add(FORMULAS)
    solver.add_weights(WEIGHTS)

    inference = solver.query(mode="LogProb")
    man_wmc = manual_wmc(solver, WEIGHTS, log_prob=True)
    assert prob(inference) == pytest.approx(man_wmc, abs=1e-5)

## from_cnf

def test_from_cnf_matches_direct_no_aux():
    """from_cnf agrees with direct solver when tseitin adds no aux vars."""
    direct = Solver()
    direct.add(FORMULAS)
    direct.add_weights(WEIGHTS)

    cnf = Solver.from_cnf(formulas=FORMULAS, var_list=list(WEIGHTS))
    cnf.add_weights(WEIGHTS)

    assert prob(cnf.query()) == pytest.approx(prob(direct.query()), abs=1e-5)


def test_from_cnf_matches_direct_with_aux():
    """from_cnf agrees with direct solver when tseitin introduces aux vars (biconditional)."""
    formula = ["Biconditional(And(a, b), Or(b, c))"]

    direct = Solver()
    direct.add(formula)
    direct.add_weights(WEIGHTS)

    cnf = Solver.from_cnf(formulas=formula, var_list=list(WEIGHTS))
    cnf.add_weights(WEIGHTS)

    assert prob(cnf.query()) == pytest.approx(prob(direct.query()), abs=1e-5)


def test_from_cnf_known_value():
    """from_cnf on the standard FORMULAS+WEIGHTS gives the known WMC."""
    cnf = Solver.from_cnf(formulas=FORMULAS, var_list=list(WEIGHTS))
    cnf.add_weights(WEIGHTS)
    assert prob(cnf.query()) == pytest.approx(0.2598400, abs=1e-5)


def test_from_cnf_matches_manual_wmc():
    """from_cnf result matches brute-force model enumeration."""
    direct = Solver()
    direct.add(FORMULAS)
    direct.add_weights(WEIGHTS)

    cnf = Solver.from_cnf(formulas=FORMULAS, var_list=list(WEIGHTS))
    cnf.add_weights(WEIGHTS)

    assert prob(cnf.query()) == pytest.approx(manual_wmc(direct, WEIGHTS), abs=1e-5)


## conditional inference

def test_conditional_inference_prob():
    solver = Solver()
    solver.add(FORMULAS)
    solver.add_weights(WEIGHTS)

    ## P(formula | evidence) = P(formula and evidence) / P(evidence)
    inference = solver.query(evidence="a")




