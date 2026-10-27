# type: ignore
"""Shared brute-force reference implementations and test cases for solver tests."""
import itertools
import math

import sympy as sp

from symprog import Solver

## (formulas, weights) test cases.
CASES = {
    "implies": (
        ["Implies(a, b)", "And(Or(a, b), Implies(a, c))"],
        {"a": 0.9, "b": 0.64, "c": 0.34},
    ),
    "single_var": (
        ["a"],
        {"a": 0.3},
    ),
    "xor_chain": (
        ["Or(And(a, Not(b)), And(Not(a), b))", "Implies(b, Or(c, d))"],
        {"a": 0.2, "b": 0.55, "c": 0.7, "d": 0.15},
    ),
    ## `f` is irrelevant after simplification but is still a variable of the program
    "irrelevant_var": (
        ["c", "Not(e)", "Or(Or(Not(e), c), Not(f))"],
        {"c": 0.656, "e": 0.581, "f": 0.853},
    ),
    "tautology_clause": (
        ["Or(a, Not(a))", "Or(b, c)"],
        {"a": 0.4, "b": 0.25, "c": 0.8},
    ),
}

## brute-force reference implementations

def _sympy_formula(formulas):
    return sp.And(*[sp.sympify(f) for f in formulas])

def _models(formulas, variables, evidence=None):
    """All satisfying assignments over `variables` that agree with `evidence`."""
    F = _sympy_formula(formulas)
    evidence = evidence or {}
    out = []
    for values in itertools.product([False, True], repeat=len(variables)):
        m = dict(zip(variables, values))
        if any(m[v] != b for v, b in evidence.items()):
            continue
        if bool(F.subs({sp.Symbol(k): v for k, v in m.items()})):
            out.append(m)
    return out

def _lit_weight(weights, var, value):
    return weights[var] if value else 1.0 - weights[var]

def _model_prob(weights, model, skip=()):
    return math.prod(_lit_weight(weights, v, b) for v, b in model.items() if v not in skip)

def reference(mode, formulas, weights, evidence=None):
    """Brute-force value of `mode`, given evidence literals (as {var: bool}).

    Evidence literals receive weight 1, so e.g. `Prob` gives P(F | e) for independent
    literal weights, consistent with the solver's semantics.
    """
    evidence = evidence or {}
    variables = sorted(weights)
    models = _models(formulas, variables, evidence)
    probs = [_model_prob(weights, m, skip=evidence) for m in models]
    Z = sum(probs)
    if mode == "Count":
        return float(len(models))
    if mode == "Prob":
        return Z
    if mode == "LogProb":
        return math.log(Z) if Z > 0 else -math.inf
    if mode == "Viterbi":
        return max(probs, default=0.0)
    if mode == "LogViterbi":
        return math.log(max(probs)) if probs and max(probs) > 0 else -math.inf
    if mode == "Fuzzy":
        return max(
            (min((_lit_weight(weights, v, b) if v not in evidence else 1.0) for v, b in m.items())
             for m in models),
            default=0.0,
        )
    if mode == "Entropy":
        if Z == 0:
            return math.nan  ## undefined for unsatisfiable formulas
        return -sum(p / Z * math.log(p / Z) for p in probs if p > 0)
    raise ValueError(mode)

def build(formulas, weights, container_type="pytorch"):
    s = Solver()
    s.add(formulas)
    s.add_weights(weights, container_type=container_type)
    return s

def value(out):
    v = out.inference
    return float(v.item()) if hasattr(v, "item") else float(v)
