# type: ignore
"""Tests for tracing the semiring value of each SDD node during inference (`trace`),
as used to draw annotated circuits in the console."""
import math

import pytest

from .reference import CASES, build, value

## modes with their ⊕ (on the first component of each value)
MODES = {
    "Prob": sum,
    "Count": sum,
    "Viterbi": max,
    "Fuzzy": max,
    "LogProb": lambda xs: math.log(sum(math.exp(x) for x in xs)),
}

def _first(v):
    v = v[0] if isinstance(v, tuple) else v
    return float(v.item()) if hasattr(v, "item") else float(v)

@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("mode", sorted(MODES))
@pytest.mark.parametrize("query", [None, "b"])
def test_trace_values(case, mode, query):
    formulas, weights = CASES[case]
    if query and query not in weights:
        pytest.skip("no such variable")
    solver = build(formulas, weights)
    trace = {}
    traced = solver.query(mode=mode, query=query, trace=trace)

    ## tracing does not change the result
    assert value(traced) == pytest.approx(value(solver.query(mode=mode, query=query)), abs=1e-6)

    ## the formula's node is traced, and each decision node is the ⊕ of its traced elements
    assert solver._formula_obj.id in trace["nodes"]
    assert trace["root"] is not None
    for node_id, elements in trace["elements"].items():
        assert _first(trace["nodes"][node_id]) == pytest.approx(
            MODES[mode]([_first(e) for e in elements]), abs=1e-5
        )
