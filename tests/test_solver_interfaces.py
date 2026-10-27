# type: ignore 
import pytest

from symprog.solvers.factory import BuildMapper

def _assert_models_have_keys(models, keys):
    for m in models:
        for k in keys:
            assert k in m

@pytest.mark.skipif(pytest.importorskip("sympy") is None, reason="sympy not available")
def test_sympy_basic_sat_and_models():
    sym = BuildMapper(solver="sympy", var_list=["a", "b", "c"])
    a, b, c = sym.Variables("a b c")
    formula = sym.Or(sym.Implies(a, b), sym.Implies(c, a))

    sat_out = sym.SAT(formula)
    assert sat_out.inference == "yes"

    count_out = sym.Count(formula)
    assert count_out.inference >= 1

    models = list(sym.Models(formula))
    assert len(models) >= 1
    _assert_models_have_keys(models, ["a", "b", "c"])


@pytest.mark.skipif(pytest.importorskip("pysdd") is None, reason="pysdd not available")
def test_sdd_sat_count_models_and_dot():
    s = BuildMapper(solver="sdd", var_list=["a", "b", "c"])
    a, b, c = s.Variables("a b c")
    formula = s.And(s.And(a, b), s.Not(c))

    sat_out = s.SAT(formula)
    assert sat_out.inference == "yes"

    count_out = s.Count(formula)

    assert isinstance(count_out.inference, int)
    # exactly one satisfying assignment: a=True, b=True, c=False
    assert count_out.inference == 1

    models = list(s.Models(formula))
    assert models == [{"a": True, "b": True, "c": False}]

    built = s.BuildFormula(formula)
    assert built.dot is not None
    assert isinstance(built.dot, str)
    assert len(built.dot) > 0


@pytest.mark.skipif(pytest.importorskip("dd") is None, reason="dd (BDD) not available")
def test_bdd_sat_count_models_and_dot():
    bd = BuildMapper(solver="bdd", var_list=["a", "b", "c"])
    a, b, c = bd.Variables("a b c")
    formula = bd.And(bd.And(a, b), bd.Not(c))

    sat_out = bd.SAT(formula)
    assert sat_out.inference == "yes"

    count_out = bd.Count(formula)
    assert isinstance(count_out.inference, int)
    assert count_out.inference == 1

    models = list(bd.Models(formula))
    assert models == [{"a": True, "b": True, "c": False}]

    built = bd.BuildFormula(formula)
    assert built.dot is not None
    assert isinstance(built.dot, str)
    assert len(built.dot) > 0

    # Secondary formula checks from notebook
    sec = bd.Implies(bd.And(a, b), bd.Not(c))
    sec_count = bd.Count(sec)
    assert sec_count.inference >= 1
    sec_models = list(bd.Models(sec))
    assert len(sec_models) >= 1

    ## TODO: check this
    #_assert_models_have_keys(sec_models, ["a", "b", "c"])


@pytest.mark.skipif(pytest.importorskip("pysat") is None, reason="pysat not available")
def test_pysat_basic_sat_models_and_stats():
    pys = BuildMapper(solver="pysat", var_list=["a", "b", "c"])
    a, b, c = pys.Variables("a b c")
    formula = pys.Or(a, b)

    sat_out = pys.SAT(formula, include_stats=True)
    assert sat_out.inference == "yes"
    # details may be None if stats aren’t requested or supported
    # but with include_stats=True, it should be a string or None; just sanity-check type
    assert sat_out.details is None or isinstance(sat_out.details, str)

    models = list(pys.Models(formula))
    assert len(models) >= 1

    ## check this, doesn't seem to return variables not in the formula 
    #_assert_models_have_keys(models, ["a", "b", "c"])
    # At least one satisfying assignment for a∨b
    assert any(m["a"] or m["b"] for m in models)

    count_out = pys.Count(formula)
    assert isinstance(count_out.inference, int), count_out.inference
    assert count_out.inference >= 1