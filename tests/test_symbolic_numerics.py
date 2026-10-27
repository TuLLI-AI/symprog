# type: ignore 
import pytest

from symprog.solvers.numerical_interfaces.symbolic import SympySymbolic #, SymengineSymbolic

def test_sympy_symbolic_basic_ops_and_assignment():
    w = SympySymbolic(weights=[0.1, 0.9], wmap={"a":0, "b":1})
    a = w.assignment("a")
    b = w.assignment("b")
    assert str(a) == "a"
    assert str(b) == "b"

    expr_add = w.add("a", "b")
    expr_mult = w.mult("a", "b")
    expr_div = w.div("a", "b")
    expr_log_add = w.log_add("a", "b")
    expr_log_mult = w.log_mult("a", "b")
    expr_log_div = w.log_div("a", "b")

    assert str(expr_add) == "a + b"
    assert str(expr_mult) == "a*b"
    assert str(expr_div) == "a/b"
    assert str(expr_log_add) == "log(exp(a) + exp(b))"
    assert str(expr_log_mult) == "a + b"
    assert str(expr_log_div) == "a - b"

    assert str(w.log("a")) == "log(a)"
    assert str(w.exp("a")) == "exp(a)"
    s = w.sigmoid("a")
    ls = w.log_sigmoid("a")
    assert "exp(-a)" in str(s)
    assert "log(1 + exp(-a))" in str(ls)

    assert str(w.neg("a")) == "1 - a"
    assert str(w.log_neg("a")) == "log(1 - exp(a))"

    assert str(w.max("a", "b")).startswith("Max(")
    assert str(w.min("a", "b")).startswith("Min(")

    assert str(w.one()) == "1"
    assert str(w.zero()) == "0"
    assert str(w.inf()) == "oo"
    assert str(w.neg_inf()) == "-oo"

def test_sympy_gradient_and_simplify():
    w = SympySymbolic(weights=[0.1, 0.9], wmap={"x":0, "y":1})
    x = w.assignment("x")
    y = w.assignment("y")
    f = x**2 + 3*y
    g = w.gradient(f, key='weights')
    assert g.shape == (2, 1)
    assert str(g[0]) == "2*x"
    assert str(g[1]) == "3"
    expr = w.simplify((x**2 + x**2))
    assert str(expr) == "2*x**2"

# def test_symengine_symbolic_mirrors_sympy():
#     w = SymengineSymbolic(weights=[0.1, 0.9], wmap={"a":0, "b":1})
#     a = w.assignment("a")
#     b = w.assignment("b")
#     assert str(a) == "a"
#     assert str(b) == "b"

#     assert str(w.add("a", "b")) == "a + b"
#     assert str(w.mult("a", "b")) == "a*b"
#     assert str(w.div("a", "b")) == "a/b"
#     assert str(w.log_add("a", "b")) == "log(exp(a) + exp(b))"
#     assert str(w.log_mult("a", "b")) == "a + b"
#     assert str(w.log_div("a", "b")) == "a - b"
#     assert str(w.neg("a")) == "1 - a"
#     assert str(w.log_neg("a")) == "log(1 - exp(a))"

#     xw = SymengineSymbolic(weights=[0.1, 0.9], wmap={"x":0, "y":1})
#     x = xw.assignment("x")
#     y = xw.assignment("y")
#     f = x**2 + 3*y
#     g = xw.gradient(f)
#     assert str(g[0]) == "2*x"
#     assert str(g[1]) == "3"