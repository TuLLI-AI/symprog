import pytest

from symprog import Parser

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def prob(program) -> float:
    return program.inference().inference.item()

# ---------------------------------------------------------------------------
# Regression: bare atom as a constraint (was silently dropped)
# ---------------------------------------------------------------------------

def test_bare_atom():
    @Parser(language="constraint")
    def prog():
        a: 0.9  # type: ignore
        a  # type: ignore

    assert prob(prog) == pytest.approx(0.9, abs=1e-5)


def test_bare_atom_operator_notation():
    @Parser(language="constraint")
    def prog():
        a: 0.9  # type: ignore
        And(a, a)  # type: ignore
        Or(a, a)   # type: ignore

    assert prob(prog) == pytest.approx(0.9, abs=1e-5)


# ---------------------------------------------------------------------------
# Basic propositional formulas
# ---------------------------------------------------------------------------

def test_negation():
    @Parser(language="constraint")
    def prog():
        a: 0.9  # type: ignore
        not a  # type: ignore

    assert prob(prog) == pytest.approx(0.1, abs=1e-5)


def test_conjunction():
    @Parser(language="constraint")
    def prog():
        a: 0.9  # type: ignore
        b: 0.7  # type: ignore
        a and b  # type: ignore

    assert prob(prog) == pytest.approx(0.9 * 0.7, abs=1e-5)


def test_disjunction():
    @Parser(language="constraint")
    def prog():
        a: 0.9  # type: ignore
        b: 0.7  # type: ignore
        a or b  # type: ignore

    assert prob(prog) == pytest.approx(1 - 0.1 * 0.3, abs=1e-5)


def test_tautology():
    @Parser(language="constraint")
    def prog():
        a: 0.9  # type: ignore
        a or not a  # type: ignore

    assert prob(prog) == pytest.approx(1.0, abs=1e-5)


def test_contradiction():
    @Parser(language="constraint")
    def prog():
        a: 0.9  # type: ignore
        a and not a  # type: ignore

    assert prob(prog) == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Operator notation: Implies, Biconditional, Xor, Not
# ---------------------------------------------------------------------------

def test_implies():
    # P(not a or b) = P(not a) + P(a)*P(b) = 0.1 + 0.9*0.7 = 0.73
    @Parser(language="constraint")
    def prog():
        a: 0.9  # type: ignore
        b: 0.7  # type: ignore
        Implies(a, b)  # type: ignore

    assert prob(prog) == pytest.approx(0.1 + 0.9 * 0.7, abs=1e-5)


def test_biconditional():
    # P(a<=>b) = P(a)*P(b) + P(~a)*P(~b) = 0.81 + 0.01 = 0.82
    @Parser(language="constraint")
    def prog():
        a: 0.9  # type: ignore
        b: 0.9  # type: ignore
        Biconditional(a, b)  # type: ignore

    assert prob(prog) == pytest.approx(0.81 + 0.01, abs=1e-5)


def test_xor_binary():
    # P(a xor b) = P(a)*P(~b) + P(~a)*P(b) = 0.9*0.3 + 0.1*0.7 = 0.34
    @Parser(language="constraint")
    def prog():
        a: 0.9  # type: ignore
        b: 0.7  # type: ignore
        Xor(a, b)  # type: ignore

    assert prob(prog) == pytest.approx(0.9 * 0.3 + 0.1 * 0.7, abs=1e-5)


def test_xor_ternary():
    # P(exactly one of a,b,c) = 0.9*0.3*0.5 + 0.1*0.7*0.5 + 0.1*0.3*0.5 = 0.185
    @Parser(language="constraint")
    def prog():
        a: 0.9  # type: ignore
        b: 0.7  # type: ignore
        c: 0.5  # type: ignore
        Xor(a, b, c)  # type: ignore

    expected = 0.9 * 0.3 * 0.5 + 0.1 * 0.7 * 0.5 + 0.1 * 0.3 * 0.5
    assert prob(prog) == pytest.approx(expected, abs=1e-5)


def test_not_operator():
    @Parser(language="constraint")
    def prog():
        a: 0.9  # type: ignore
        Not(a)  # type: ignore

    assert prob(prog) == pytest.approx(0.1, abs=1e-5)


# ---------------------------------------------------------------------------
# Infix / operator-notation equivalence
# ---------------------------------------------------------------------------

def test_and_infix_vs_operator():
    @Parser(language="constraint")
    def prog_infix():
        a: 0.9  # type: ignore
        b: 0.7  # type: ignore
        a and b  # type: ignore

    @Parser(language="constraint")
    def prog_op():
        a: 0.9  # type: ignore
        b: 0.7  # type: ignore
        And(a, b)  # type: ignore

    assert prob(prog_infix) == pytest.approx(prob(prog_op), abs=1e-6)


def test_or_infix_vs_operator():
    @Parser(language="constraint")
    def prog_infix():
        a: 0.9  # type: ignore
        b: 0.7  # type: ignore
        a or b  # type: ignore

    @Parser(language="constraint")
    def prog_op():
        a: 0.9  # type: ignore
        b: 0.7  # type: ignore
        Or(a, b)  # type: ignore

    assert prob(prog_infix) == pytest.approx(prob(prog_op), abs=1e-6)


def test_not_infix_vs_operator():
    @Parser(language="constraint")
    def prog_infix():
        a: 0.9  # type: ignore
        not a  # type: ignore

    @Parser(language="constraint")
    def prog_op():
        a: 0.9  # type: ignore
        Not(a)  # type: ignore

    assert prob(prog_infix) == pytest.approx(prob(prog_op), abs=1e-6)


# ---------------------------------------------------------------------------
# Multiple constraints (implicit conjunction across lines)
# ---------------------------------------------------------------------------

def test_multiple_constraints():
    @Parser(language="constraint")
    def prog():
        a: 0.9  # type: ignore
        b: 0.7  # type: ignore
        a  # type: ignore
        b  # type: ignore

    # two separate lines → P(a and b) = 0.9 * 0.7
    assert prob(prog) == pytest.approx(0.9 * 0.7, abs=1e-5)


def test_multiple_constraints_equals_conjunction():
    @Parser(language="constraint")
    def prog_split():
        a: 0.9  # type: ignore
        b: 0.7  # type: ignore
        a  # type: ignore
        b  # type: ignore

    @Parser(language="constraint")
    def prog_conjoined():
        a: 0.9  # type: ignore
        b: 0.7  # type: ignore
        a and b  # type: ignore

    assert prob(prog_split) == pytest.approx(prob(prog_conjoined), abs=1e-6)


# ---------------------------------------------------------------------------
# Conditioning
# ---------------------------------------------------------------------------

def test_conditioning_self():
    """P(a | a) = 1."""
    @Parser(language="constraint")
    def prog():
        a: 0.9  # type: ignore
        a  # type: ignore
        condition: a  # type: ignore

    assert prob(prog) == pytest.approx(1.0, abs=1e-5)


def test_conditioning_independent():
    """P(a => b | a) = P(b | a) = P(b) = 0.7 for independent a, b."""
    @Parser(language="constraint")
    def prog():
        a: 0.5  # type: ignore
        b: 0.7  # type: ignore
        Implies(a, b)  # type: ignore
        condition: a  # type: ignore

    assert prob(prog) == pytest.approx(0.7, abs=1e-5)


# ---------------------------------------------------------------------------
# First-order grounding
# ---------------------------------------------------------------------------

def test_grounding_xor():
    """Xor(s(X)) grounds to Xor(s__a, s__b)."""
    @Parser(language="constraint")
    def prog():
        a: X  # type: ignore
        b: X  # type: ignore
        s: {a: 0.9, b: 0.7}  # type: ignore
        Xor(s(X))  # type: ignore

    # P(Xor(s__a, s__b)) = 0.9*0.3 + 0.1*0.7 = 0.34
    assert prob(prog) == pytest.approx(0.9 * 0.3 + 0.1 * 0.7, abs=1e-5)


def test_grounding_and():
    """And(s(X)) grounds to And(s__a, s__b)."""
    @Parser(language="constraint")
    def prog():
        a: X  # type: ignore
        b: X  # type: ignore
        s: {a: 0.9, b: 0.7}  # type: ignore
        And(s(X))  # type: ignore

    assert prob(prog) == pytest.approx(0.9 * 0.7, abs=1e-5)

def test_grounding_single():
    """And(s(X)) grounds to And(s__a)."""
    @Parser(language="constraint")
    def prog():
        a: X  # type: ignore
        s: {a: 0.9}  # type: ignore
        And(s(X))  # type: ignore

    assert prob(prog) == pytest.approx(0.9, abs=1e-5)


def test_grounding_or():
    """Or(s(X)) grounds to Or(s__a, s__b)."""
    @Parser(language="constraint")
    def prog():
        a: X  # type: ignore
        b: X  # type: ignore
        s: {a: 0.9, b: 0.7}  # type: ignore
        Or(s(X))  # type: ignore

    assert prob(prog) == pytest.approx(1 - 0.1 * 0.3, abs=1e-5)


def test_grounding_aggregate_condition():
    """condition: Xor(s(X)) grounds to a single Xor over the domain, not one per element."""
    @Parser(language="constraint")
    def prog():
        a: X  # type: ignore
        b: X  # type: ignore
        s: {a: 0.9, b: 0.7}  # type: ignore
        t: {a: 1.0, b: 0.0}  # type: ignore
        Or(s(X) and t(X))  # type: ignore
        condition: Xor(s(X))  # type: ignore

    # P(s__a | exactly one of s__a, s__b) = 0.9*0.3 / (0.9*0.3 + 0.1*0.7)
    assert prob(prog) == pytest.approx(0.27 / 0.34, abs=1e-5)


def test_grounding_constant_atom():
    """A body atom on a domain constant, s(a), grounds to s__a (was silently dropped)."""
    @Parser(language="constraint")
    def prog():
        a: X  # type: ignore
        b: X  # type: ignore
        s: {a: 0.9, b: 0.7}  # type: ignore
        s(a)  # type: ignore

    assert prob(prog) == pytest.approx(0.9, abs=1e-5)


def test_grounding_constant_formula_and_condition():
    """Constants ground inside formulas and conditions alike."""
    @Parser(language="constraint")
    def prog():
        a: X  # type: ignore
        b: X  # type: ignore
        s: {a: 0.9, b: 0.7}  # type: ignore
        s(a) and not s(b)  # type: ignore
        condition: s(a) or s(b)  # type: ignore

    assert prob(prog) == pytest.approx(0.9 * 0.3 / (1 - 0.1 * 0.3), abs=1e-5)


def test_grounding_universal_condition():
    """condition: s(X) (no aggregate) conjoins its ground instances into one condition."""
    @Parser(language="constraint")
    def prog():
        a: X  # type: ignore
        b: X  # type: ignore
        s: {a: 0.9, b: 0.7}  # type: ignore
        t: {a: 0.5, b: 0.5}  # type: ignore
        And(t(X))  # type: ignore
        condition: Implies(s(X), t(X))  # type: ignore

    # P(t_a & t_b | (s_a -> t_a) & (s_b -> t_b)), each element independent:
    # P(t | s -> t) = 0.5 / (1 - p_s * 0.5)
    expected = (0.5 / (1 - 0.9 * 0.5)) * (0.5 / (1 - 0.7 * 0.5))
    assert prob(prog) == pytest.approx(expected, abs=1e-5)


# ---------------------------------------------------------------------------
# Invalid programs
# ---------------------------------------------------------------------------

def test_import_raises():
    with pytest.raises((ValueError, Exception)):
        @Parser(language="constraint")
        def prog():
            import os  # type: ignore


def test_non_float_weight_raises():
    with pytest.raises((ValueError, Exception)):
        @Parser(language="constraint")
        def prog():
            a: 1  # type: ignore  # int, not float
            a  # type: ignore


def test_nested_function_raises():
    with pytest.raises((ValueError, Exception)):
        @Parser(language="constraint")
        def prog():
            a: 0.9  # type: ignore
            def helper(): pass  # type: ignore

# ---------------------------------------------------------------------------
# Programs without weights (should default to 0.5)
# ---------------------------------------------------------------------------

#### doesn't work 
# def test_no_weight():
#     @Parser(language="constraint")
#     def prog():
#         a and b  # type: ignore

#     assert prob(prog) == pytest.approx(0.25, abs=1e-5)