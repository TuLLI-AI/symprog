# type: ignore
"""The preset programs of the B-RASP simulator (https://b-rasp.github.io), written in the
`brasp` language, and checked against a port of the simulator (`brasp_simulator.py`) and
against the languages they recognize."""
import itertools
import re

import pytest

from symprog import Parser, Solver
from symprog.lang.languages.brasp import Compile

from .brasp_simulator import PRESETS, accepts, run

# ---------------------------------------------------------------------------
# The presets, in the `brasp` language (each returns its vectors, by simulator name)
# ---------------------------------------------------------------------------

@Parser(language="brasp", alphabet="lr")
def dyck_1_2(i):
    P_l = last((Ql[j] for j in before(i)), default=False)
    S_r = first((Qr[j] for j in after(i)), default=False)
    I = (Ql[i] and S_r[i]) or (P_l[i] and Qr[i])
    A_r = first((Qr[j] for j in after(i) if not I[j]), default=False)
    B_l = last((Ql[j] for j in before(i) if not I[j]), default=False)
    C = I[i] or (Ql[i] and A_r[i]) or (Qr[i] and B_l[i])
    Y = last((False for j in everywhere(i) if not C[j]), default=True)
    return {"P_l": P_l, "S_r": S_r, "I": I, "A_r": A_r, "B_l": B_l, "C": C, "Y": Y}


@Parser(language="brasp", alphabet="abc")
def stair_3(i):
    ## the most recent earlier a or b (c's are skipped) is an a, as is the current symbol
    A_2 = last((Qa[i] and Qa[j] for j in before(i) if Qa[j] or Qb[j]), default=False)
    A_3 = last((Qa[i] and A_2[j] for j in before(i) if Qa[j] or Qb[j]), default=False)
    Y = last((True for j in everywhere(i) if A_3[j]), default=False)
    return {"A_2": A_2, "A_3": A_3, "Y": Y}


@Parser(language="brasp", alphabet="ab")
def ab_plus(i):
    AB = last((Qa[j] and Qb[i] for j in before(i)), default=False)
    BA = first((Qa[i] and Qb[j] for j in after(i)), default=False)
    I = AB[i] or BA[i]
    Y = last((False for j in everywhere(i) if not I[j]), default=True)
    return {"AB": AB, "BA": BA, "I": I, "Y": Y}


@Parser(language="brasp", alphabet="abc")
def b_prec_a_no_succ_c(i):
    P_a = last((True for j in before(i) if Qa[j]), default=False)
    S_c = first((True for j in after(i) if Qc[j]), default=False)
    I = Qb[i] and P_a[i] and not S_c[i]
    Y = last((True for j in everywhere(i) if I[j]), default=False)
    return {"P_a": P_a, "S_c": S_c, "I": I, "Y": Y}


@Parser(language="brasp", alphabet="abc")
def induction_heads(i):
    PREV_a = last((Qa[j] for j in before(i)), default=False)
    PREV_b = last((Qb[j] for j in before(i)), default=False)
    PREV_c = last((Qc[j] for j in before(i)), default=False)
    ## the symbol that followed the most recent earlier occurrence of the current symbol
    PREDICT_a = last((Qa[j] for j in before(i) if (Qa[i] and PREV_a[j]) or (Qb[i] and PREV_b[j]) or (Qc[i] and PREV_c[j])), default=False)
    PREDICT_b = last((Qb[j] for j in before(i) if (Qa[i] and PREV_a[j]) or (Qb[i] and PREV_b[j]) or (Qc[i] and PREV_c[j])), default=False)
    PREDICT_c = last((Qc[j] for j in before(i) if (Qa[i] and PREV_a[j]) or (Qb[i] and PREV_b[j]) or (Qc[i] and PREV_c[j])), default=False)
    ## the prediction made at the previous position was right
    Y = last(((PREDICT_a[j] and Qa[i]) or (PREDICT_b[j] and Qb[i]) or (PREDICT_c[j] and Qc[i]) for j in before(i)), default=False)
    return {"PREV_a": PREV_a, "PREV_b": PREV_b, "PREV_c": PREV_c,
            "PREDICT_a": PREDICT_a, "PREDICT_b": PREDICT_b, "PREDICT_c": PREDICT_c, "Y": Y}


@Parser(language="brasp", alphabet="abc123")
def retrieval(i):
    P_a = last((Qa[j] for j in before(i)), default=False)
    P_b = last((Qb[j] for j in before(i)), default=False)
    P_c = last((Qc[j] for j in before(i)), default=False)
    Y_1 = last((Q1[j] for j in before(i) if (P_a[i] and P_a[j]) or (P_b[i] and P_b[j]) or (P_c[i] and P_c[j])), default=False)
    Y_2 = last((Q2[j] for j in before(i) if (P_a[i] and P_a[j]) or (P_b[i] and P_b[j]) or (P_c[i] and P_c[j])), default=False)
    Y_3 = last((Q3[j] for j in before(i) if (P_a[i] and P_a[j]) or (P_b[i] and P_b[j]) or (P_c[i] and P_c[j])), default=False)
    Y_unknown = (P_a[i] or P_b[i] or P_c[i]) and not (Y_1[i] or Y_2[i] or Y_3[i])
    return {"P_a": P_a, "P_b": P_b, "P_c": P_c, "Y_1": Y_1, "Y_2": Y_2, "Y_3": Y_3, "Y_?": Y_unknown}


PROGRAMS = {
    "dyck_1_2": dyck_1_2,
    "stair_3": stair_3,
    "ab_plus": ab_plus,
    "b_prec_a_no_succ_c": b_prec_a_no_succ_c,
    "induction_heads": induction_heads,
    "retrieval": retrieval,
}

## the longest words checked exhaustively
MAX_LENGTH = {"dyck_1_2": 10, "stair_3": 7, "ab_plus": 10, "b_prec_a_no_succ_c": 7,
              "induction_heads": 6, "retrieval": 4}

RECOGNIZERS = ["dyck_1_2", "stair_3", "ab_plus", "b_prec_a_no_succ_c"]


def words(alphabet: str, max_length: int):
    for n in range(max_length + 1):
        yield from map("".join, itertools.product(alphabet, repeat=n))


def accepted(program, word: str) -> bool:
    """The simulator's convention: the output `Y` is true at the last position."""
    return len(word) > 0 and program.execute(word)["Y"][len(word)]


# ---------------------------------------------------------------------------
# Same vectors as the simulator, on every short word
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(PRESETS))
def test_traces_match_simulator(name):
    alphabet, operations = PRESETS[name]
    program = PROGRAMS[name]
    for word in words(alphabet, MAX_LENGTH[name]):
        expected = run(operations, word)
        vectors = program.execute(word)
        for key, vector in vectors.items():
            assert list(vector) == expected[key], (name, word, key)


# ---------------------------------------------------------------------------
# The recognizers accept the languages they are named after
# ---------------------------------------------------------------------------

def in_dyck_1_2(word: str) -> bool:
    depth = 0
    for c in word:
        depth += 1 if c == "l" else -1
        if not 0 <= depth <= 2:
            return False
    return depth == 0

LANGUAGES = {
    "dyck_1_2": lambda w: len(w) > 0 and in_dyck_1_2(w),
    "stair_3": lambda w: re.fullmatch(r"[abc]*ac*ac*a[abc]*", w) is not None,  ## aaa, ignoring c's
    "ab_plus": lambda w: re.fullmatch(r"(ab)+", w) is not None,
    "b_prec_a_no_succ_c": lambda w: re.fullmatch(r"[abc]*a[bc]*ba*", w) is not None,
}

@pytest.mark.parametrize("name", RECOGNIZERS)
def test_recognized_languages(name):
    alphabet, operations = PRESETS[name]
    for word in words(alphabet, MAX_LENGTH[name]):
        assert accepted(PROGRAMS[name], word) == LANGUAGES[name](word) == accepts(operations, word), word


# ---------------------------------------------------------------------------
# Compiled for all words of a length, the recognizers accept the same words
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", RECOGNIZERS)
def test_compiled_recognizers(name):
    alphabet, operations = PRESETS[name]
    for n in range(1, 6):
        compiled = Compile(alphabet, n)
        Y = PROGRAMS[name].prog(compiled)["Y"]
        expected = {w for w in words(alphabet, n) if len(w) == n and accepts(operations, w)}
        if Y[n] is False:
            assert expected == set()
            continue
        solver = Solver.from_kwargs(
            var_list=compiled.inputs + compiled.defined,
            var_weights={v: 1 / len(alphabet) for v in compiled.inputs},
            aux_vars=set(compiled.defined),
            categoricals=set(compiled.inputs),
        )
        solver.add(compiled.formulas + ([] if Y[n] is True else [Y[n]]))
        assert {compiled.word(m) for m in solver.models()} == expected, (name, n)


# ---------------------------------------------------------------------------
# Non-strict masks (j <= i, i <= j): not used by the presets, but supported by the simulator
# ---------------------------------------------------------------------------

NON_STRICT = ("abc", [
    ("init", "a"),
    ("init", "b"),
    ("init", "c"),
    ("attention", "SELF_a", "rightmost", "j<=i", "1", "Q_a(j)", "0"),
    ("attention", "LAST_ab_is_a", "rightmost", "j<=i", "Q_a(j) | Q_b(j)", "Q_a(j)", "Q_c(i)"),
    ("attention", "FIRST_ab_is_b", "leftmost", "i<=j", "Q_a(j) | Q_b(j)", "Q_b(j)", "0"),
    ("attention", "FROM_a_LATER_b", "leftmost", "i<=j", "Q_b(j)", "Q_a(i)", "Q_c(i)"),
    ("attention", "Y", "rightmost", "j<=i", "FROM_a_LATER_b(j)", "LAST_ab_is_a(j) & not SELF_a(i)", "FIRST_ab_is_b(i)"),
])

@Parser(language="brasp", alphabet="abc")
def non_strict(i):
    SELF_a = last((Qa[j] for j in upto(i)), default=False)
    LAST_ab_is_a = last((Qa[j] for j in upto(i) if Qa[j] or Qb[j]), default=Qc[i])
    FIRST_ab_is_b = first((Qb[j] for j in onwards(i) if Qa[j] or Qb[j]), default=False)
    FROM_a_LATER_b = first((Qa[i] for j in onwards(i) if Qb[j]), default=Qc[i])
    Y = last((LAST_ab_is_a[j] and not SELF_a[i] for j in upto(i) if FROM_a_LATER_b[j]), default=FIRST_ab_is_b[i])
    return {"SELF_a": SELF_a, "LAST_ab_is_a": LAST_ab_is_a, "FIRST_ab_is_b": FIRST_ab_is_b,
            "FROM_a_LATER_b": FROM_a_LATER_b, "Y": Y}


def test_non_strict_masks_match_simulator():
    alphabet, operations = NON_STRICT
    for word in words(alphabet, 6):
        expected = run(operations, word)
        for key, vector in non_strict.execute(word).items():
            assert list(vector) == expected[key], (word, key)
    ## the position itself is included: SELF_a is Q_a
    assert list(non_strict.execute("abcab")["SELF_a"]) == [True, False, False, True, False]


def test_non_strict_masks_compiled():
    alphabet, operations = NON_STRICT
    for n in range(1, 6):
        compiled = Compile(alphabet, n)
        Y = non_strict.prog(compiled)["Y"]
        expected = {w for w in words(alphabet, n) if len(w) == n and accepts(operations, w)}
        if Y[n] is False:
            assert expected == set()
            continue
        solver = Solver.from_kwargs(
            var_list=compiled.inputs + compiled.defined,
            var_weights={v: 1 / len(alphabet) for v in compiled.inputs},
            aux_vars=set(compiled.defined),
            categoricals=set(compiled.inputs),
        )
        solver.add(compiled.formulas + ([] if Y[n] is True else [Y[n]]))
        assert {compiled.word(m) for m in solver.models()} == expected, n


# ---------------------------------------------------------------------------
# The sequence-to-sequence presets on example inputs
# ---------------------------------------------------------------------------

def test_retrieval_example():
    word = "a3b2b1a2c1a1c3"
    vectors = retrieval.execute(word)
    outputs = {"1": vectors["Y_1"], "2": vectors["Y_2"], "3": vectors["Y_3"], "?": vectors["Y_?"]}
    result = "".join(
        word[i - 1] if word[i - 1] in "abc" else next(s for s, v in outputs.items() if v[i])
        for i in range(1, len(word) + 1)
    )
    assert result == "a?b?b2a3c?a2c1"


def test_induction_heads_example():
    ## after "ab...a", the induction head predicts "b"
    word = "abcab"
    vectors = induction_heads.execute(word)
    assert vectors["PREDICT_b"][4] and not vectors["PREDICT_a"][4] and not vectors["PREDICT_c"][4]
    assert [vectors["Y"][i] for i in range(1, 6)] == [False, False, False, False, True]


# ---------------------------------------------------------------------------
# The presets as transformers: layers and heads
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name, layers, heads", [
    ("dyck_1_2", 3, 5),
    ("stair_3", 3, 3),
    ("ab_plus", 2, 3),
    ("b_prec_a_no_succ_c", 2, 3),
    ("induction_heads", 3, 7),  ## previous token, induction, then checking the prediction
    ("retrieval", 2, 6),
])
def test_architecture_size(name, layers, heads):
    arch = PROGRAMS[name].language.architecture
    assert (arch.layers, len(arch.heads)) == (layers, heads)


def test_architecture_of_dyck():
    arch = dyck_1_2.language.architecture
    assert {op.name: (op.kind, op.layer) for op in arch.operations} == {
        "P_l": ("attention", 1), "S_r": ("attention", 1), "I": ("position-wise", 1),
        "A_r": ("attention", 2), "B_l": ("attention", 2), "C": ("position-wise", 2),
        "Y": ("attention", 3),
    }
    (_, A_r), = [(op, h) for op, h in arch.heads if op.name == "A_r"]
    assert (A_r.choose, A_r.mask, A_r.query, A_r.key, A_r.value) == ("first", "after", [], ["I"], ["Qr"])
    assert arch.width == 2 + 7
    assert arch.outputs == ["P_l", "S_r", "I", "A_r", "B_l", "C", "Y"]


def test_architecture_roles():
    """Query and key features of the induction head; values read at i are combined after attention."""
    arch = induction_heads.language.architecture
    (_, predict_a), = [(op, h) for op, h in arch.heads if op.name == "PREDICT_a"]
    assert predict_a.query == ["Qa", "Qb", "Qc"] and predict_a.key == ["PREV_a", "PREV_b", "PREV_c"]
    assert predict_a.value == ["Qa"] and predict_a.layer == 2
    (_, y), = [(op, h) for op, h in arch.heads if op.name == "Y"]
    assert y.value == ["PREDICT_a", "PREDICT_b", "PREDICT_c"] and y.value_at_i == ["Qa", "Qb", "Qc"]


def test_residual_stream_of_dyck():
    """Each coordinate is written once, by the stage that computes it, and read later."""
    stages = [(s.layer, s.kind, s.writes) for s in dyck_1_2.language.architecture.stages()]
    assert stages == [
        (0, "embedding", ["Ql", "Qr"]),
        (1, "attention", ["P_l", "S_r"]),
        (1, "ffn", ["I"]),
        (2, "attention", ["A_r", "B_l"]),
        (2, "ffn", ["C"]),
        (3, "attention", ["Y"]),
        (4, "readout", []),
    ]
    arch = dyck_1_2.language.architecture
    written = []
    for stage in arch.stages():
        assert set(stage.reads) <= set(written)  ## only coordinates written by earlier stages
        written += stage.writes
    assert written == arch.coordinates and len(set(written)) == arch.width == 9


# ---------------------------------------------------------------------------
# Semantics: the solver's models are the accepted words of a length
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", RECOGNIZERS)
def test_semantics_match_simulator(name):
    alphabet, operations = PRESETS[name]
    program = PROGRAMS[name]
    for n in range(1, 6):
        solver = program.language.compile(program.python_prog, n, output="Y")
        expected = {w for w in words(alphabet, n) if len(w) == n and accepts(operations, w)}
        assert {program.language.word(m) for m in solver.models()} == expected, (name, n)
        assert solver.query(mode="Count").inference.item() == pytest.approx(len(expected))
        assert solver.query(mode="Prob").inference.item() == pytest.approx(len(expected) / len(alphabet) ** n)


def test_semantics_needs_a_recognizer():
    with pytest.raises(ValueError, match="needs a recognizer"):
        retrieval.language.compile(retrieval.python_prog, 4)
    with pytest.raises(ValueError, match="with the key"):
        retrieval.language.compile(retrieval.python_prog, 4, output="Y")
