import itertools

import pytest

from symprog import Parser, Solver
from symprog.lang.languages.brasp import BraspSyntaxError, Compile, Vector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bits(vector: Vector) -> str:
    return "".join("1" if v else "0" for v in vector)


def accepted_words(program, alphabet: str, n: int) -> set[str]:
    """Compiles `program` (whose output is one vector) for length n, and returns the words
    whose output is true at the last position, by model enumeration."""
    compiled = Compile(alphabet, n)
    output = program.prog(compiled)
    if output[n] is False:
        return set()
    solver = Solver.from_kwargs(
        var_list=compiled.inputs + compiled.defined,
        var_weights={v: 1 / len(alphabet) for v in compiled.inputs},
        aux_vars=set(compiled.defined),
        categoricals=set(compiled.inputs),
    )
    solver.add(compiled.formulas + ([] if output[n] is True else [output[n]]))
    return {compiled.word(m) for m in solver.models()}


# ---------------------------------------------------------------------------
# The basic operations, on `abba`
# ---------------------------------------------------------------------------

def test_basic_operations():
    @Parser(language="brasp", alphabet="ab")
    def prog(i):
        prev_a = last((Qa[j] for j in before(i)), default=False)  # type: ignore
        seen_a = last((True for j in before(i) if Qa[j]), default=False)  # type: ignore
        b_after_a = Qb[i] and seen_a[i]  # type: ignore
        x = last((prev_a[j] for j in before(i) if Qb[j]), default=False)  # type: ignore
        return prev_a, seen_a, b_after_a, x

    prev_a, seen_a, b_after_a, x = prog.execute("abba")
    assert bits(prev_a) == "0100"
    assert bits(seen_a) == "0111"
    assert bits(b_after_a) == "0110"
    assert bits(x) == "0010"


def test_leftmost_vs_rightmost():
    @Parser(language="brasp", alphabet="ab")
    def prog(i):
        prev_a = last((Qa[j] for j in before(i)), default=False)  # type: ignore
        x_last = last((prev_a[j] for j in before(i) if Qb[j]), default=False)  # type: ignore
        x_first = first((prev_a[j] for j in before(i) if Qb[j]), default=False)  # type: ignore
        return x_last, x_first

    x_last, x_first = prog.execute("abba")
    assert bits(x_last) == "0010"
    assert bits(x_first) == "0011"  ## at 4, the leftmost earlier b is 2, preceded by a


def test_masks_and_defaults():
    @Parser(language="brasp", alphabet="ab")
    def prog(i):
        first_position = last((False for j in before(i)), default=True)  # type: ignore
        last_position = first((False for j in after(i)), default=True)  # type: ignore
        some_b = last((True for j in everywhere(i) if Qb[j]), default=False)  # type: ignore
        return first_position, last_position, some_b

    first_position, last_position, some_b = prog.execute("aab")
    assert bits(first_position) == "100"
    assert bits(last_position) == "001"
    assert bits(some_b) == "111"
    assert bits(prog.execute("aaa")[2]) == "000"


def test_mod_and_if_else():
    @Parser(language="brasp", alphabet="ab")
    def prog(i):
        odd = i % 2 == 1  # type: ignore
        y = Qa[i] if odd[i] else Qb[i]  # type: ignore
        odd_a = last((True for j in before(i) if Qa[j] and j % 2 == 1), default=False)  # type: ignore
        return odd, y, odd_a

    odd, y, odd_a = prog.execute("abab")
    assert bits(odd) == "1010"
    assert bits(y) == "1111"
    assert bits(odd_a) == "0111"


def test_binary_score():
    """A score that mentions both positions: the most recent earlier occurrence of my symbol."""
    @Parser(language="brasp", alphabet="ab")
    def prog(i):
        prev_same = last((True for j in before(i) if (Qa[i] and Qa[j]) or (Qb[i] and Qb[j])), default=False)  # type: ignore
        return prev_same

    assert bits(prog.execute("abba")) == "0011"


def test_empty_word_and_unknown_symbols():
    @Parser(language="brasp", alphabet="ab")
    def prog(i):
        y = not Qa[i]  # type: ignore
        return y

    assert len(prog.execute("")) == 0
    with pytest.raises(ValueError, match="not in the alphabet"):
        prog.execute("abc")


# ---------------------------------------------------------------------------
# Examples from Yang, Chiang and Angluin (2024)
# ---------------------------------------------------------------------------

def dyck_1_2():
    @Parser(language="brasp", alphabet="lr")
    def prog(i):
        Pl = last((Ql[j] for j in before(i)), default=False)  # type: ignore
        Sr = first((Qr[j] for j in after(i)), default=False)  # type: ignore
        I = (Ql[i] and Sr[i]) or (Qr[i] and Pl[i])  # type: ignore
        Bl = last((Ql[j] for j in before(i) if not I[j]), default=False)  # type: ignore
        Ar = first((Qr[j] for j in after(i) if not I[j]), default=False)  # type: ignore
        C = I[i] or (Ql[i] and Ar[i]) or (Qr[i] and Bl[i])  # type: ignore
        Y = last((False for j in everywhere(i) if not C[j]), default=True)  # type: ignore
        return {"Pl": Pl, "Sr": Sr, "I": I, "Bl": Bl, "Ar": Ar, "C": C, "Y": Y}
    return prog


def in_dyck_1_2(word: str) -> bool:
    depth = 0
    for c in word:
        depth += 1 if c == "l" else -1
        if not 0 <= depth <= 2:
            return False
    return depth == 0


def test_dyck_figure_2():
    figure_2 = {
        "llrrllrlrr": {"Pl": "0110011010", "Sr": "0110010110", "I": "0110011110", "Bl": "0111011111",
                       "Ar": "1110111110", "C": "1111111111", "Y": "1111111111"},
        "lrrlllrrrl": {"Pl": "0100111000", "Sr": "1100011100", "I": "1100011000", "Bl": "0000111100",
                       "Ar": "1100111100", "C": "1100111100", "Y": "0000000000"},
    }
    prog = dyck_1_2()
    for word, rows in figure_2.items():
        vectors = prog.execute(word)
        assert {name: bits(vectors[name]) for name in rows} == rows


def test_dyck_all_words():
    prog = dyck_1_2()
    for n in range(1, 11):
        for word in map("".join, itertools.product("lr", repeat=n)):
            assert prog.execute(word)["Y"][n] == in_dyck_1_2(word)


def test_associative_recall():
    @Parser(language="brasp", alphabet="abc123")
    def prog(i):
        ## Px: the previous symbol is the letter x
        Pa = last((Qa[j] for j in before(i)), default=False)  # type: ignore
        Pb = last((Qb[j] for j in before(i)), default=False)  # type: ignore
        Pc = last((Qc[j] for j in before(i)), default=False)  # type: ignore
        ## Yd: the number after the most recent earlier occurrence of the same letter is d
        Y1 = last((Q1[j] for j in before(i) if (Pa[i] and Pa[j]) or (Pb[i] and Pb[j]) or (Pc[i] and Pc[j])), default=False)  # type: ignore
        Y2 = last((Q2[j] for j in before(i) if (Pa[i] and Pa[j]) or (Pb[i] and Pb[j]) or (Pc[i] and Pc[j])), default=False)  # type: ignore
        Y3 = last((Q3[j] for j in before(i) if (Pa[i] and Pa[j]) or (Pb[i] and Pb[j]) or (Pc[i] and Pc[j])), default=False)  # type: ignore
        unknown = (Pa[i] or Pb[i] or Pc[i]) and not (Y1[i] or Y2[i] or Y3[i])  # type: ignore
        return {"a": Qa, "b": Qb, "c": Qc, "1": Y1, "2": Y2, "3": Y3, "?": unknown}  # type: ignore

    word = "a3b2b1a2c1a1c3"
    outputs = prog.execute(word)
    result = "".join(next(s for s, v in outputs.items() if v[i]) for i in range(1, len(word) + 1))
    assert result == "a?b?b2a3c?a2c1"


# ---------------------------------------------------------------------------
# The same program, compiled for all words of a length
# ---------------------------------------------------------------------------

def test_compile_matches_execute():
    @Parser(language="brasp", alphabet="lr")
    def prog(i):
        Pl = last((Ql[j] for j in before(i)), default=False)  # type: ignore
        Sr = first((Qr[j] for j in after(i)), default=False)  # type: ignore
        I = (Ql[i] and Sr[i]) or (Qr[i] and Pl[i])  # type: ignore
        Bl = last((Ql[j] for j in before(i) if not I[j]), default=False)  # type: ignore
        Ar = first((Qr[j] for j in after(i) if not I[j]), default=False)  # type: ignore
        C = I[i] or (Ql[i] and Ar[i]) or (Qr[i] and Bl[i])  # type: ignore
        Y = last((False for j in everywhere(i) if not C[j]), default=True)  # type: ignore
        return Y

    for n in range(1, 9):
        expected = {w for w in map("".join, itertools.product("lr", repeat=n)) if prog.execute(w)[n]}
        assert accepted_words(prog, "lr", n) == expected


def test_compile_with_mod_and_constants():
    @Parser(language="brasp", alphabet="ab")
    def prog(i):
        even = i % 2 == 0  # type: ignore
        y = last((Qa[j] for j in before(i) if j % 2 == 1), default=False) and even[i]  # type: ignore
        return y

    for n in range(1, 6):
        expected = {w for w in map("".join, itertools.product("ab", repeat=n)) if prog.execute(w)[n]}
        assert accepted_words(prog, "ab", n) == expected


# ---------------------------------------------------------------------------
# Programs that are not B-RASP
# ---------------------------------------------------------------------------

def test_errors():
    with pytest.raises(BraspSyntaxError, match="not defined"):
        @Parser(language="brasp", alphabet="ab")
        def undefined(i):
            y = z[i]  # type: ignore
            return y

    with pytest.raises(BraspSyntaxError, match="already defined"):
        @Parser(language="brasp", alphabet="ab")
        def redefined(i):
            y = Qa[i]  # type: ignore
            y = Qb[i]  # type: ignore
            return y

    with pytest.raises(BraspSyntaxError, match="not defined"):
        @Parser(language="brasp", alphabet="ab")
        def self_reference(i):
            y = last((y[j] for j in before(i)), default=False)  # type: ignore
            return y

    with pytest.raises(BraspSyntaxError, match="index the vector"):
        @Parser(language="brasp", alphabet="ab")
        def unindexed(i):
            y = Qa  # type: ignore
            return y

    with pytest.raises(BraspSyntaxError, match="position variable in scope"):
        @Parser(language="brasp", alphabet="ab")
        def j_outside_attention(i):
            y = Qa[j]  # type: ignore
            return y

    with pytest.raises(BraspSyntaxError, match="position variable in scope"):
        @Parser(language="brasp", alphabet="ab")
        def shifted_index(i):
            y = Qa[i - 1]  # type: ignore
            return y

    with pytest.raises(BraspSyntaxError, match="not part of B-RASP"):
        @Parser(language="brasp", alphabet="ab")
        def arithmetic(i):
            y = Qa[i] + Qb[i]  # type: ignore
            return y

    with pytest.raises(BraspSyntaxError, match="cannot be nested"):
        @Parser(language="brasp", alphabet="ab")
        def nested(i):
            y = last((last((Qa[k] for k in before(j)), default=False) for j in before(i)), default=False)  # type: ignore
            return y

    with pytest.raises(BraspSyntaxError, match="default="):
        @Parser(language="brasp", alphabet="ab")
        def no_default(i):
            y = last(Qa[j] for j in before(i))  # type: ignore
            return y

    with pytest.raises(BraspSyntaxError, match="the mask"):
        @Parser(language="brasp", alphabet="ab")
        def bad_mask(i):
            y = last((Qa[j] for j in range(i)), default=False)  # type: ignore
            return y

    with pytest.raises(BraspSyntaxError, match="i % m == r"):
        @Parser(language="brasp", alphabet="ab")
        def bad_comparison(i):
            y = i == 1  # type: ignore
            return y

    with pytest.raises(BraspSyntaxError, match="must `return`"):
        @Parser(language="brasp", alphabet="ab")
        def no_return(i):
            y = Qa[i]  # type: ignore

    with pytest.raises(BraspSyntaxError, match="declare the alphabet"):
        @Parser(language="brasp")
        def no_alphabet(i):
            y = Qa[i]  # type: ignore
            return y


# ---------------------------------------------------------------------------
# alphabet(...) and query(...) in the program
# ---------------------------------------------------------------------------

def test_alphabet_and_queries():
    @Parser(language="brasp")
    def prog(i):
        alphabet("ab")  # type: ignore
        prev_a = last((Qa[j] for j in before(i)), default=False)  # type: ignore
        return prev_a
        query("abba")  # type: ignore
        query("")  # type: ignore

    assert prog.language.alphabet == ["a", "b"]
    assert prog.language.queries == ["abba", ""]
    results = prog.execute()
    assert list(results) == ["abba", ""]
    assert bits(results["abba"]) == "0100" and len(results[""]) == 0
    assert bits(prog.execute("ab")) == "01"  ## a word still overrides the queries


def test_trace_of_all_vectors():
    from symprog.lang.languages.brasp import Run

    @Parser(language="brasp")
    def prog(i):
        alphabet("ab")  # type: ignore
        prev_a = last((Qa[j] for j in before(i)), default=False)  # type: ignore
        y = Qb[i] and prev_a[i]  # type: ignore
        return y

    run = Run(prog.language.alphabet, "abab")
    prog.prog(run)
    assert list(run.vectors) == ["Qa", "Qb", "prev_a", "y"]
    assert bits(run.vectors["y"]) == "0101"


def test_declaration_errors():
    with pytest.raises(ValueError, match="no input"):
        @Parser(language="brasp", alphabet="ab")
        def no_queries(i):
            y = Qa[i]  # type: ignore
            return y
        no_queries.execute()

    with pytest.raises(BraspSyntaxError, match="not in the alphabet"):
        @Parser(language="brasp")
        def bad_query(i):
            alphabet("ab")  # type: ignore
            y = Qa[i]  # type: ignore
            return y
            query("abc")  # type: ignore

    with pytest.raises(BraspSyntaxError, match="different alphabet"):
        @Parser(language="brasp", alphabet="ab")
        def two_alphabets(i):
            alphabet("abc")  # type: ignore
            y = Qa[i]  # type: ignore
            return y

    with pytest.raises(BraspSyntaxError, match="before defining vectors"):
        @Parser(language="brasp")
        def late_alphabet(i):
            y = Qa[i]  # type: ignore
            alphabet("ab")  # type: ignore
            return y

    with pytest.raises(BraspSyntaxError, match="distinct symbols"):
        @Parser(language="brasp")
        def repeated_symbols(i):
            alphabet("aab")  # type: ignore
            y = Qa[i]  # type: ignore
            return y

    with pytest.raises(BraspSyntaxError, match="one `return`"):
        @Parser(language="brasp", alphabet="ab")
        def two_returns(i):
            y = Qa[i]  # type: ignore
            return y
            return y

    with pytest.raises(BraspSyntaxError, match='with one string'):
        @Parser(language="brasp", alphabet="ab")
        def query_not_string(i):
            y = Qa[i]  # type: ignore
            return y
            query(ab)  # type: ignore


def test_architecture_embedding_positional_and_inline_heads():
    @Parser(language="brasp")
    def prog(i):
        alphabet("ab")  # type: ignore
        even = i % 2 == 0  # type: ignore
        odd_a = last((True for j in before(i) if Qa[j] and j % 2 == 1), default=False)  # type: ignore
        y = last((Qa[j] for j in before(i)), default=False) and even[i]  # type: ignore
        z = last((y[j] for j in before(i)), default=Qb[i])  # type: ignore
        return z

    arch = prog.language.architecture
    layers = {op.name: (op.kind, op.layer) for op in arch.operations}
    ## a function of the position alone is part of the embedding
    assert layers["even"] == ("position-wise", 0)
    ## an attention inside a position-wise expression is a head, combined by the FFN
    assert layers["y"] == ("position-wise", 1) and len(arch.ffn_in(1)) == 1
    assert layers["odd_a"] == ("attention", 1) and layers["z"] == ("attention", 2)
    (z,) = [h for op, h in arch.heads if op.name == "z"]
    assert z.value == ["y"] and z.default == ["Qb"]
    assert arch.positional == ["pos % 2 == 0", "pos % 2 == 1"]
    ## coordinates: 2 symbols, 2 positional encodings, 4 vectors, and the head inside `y`
    assert (arch.layers, len(arch.heads), arch.width) == (2, 3, 2 + 2 + 4 + 1)

    stages = [(s.layer, s.kind, s.writes, s.reads) for s in arch.stages()]
    assert stages == [
        (0, "embedding", ["Qa", "Qb", "pos % 2 == 0", "pos % 2 == 1", "even"], []),
        (1, "attention", ["odd_a", "y · head 1"], ["Qa", "pos % 2 == 1"]),
        (1, "ffn", ["y"], ["even", "y · head 1"]),
        (2, "attention", ["z"], ["y", "Qb"]),
        (3, "readout", [], ["z"]),
    ]


def test_shared_positional_encoding():
    """`i % 2 == 0` and `j % 2 == 0` are one encoding, read at different positions."""
    @Parser(language="brasp", alphabet="ab")
    def prog(i):
        even = i % 2 == 0  # type: ignore
        even_a = last((True for j in before(i) if Qa[j] and j % 2 == 0), default=False)  # type: ignore
        return even_a

    arch = prog.language.architecture
    assert arch.positional == ["pos % 2 == 0"] and arch.width == 2 + 1 + 2


def test_builders_from_a_reloaded_module():
    """A parsed program accepts builders from a fresh copy of the module (as after Streamlit
    reloads it), whose classes differ from the ones the program was parsed with."""
    import importlib.util
    import symprog.lang.languages.brasp as brasp

    @Parser(language="brasp")
    def prog(i):
        alphabet("ab")  # type: ignore
        prev_a = last((Qa[j] for j in before(i)), default=False)  # type: ignore
        return prev_a

    import warnings
    spec = importlib.util.spec_from_file_location("brasp_reloaded", brasp.__file__)
    fresh = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():  ## the fresh copy registers the language again
        warnings.simplefilter("ignore", UserWarning)
        spec.loader.exec_module(fresh)
    assert fresh.Run is not brasp.Run

    run = fresh.Run(["a", "b"], "abba")
    assert bits(prog.prog(run)) == "0100"
    assert list(run.vectors) == ["Qa", "Qb", "prev_a"]


# ---------------------------------------------------------------------------
# length(...) and inference
# ---------------------------------------------------------------------------

def test_inference_over_a_length():
    @Parser(language="brasp")
    def prog(i):
        alphabet("ab")  # type: ignore
        length(4)  # type: ignore
        ## some a is followed (anywhere later) by a b
        later_b = first((True for j in after(i) if Qb[j]), default=False)  # type: ignore
        Y = last((True for j in everywhere(i) if Qa[j] and later_b[j]), default=False)  # type: ignore
        return Y

    expected = {w for w in map("".join, itertools.product("ab", repeat=4)) if "a" in w and "b" in w[w.index("a"):]}
    assert prog.inference(mode="Count").inference.item() == len(expected)
    assert prog.inference(mode="Prob").inference.item() == pytest.approx(len(expected) / 16)
    assert {prog.language.word(m) for m in prog.solver.models()} == expected
    ## conditional: the last symbol of an accepted word
    p_b_last = sum(w[-1] == "b" for w in expected) / len(expected)
    assert prog.inference(query="Qb_4").inference.item() == pytest.approx(p_b_last)


def test_inference_constant_outputs():
    @Parser(language="brasp", alphabet="ab", length=3)
    def never(i):
        Y = last((True for j in before(i) if Qa[j] and not Qa[j]), default=False)  # type: ignore
        return Y

    assert never.inference(mode="Count").inference.item() == 0

    @Parser(language="brasp", alphabet="ab", length=3)
    def always(i):
        Y = last((False for j in before(i) if Qa[j] and not Qa[j]), default=True)  # type: ignore
        return Y

    assert always.inference(mode="Count").inference.item() == 8
    assert always.inference(mode="Prob").inference.item() == pytest.approx(1.0)


def test_length_errors():
    @Parser(language="brasp", alphabet="ab")
    def no_length(i):
        y = Qa[i]  # type: ignore
        return y

    with pytest.raises(ValueError, match="length"):
        no_length.inference()

    with pytest.raises(BraspSyntaxError, match="length"):
        @Parser(language="brasp", alphabet="ab")
        def bad_length(i):
            length(0)  # type: ignore
            y = Qa[i]  # type: ignore
            return y

    with pytest.raises(BraspSyntaxError, match="once"):
        @Parser(language="brasp", alphabet="ab")
        def two_lengths(i):
            length(3)  # type: ignore
            length(4)  # type: ignore
            y = Qa[i]  # type: ignore
            return y


def test_unused_vectors():
    @Parser(language="brasp")
    def prog(i):
        alphabet("ab")
        prev_a = last((Qa[j] for j in before(i)), default=False)
        seen_a = last((True for j in before(i) if Qa[j]), default=False)
        b_after_a = Qb[i] and seen_a[i]
        x = last((prev_a[j] for j in before(i) if Qb[j]), default=False)
        return x

    assert prog.language.architecture.unused == ["seen_a", "b_after_a"]
