import itertools
import random

import pytest
import torch

from symprog import Parser
from symprog.lang.languages.brasp_compiler import compile_transformer

PROGRAMS = {
    "warm-up": ("ab", '''
prev_a = last((Qa[j] for j in before(i)), default=False)
seen_a = last((True for j in before(i) if Qa[j]), default=False)
b_after_a = Qb[i] and seen_a[i]
x = last((prev_a[j] for j in before(i) if Qb[j]), default=False)
return x
'''),
    "scores of i and j": ("abc", '''
s = last((Qc[j] for j in before(i) if (Qa[i] and Qb[j]) or (Qb[i] and Qa[j])), default=True)
t = first((s[j] for j in upto(i) if Qc[j] or s[j]), default=False)
y = t[i] if Qa[i] else not s[i]
return y
'''),
    "mod in scores and values": ("ab", '''
odd_a = last((j % 3 == 1 for j in before(i) if Qa[j] and j % 2 == 0), default=i % 2 == 1)
even = i % 2 == 0
y = odd_a[i] and even[i]
return y
'''),
    "value at i, default at i": ("ab", '''
v = last((Qa[i] and Qb[j] for j in upto(i) if Qa[j] or Qb[j]), default=True)
w = last((v[j] for j in before(i) if not v[j]), default=Qa[i])
return w
'''),
    "dyck depth 2 (future masks)": ("lr", '''
P_l = last((Ql[j] for j in before(i)), default=False)
S_r = first((Qr[j] for j in after(i)), default=False)
I = (Ql[i] and S_r[i]) or (P_l[i] and Qr[i])
A_r = first((Qr[j] for j in after(i) if not I[j]), default=False)
B_l = last((Ql[j] for j in before(i) if not I[j]), default=False)
C = I[i] or (Ql[i] and A_r[i]) or (Qr[i] and B_l[i])
Y = last((False for j in everywhere(i) if not C[j]), default=True)
return Y
'''),
    "heads inside an expression": ("ab", '''
y = last((Qa[j] for j in before(i)), default=False) and not first((Qb[j] for j in onwards(i)), default=True)
return y
'''),
}


def parse(name):
    alphabet, body = PROGRAMS[name]
    source = "def prog(i):\n" + "\n".join("    " + line for line in body.strip().splitlines())
    return Parser(language="brasp", alphabet=alphabet).parse(source)


def words(alphabet, n):
    return [w for k in range(1, n + 1) for w in map("".join, itertools.product(alphabet, repeat=k))]


@pytest.mark.parametrize("name", PROGRAMS)
def test_compiled_transformer_runs_the_program(name):
    prog = parse(name)
    model = compile_transformer(prog, max_length=7)
    n = 7 if len(prog.language.alphabet) == 2 else 5
    assert model.check(prog, words(prog.language.alphabet, n)) == []


@pytest.mark.parametrize("name", ["warm-up", "dyck depth 2 (future masks)"])
def test_long_words(name):
    prog = parse(name)
    model = compile_transformer(prog, max_length=64)
    random.seed(0)
    sample = ["".join(random.choice(prog.language.alphabet) for _ in range(random.randint(40, 64)))
              for _ in range(30)]
    assert model.check(prog, sample) == []


def test_fidelity_and_readings():
    prog = parse("warm-up")
    model = compile_transformer(prog)
    fidelity = model.fidelity(prog, words("ab", 5))
    assert set(fidelity) >= {"Qa", "Qb", "prev_a", "seen_a", "b_after_a", "x"}
    assert min(fidelity.values()) > 0.999

    reading = model.reading("abba")
    assert [round(p) for p in reading.probability["x"]] == [0, 0, 1, 0]
    ## x attends to the most recent earlier b, or to BOS (no match)
    assert reading.attention["x"].argmax(-1).tolist() == [0, 0, 0, 2, 3]
    ## the attention of each head is a distribution over BOS and the visible positions
    for pattern in reading.attention.values():
        assert torch.allclose(pattern.sum(-1), torch.ones(5))


def test_cpt():
    prog = parse("warm-up")
    model = compile_transformer(prog)
    rows = model.cpt("x")
    ## no match: the default (the copied value does not matter); a match: the copied value
    assert {(r["x.found"], r["x.prev_a@j"], r["program"]) for r in rows} == {(0, None, 0), (1, 0, 0), (1, 1, 1)}
    for r in rows:
        assert abs(r["Pr(x = 1)"] - r["program"]) < 1e-3
    ## an FFN combining vectors of the same layer
    rows = model.cpt("b_after_a")
    assert all(abs(r["Pr(b_after_a = 1)"] - r["program"]) < 1e-3 for r in rows)


def test_trainable():
    prog = parse("warm-up")
    model = compile_transformer(prog)
    x = model(model.tokens(["abba", "baab"]))
    loss = -torch.log(model.probabilities(x, ["x"])["x"]).mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_max_length():
    prog = parse("warm-up")
    model = compile_transformer(prog, max_length=4)
    with pytest.raises(ValueError, match="at most 4"):
        model.run("abbab")
