# type: ignore
import itertools
import math
import traceback
import sympy as sp
import pandas as pd
import streamlit as st
from code_editor import code_editor
from symprog import Parser

# ---------------------------------------------------------------------------
# Example programs
# ---------------------------------------------------------------------------

CONSTRAINT_EXAMPLES = {
    "Constraints": """\
    
a: 0.9 
b: 0.64
c: 0.34
d: 0.93

Implies(a, b)
And(Or(a, b), Implies(a, c))
Or(a, d)
""",
    "CNF (random 3-SAT)": """\
## a random 3-CNF: 10 variables, 35 clauses
## each line is a clause (a disjunction of literals), and the
## program is their conjunction, i.e., a formula in CNF
## in the circuit panel, the Count semiring gives the number of satisfying
## assignments (out of 2^10) and Viterbi the weight of the most likely one;
## the models panel lists the satisfying assignments

## variable weights
x1: 0.59; x2: 0.63; x3: 0.15; x4: 0.43; x5: 0.75
x6: 0.72; x7: 0.61; x8: 0.48; x9: 0.71; x10: 0.55

## clauses
not x10 or x4 or x3
not x5 or x9 or x3
not x6 or not x8 or x2
not x4 or x9 or not x8
x9 or not x1 or x2
not x1 or not x8 or x6
x4 or not x10 or x9
x2 or x10 or not x6
x5 or x9 or x10
not x9 or x4 or not x5
not x10 or x4 or not x5
not x5 or x8 or x2
x1 or x2 or x7
x4 or not x10 or x7
x6 or not x2 or x10
x4 or not x10 or not x1
x3 or not x6 or not x7
not x4 or x1 or x2
not x10 or x2 or not x7
x10 or not x1 or not x4
x4 or x1 or x9
x2 or not x4 or not x10
x9 or x8 or not x1
x5 or x6 or not x8
x4 or not x1 or not x3
not x10 or x8 or x3
not x9 or x5 or not x6
not x1 or not x8 or not x2
x4 or x8 or x6
not x10 or not x3 or x5
not x1 or not x4 or not x6
not x10 or x7 or not x1
not x1 or x3 or not x8
x8 or not x9 or not x1
not x8 or x1 or x7
""",
    "Graph coloring": """\
## 3-color a 4-node graph: a-b, a-c, b-c, c-d
## each node gets exactly one color (Xor), with soft color preferences
a_r: 0.6; a_g: 0.2; a_b: 0.2
b_r: 0.3; b_g: 0.4; b_b: 0.3
c_r: 0.5; c_g: 0.25; c_b: 0.25
d_r: 0.2; d_g: 0.2; d_b: 0.6

Xor(a_r, a_g, a_b)
Xor(b_r, b_g, b_b)
Xor(c_r, c_g, c_b)
Xor(d_r, d_g, d_b)

## neighbors get different colors
not (a_r and b_r); not (a_g and b_g); not (a_b and b_b)
not (a_r and c_r); not (a_g and c_g); not (a_b and c_b)
not (b_r and c_r); not (b_g and c_g); not (b_b and c_b)
not (c_r and d_r); not (c_g and d_g); not (c_b and d_b)
""",
    "Fault diagnosis": """\
## a small circuit: x = AND(in1, in2), y = NOT(x), out = OR(y, in3)
## each gate works with high probability; a broken gate's output is unconstrained
in1: 0.5; in2: 0.5; in3: 0.5
ok_and: 0.95; ok_not: 0.95; ok_or: 0.95
x: 0.5; y: 0.5; out: 0.5

Implies(ok_and, Biconditional(x, in1 and in2))
Implies(ok_not, Biconditional(y, not x))
Implies(ok_or, Biconditional(out, y or in3))

## observation: all inputs high, but the output is low
condition: in1 and in2 and in3 and (not out)
""",
    "Pigeonhole (unsat)": """\
## 3 pigeons, 2 holes: every pigeon in a hole, no two pigeons share one
## unsatisfiable: Count = 0, SAT = no
p1_h1: 0.5; p1_h2: 0.5
p2_h1: 0.5; p2_h2: 0.5
p3_h1: 0.5; p3_h2: 0.5

p1_h1 or p1_h2
p2_h1 or p2_h2
p3_h1 or p3_h2

not (p1_h1 and p2_h1); not (p1_h1 and p3_h1); not (p2_h1 and p3_h1)
not (p1_h2 and p2_h2); not (p1_h2 and p3_h2); not (p2_h2 and p3_h2)
""",
    "Smokers (first-order)": """\
## people
anna: P; bob: P; carl: P

## stress and smoking habits
stress: {anna: 0.3, bob: 0.6, carl: 0.1}
smokes: {anna: 0.4, bob: 0.4, carl: 0.4}
cancer: {anna: 0.2, bob: 0.2, carl: 0.2}

## stressed people smoke, and smokers get cancer
Implies(stress(P), smokes(P))
Implies(smokes(P), cancer(P))

## bob has cancer: what is the most likely explanation? (see Viterbi)
condition: cancer(bob)
""",
    "Conjunction": """\
a: 0.9
b: 0.7
a and b
""",
    "Disjunction": """\
a: 0.9
b: 0.7
a or b
""",
    "Implies": """\
a: 0.8
b: 0.6
Implies(a, b)
""",
    "Alarm (noisy-or)": """\
burglary:   0.1
earthquake: 0.2
alarm_on:   0.5

(burglary or earthquake) and alarm_on
""",
    "First-order (list)": """\
## domains 
a: X; b: X 

## ground weights 
s: {a: 0.5, b: 0.5}
f: {a: 0.5, b: 0.5}

Implies(s(X),f(X))
Xor(s(X))
""",
}

PLP_EXAMPLES = {
    "Sprinkler (explaining away)": """\
## the classic wet-grass network
cloudy: 0.5
rain[0.8] :- cloudy
rain[0.2] :- ~cloudy
sprinkler[0.1] :- cloudy
sprinkler[0.5] :- ~cloudy

wet[0.99] :- rain & sprinkler
wet[0.9] :- rain & ~sprinkler
wet[0.9] :- ~rain & sprinkler

## the grass is wet: was it the rain or the sprinkler?
evidence(wet, True)
query(rain)
query(sprinkler)
""",
    "Monty Hall": """\
## the prize is behind one of three doors; you pick door 1
prize1[1/3] | prize2[1/3] | prize3[1/3]

## the host opens a door without the prize (other than yours)
open2[0.5] :- prize1
open3 :- prize1 & ~open2
open3 :- prize2
open2 :- prize3

## the host opened door 3: should you switch to door 2?
evidence(open3, True)
query(prize1)
query(prize2)
""",
    "Medical diagnosis": """\
## diseases (priors)
flu: 0.1; cold: 0.3; covid: 0.05

## noisy-or symptom model, with a small leak
fever[0.8] :- flu
fever[0.1] :- cold
fever[0.7] :- covid
cough[0.5] :- flu
cough[0.7] :- cold
cough[0.6] :- covid
anosmia[0.5] :- covid
fever[0.01] :- leak
leak: 1.0

## the patient has a fever and a cough, but no loss of smell
evidence(fever, True)
evidence(cough, True)
evidence(anosmia, False)
query(flu)
query(cold)
query(covid)
""",
    "HMM (umbrella)": """\
## hidden weather over three days, with a sticky transition
r0: 0.5
r1[0.7] :- r0
r1[0.3] :- ~r0
r2[0.7] :- r1
r2[0.3] :- ~r1

## umbrella sightings depend on the weather
u0[0.9] :- r0
u0[0.2] :- ~r0
u1[0.9] :- r1
u1[0.2] :- ~r1
u2[0.9] :- r2
u2[0.2] :- ~r2

## observations: umbrella, umbrella, no umbrella
evidence(u0, True)
evidence(u1, True)
evidence(u2, False)
query(r0)
query(r1)
query(r2)
""",
    "Viral marketing": """\
## who trusts whom (a DAG), and who sees an ad
trusts[ann, bob]: 0.8
trusts[ann, cat]: 0.5
trusts[bob, dan]: 0.6
trusts[cat, dan]: 0.7
trusts[dan, eve]: 0.9
sees_ad[ann]: 0.6

## people buy after seeing an ad, or when someone they trust buys
buys[X, 0.7] :- sees_ad[X]
buys[Y, 0.5] :- trusts[X, Y] & buys[X]

query(buys[eve])
query(buys[dan])
""",
    "Inhibition": """\
## probabilistic facts
p1: 0.4; p2: 0.75
n1: 0.3; n2: 0.6

## weighted clauses 
p[0.6] :- p1
p[0.7] :- p2
n[0.7] :- n1
n[0.6] :- n2

y :- p
y[False] :- n ## inhibition

## observations
evidence(p1, False)
evidence(n1, False)

## query
query(y)
""",
    "Two-effect inhibition": """\
c1: 0.5; c2: 0.5

e1[0.3] :- c1
e1[0.2, False] :- c2

e2[0.4] :- c2
e2[0.2, False] :- c1

e :- e1
e :- e2

evidence(c1, True)
evidence(c2, True)
query(e)
""",
    "Bayes" : """\
## probabilistic facts
burglary: 0.7
earthquake: 0.2

## weighted clauses for alarm
alarm[0.9] :- burglary & earthquake
alarm[0.8] :- burglary & ~earthquake
alarm[0.1] :- ~burglary & earthquake

## evidence and query
evidence(alarm, True)
query(burglary)""", 
    "Bayes (first-order)" : """\
person[john] : 0.5; person[mary] : 0.7

burglary: 0.7 
earthquake: 0.2 

alarm[0.9] :- burglary & earthquake 
alarm[0.8] :- burglary & ~earthquake
alarm[0.1] :- ~burglary & earthquake

calls[X, 0.8] :- alarm & person[X]
calls[X, 0.1] :- ~alarm & person[X]

query(calls[john]) 
""", 
    "graph" : """\
edge[n1,n2]: 0.6
edge[n1,n3]: 0.1 
edge[n2,n5]: 0.4
edge[n2,n6]: 0.3
edge[n3,n4]: 0.3 
edge[n4,n5]: 0.8
edge[n5,n6]: 0.2 

path[X,Y] :- edge[X,Y]
path[X,Y] :- edge[X,Z] & path[Z,Y]

query(path[n1,n6])
""", 
    "coin" : """\
coin[c1]: 1.0; coin[c2]: 1.0; coin[c3]: 1.0; coin[c4]: 1.0
heads[C, 0.6] :- coin[C]

someHeads :- heads[_] ## wildcard
query(someHeads)
""", 
    "AD" : """\
b0[0.3] | b1[0.5] | b2[0.03]
p1 : 0.8 
p2 : 0.7

y :- b0 
y :- b1 & p1
y :- b2 & p2

query(y)""",
}

## B-RASP programs are the body of `def prog(i)`: each line defines a vector by its value
## at position i (the presets of the B-RASP simulator, https://b-rasp.github.io)
BRASP_EXAMPLES = {
    "Warm-up": """\
## every value is a Boolean vector, with one bit per position i
## Qa[i]: the symbol at i is a (one such vector per symbol)
alphabet("ab")
## inference and circuits: over all words of length 4
length(4)

## attention: last((value for j in mask if score), default=...)
## takes the rightmost position j allowed by the mask where the score holds,
## and returns the value there (or the default if there is none)

## the previous symbol is a (every earlier position matches; the rightmost is i - 1)
prev_a = last((Qa[j] for j in before(i)), default=False)

## an a appeared before (only earlier a's match the score)
seen_a = last((True for j in before(i) if Qa[j]), default=False)

## position-wise: and, or, not of vectors at the same position
b_after_a = Qb[i] and seen_a[i]

## the most recent earlier b was preceded by an a
x = last((prev_a[j] for j in before(i) if Qb[j]), default=False)

return x

query("abba")
query("aabab")
""",
    "Positions (mod)": """\
## (ab)*: a at the odd positions, b at the even ones, and an even length
## positions are 1, 2, ..., n, and the word is accepted if the output is 1 at n
alphabet("ab")
length(6)

## positional predicate: i % m == r is the same bit pattern for every word,
## here 1 0 1 0 ...; in the transformer it is a positional encoding (layer 0)
odd = i % 2 == 1

## a function of the symbol and the position: also part of the embedding
misplaced = (odd[i] and Qb[i]) or (not odd[i] and Qa[i])

## some position up to i (upto includes i itself) holds the wrong symbol
any_misplaced = last((True for j in upto(i) if misplaced[j]), default=False)

## nothing misplaced so far, and an even number of symbols
## (without i % m == r, B-RASP cannot tell even from odd lengths)
Y = not any_misplaced[i] and not odd[i]

return Y

query("abab")
query("aba")
query("abba")
""",
    "Dyck-1 of depth 2": """\
## balanced brackets (l, r), nested at most two deep
alphabet("lr")
## inference and circuits: over all words of length 6
length(6)

## the previous symbol is l, the next symbol is r
P_l = last((Ql[j] for j in before(i)), default=False)
S_r = first((Qr[j] for j in after(i)), default=False)

## immediately matched: l followed by r, or r preceded by l
I = (Ql[i] and S_r[i]) or (P_l[i] and Qr[i])

## the nearest unmatched symbols after and before i
A_r = first((Qr[j] for j in after(i) if not I[j]), default=False)
B_l = last((Ql[j] for j in before(i) if not I[j]), default=False)

## every symbol is matched (accepted when Y is true at the last position)
C = I[i] or (Ql[i] and A_r[i]) or (Qr[i] and B_l[i])
Y = last((False for j in everywhere(i) if not C[j]), default=True)
return Y

query("llrrllrlrr")
query("lrrlllrrrl")
""",
    "Stair_3": """\
## aaa, ignoring c's (a b breaks a run of a's)
alphabet("abc")
## inference and circuits: over all words of length 6
length(6)

## i is an a, and so is the most recent earlier a or b (c's are skipped)
A_2 = last((Qa[i] and Qa[j] for j in before(i) if Qa[j] or Qb[j]), default=False)
A_3 = last((Qa[i] and A_2[j] for j in before(i) if Qa[j] or Qb[j]), default=False)

## some position ends a run of three
Y = last((True for j in everywhere(i) if A_3[j]), default=False)
return Y

query("acacca")
query("aacba")
""",
    "(ab)+": """\
alphabet("ab")
## inference and circuits: over all words of length 8
length(8)

## b preceded by a, and a followed by b
AB = last((Qa[j] and Qb[i] for j in before(i)), default=False)
BA = first((Qa[i] and Qb[j] for j in after(i)), default=False)
I = AB[i] or BA[i]

## every position is in an ab pair
Y = last((False for j in everywhere(i) if not I[j]), default=True)
return Y

query("ababab")
query("abba")
""",
    "{a,b,c}*a{b,c}*ba*": """\
## some b has an a before it, and no c after it
alphabet("abc")
## inference and circuits: over all words of length 5
length(5)

P_a = last((True for j in before(i) if Qa[j]), default=False)
S_c = first((True for j in after(i) if Qc[j]), default=False)
I = Qb[i] and P_a[i] and not S_c[i]

Y = last((True for j in everywhere(i) if I[j]), default=False)
return Y

query("cabba")
query("abcb")
query("abca")
""",
    "Induction heads": """\
alphabet("abc")

## the previous symbol
PREV_a = last((Qa[j] for j in before(i)), default=False)
PREV_b = last((Qb[j] for j in before(i)), default=False)
PREV_c = last((Qc[j] for j in before(i)), default=False)

## predict the symbol that followed the most recent earlier occurrence of the current symbol
PREDICT_a = last((Qa[j] for j in before(i) if (Qa[i] and PREV_a[j]) or (Qb[i] and PREV_b[j]) or (Qc[i] and PREV_c[j])), default=False)
PREDICT_b = last((Qb[j] for j in before(i) if (Qa[i] and PREV_a[j]) or (Qb[i] and PREV_b[j]) or (Qc[i] and PREV_c[j])), default=False)
PREDICT_c = last((Qc[j] for j in before(i) if (Qa[i] and PREV_a[j]) or (Qb[i] and PREV_b[j]) or (Qc[i] and PREV_c[j])), default=False)

## the prediction made at the previous position was right (shown in the trace)
Y = last(((PREDICT_a[j] and Qa[i]) or (PREDICT_b[j] and Qb[i]) or (PREDICT_c[j] and Qc[i]) for j in before(i)), default=False)

## the output: the predicted next symbol at each position
return {"a": PREDICT_a, "b": PREDICT_b, "c": PREDICT_c}

query("abcab")
query("abcabcab")
""",
    "Retrieval": """\
## associative recall: letters and numbers alternate; at each number, output the number
## that followed the most recent earlier occurrence of the preceding letter (or ?)
alphabet("abc123")

## the previous symbol is the letter a, b, c
P_a = last((Qa[j] for j in before(i)), default=False)
P_b = last((Qb[j] for j in before(i)), default=False)
P_c = last((Qc[j] for j in before(i)), default=False)

## the most recent earlier position preceded by the same letter holds 1, 2, 3
Y_1 = last((Q1[j] for j in before(i) if (P_a[i] and P_a[j]) or (P_b[i] and P_b[j]) or (P_c[i] and P_c[j])), default=False)
Y_2 = last((Q2[j] for j in before(i) if (P_a[i] and P_a[j]) or (P_b[i] and P_b[j]) or (P_c[i] and P_c[j])), default=False)
Y_3 = last((Q3[j] for j in before(i) if (P_a[i] and P_a[j]) or (P_b[i] and P_b[j]) or (P_c[i] and P_c[j])), default=False)
unknown = (P_a[i] or P_b[i] or P_c[i]) and not (Y_1[i] or Y_2[i] or Y_3[i])
return {"1": Y_1, "2": Y_2, "3": Y_3, "?": unknown}

query("a3b2b1a2c1a1c3")
""",
}

LANGUAGE_EXAMPLES = {
    "constraint": CONSTRAINT_EXAMPLES,
    "plp": PLP_EXAMPLES,
    "brasp": BRASP_EXAMPLES,
}

LANGUAGE_LABELS = {
    "constraint": "Probabilistic Logic",
    "plp": "Probabilistic Logic Programming",
    "brasp": "B-RASP",
}

## the function each language's program is the body of
LANGUAGE_SIGNATURES = {"brasp": "prog(i)"}

## output options of each language (all of them, for languages not listed)
OUTPUT_OPTIONS = ["execute", "sample", "inference", "models", "circuit"]
LANGUAGE_OUTPUT_OPTIONS = {"brasp": ["execute", "transformer", "inference", "models", "circuit"]}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_and_run(code: str, language: str):
    """Parse *code* and return a ProbabilisticProgram, or raise."""
    indent = "    "
    indented = "\n".join(indent + line for line in code.splitlines())
    src = f"def {LANGUAGE_SIGNATURES.get(language, 'prog()')}:\n{indented}\n"
    parser = Parser(language=language)
    return parser.parse(src)


@st.cache_resource(max_entries=16, show_spinner=False)
def parsed_program(code: str, language: str):
    """`parse_and_run`, reusing the program (and its compiled circuit) across reruns."""
    return parse_and_run(code, language)


## semirings of the inference modes, for the circuit panel (and the disabled algebra panel):
## (domain, ⊕, ⊗, 0, 1, weights of ℓ and ¬ℓ, smoothing)
ALGEBRA = {
    "Prob": ("ℝ≥0", "+", "×", "0", "1", "w, 1−w", False),
    "LogProb": ("ℝ ∪ {−∞}", "logaddexp", "+", "−∞", "0", "log w, log(1−w)", False),
    "Count": ("ℕ", "+", "×", "0", "1", "1, 1", True),
    "Viterbi": ("[0, 1]", "max", "×", "0", "1", "w, 1−w", True),
    "LogViterbi": ("ℝ ∪ {−∞}", "max", "+", "−∞", "0", "log w, log(1−w)", True),
    "Fuzzy": ("[0, 1]", "max", "min", "0", "1", "w, 1−w", True),
    "Entropy": ("pairs (p, h)", "(p₁+p₂, h₁+h₂)", "(p₁p₂, p₁h₂+p₂h₁)", "(0, 0)", "(1, 0)", "(w, w log w), …", True),
    "Formula": ("formulas", "∨", "∧", "⊥", "⊤", "ℓ, ¬ℓ", False),
}

## models enumerated in the models panel (programs with more are not shown)
MAX_MODELS = 4096

## longest symbolic expression rendered as LaTeX (longer ones are shown as text)
MAX_LATEX = 2000

## largest polynomials (in sympy operations) simplified in the circuit panel
MAX_SIMPLIFY_OPS = 100

## runs of a program tried by `execute` before giving up on its evidence
MAX_EXECUTE_ATTEMPTS = 10000


def as_list(result):
    """Inference returns one result per query (or a single result without queries)."""
    return result if isinstance(result, list) else [result]


def show_value(value) -> str:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def show_expression(expr) -> None:
    text = sp.latex(expr) if isinstance(expr, sp.Basic) else str(expr)
    if len(text) <= MAX_LATEX:
        st.latex(text)
    else:
        st.code(str(expr), language="python", wrap_lines=True)


## NOTE: the algebra panel (`algebra_results` and `render_algebra`) is currently disabled in
## `render_tab`; `ALGEBRA` is still used by the circuit panel

@st.cache_resource(max_entries=16, show_spinner="Evaluating the program in each semiring…")
def algebra_results(code: str, language: str):
    """The values and symbolic expressions of a program in each semiring of `ALGEBRA`
    (cached, since symbolic Viterbi and Fuzzy expressions can take seconds)."""
    prog = parsed_program(code, language)
    queries = sorted(prog._language.query_literals) or ["formula"]
    rows, expressions = [], {}
    for mode, (domain, plus, times, zero, one, literals, smoothing) in ALGEBRA.items():
        row = {"mode": mode, "domain": domain, "⊕": plus, "⊗": times, "0": zero, "1": one,
               "ℓ, ¬ℓ": literals, "smoothing": smoothing}
        try:
            values = [show_value(r.inference) for r in as_list(prog.inference(mode=mode))]
            expressions[mode] = [r.inference for r in as_list(prog.inference(mode=mode, symbolic=True))]
        except NotImplementedError as e:
            values = ["n/a"] * len(queries)
            expressions[mode] = str(e)
        row.update(dict(zip(queries, values)))
        rows.append(row)
    return queries, rows, expressions


def render_algebra(code: str, language: str) -> None:
    """The program evaluated in each semiring of `ALGEBRA`: the same circuit, read with
    different (⊕, ⊗, 0, 1) and literal weights."""
    queries, rows, expressions = algebra_results(code, language)
    st.caption(
        "Every inference is the same circuit evaluated in a different semiring: ⊕ at the "
        "or-nodes, ⊗ at the and-nodes, and the literal weights at the leaves. Semirings "
        "with *smoothing* need every variable on every branch: a missing variable contributes "
        "ℓ ⊕ ¬ℓ, which is 1 for Prob but max(w, 1−w) for Viterbi. With queries, Prob and "
        "LogProb are conditional, P(q | evidence)."
    )
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    st.markdown("**Symbolic expressions**")
    for tab, (mode, exprs) in zip(st.tabs(list(expressions)), expressions.items()):
        with tab:
            if isinstance(exprs, str):
                st.info(exprs)
                continue
            for query, expr in zip(queries, exprs):
                if len(queries) > 1:
                    st.markdown(f"`{query}`")
                show_expression(expr)


def program_models(prog):
    """Enumerates the models of a program (including derived atoms) that agree with its
    evidence, with the weight of each model as the solver computes it: evidence variables,
    derived atoms and negated choices of annotated disjunctions have weight 1.

    Returns:
        The models (as a list of dicts), their weights and fuzzy (min) weights, or
        None if the program has more than `MAX_MODELS` models.
    """
    solver = prog.solver
    weights = dict(zip(solver.weight_names, solver.weights.parameters["weights"].tolist()))
    categoricals = solver._categoricals or set()
    evidence = prog._language.evidence_literals
    observed = {e.lstrip("~") for e in evidence}

    def literal_weight(var, value):
        if var not in weights or var in observed:
            return 1.0
        if value:
            return weights[var]
        return 1.0 if var in categoricals else 1.0 - weights[var]

    models = list(itertools.islice(solver._solver.Models(solver._formula_obj), MAX_MODELS + 1))
    if len(models) > MAX_MODELS:
        return None
    models = [m for m in models if all(holds(m, e) for e in evidence)]
    literal_weights = [[literal_weight(v, b) for v, b in m.items()] for m in models]
    return (
        models,
        [math.prod(w) for w in literal_weights],
        [min(w, default=1.0) for w in literal_weights],
    )


def holds(model: dict, literal: str) -> bool:
    return model.get(literal.lstrip("~")) == (not literal.startswith("~"))


def render_models(prog) -> None:
    """The models of the program, as a brute-force check of the circuit's results."""
    prog.inference()
    enumerated = program_models(prog)
    if enumerated is None:
        st.info(
            f"This program has more than {MAX_MODELS} models, too many to list. Avoiding this "
            "enumeration is what compiling to a circuit is for."
        )
        return
    models, weights, fuzzy = enumerated
    ## a program with a condition is encoded as `_F <-> formula`, so `_F` is queried: P(formula | condition)
    conditional = any("_F" in m for m in models)
    queries = sorted(prog._language.query_literals) or (["_F"] if conditional else [])
    label = lambda query: "formula" if query == "_F" else query
    Z = sum(weights)
    if not models and prog._language.evidence_literals:
        st.warning("No model agrees with the evidence, so conditional probabilities are undefined (nan).")
    elif not models:
        st.warning("The program is unsatisfiable: it has no models.")

    ## one summary row per query, next to the solver's results
    summary = []
    solver_results = {m: as_list(prog.inference(mode=m)) for m in ("Count", "Prob", "Viterbi", "Fuzzy")}
    for i, query in enumerate(queries or [None]):
        selected = [j for j, m in enumerate(models) if query is None or holds(m, query)]
        total = sum(weights[j] for j in selected)
        brute = {
            "Count": len(selected),
            "Prob": (total / Z if Z else float("nan")) if query else total,
            "Viterbi": max((weights[j] for j in selected), default=0.0),
            "Fuzzy": max((fuzzy[j] for j in selected), default=0.0),
        }
        solver = {m: float(r[i].inference) for m, r in solver_results.items()}
        summary.append({
            "query": label(query) if query else "formula",
            **{m: f"{v:.4g}" for m, v in brute.items()},
            "solver agrees": all(
                math.isclose(brute[m], solver[m], abs_tol=1e-4)
                or (math.isnan(brute[m]) and math.isnan(solver[m]))
                for m in brute
            ),
        })

    st.caption(
        "Each row is a model of the program (a satisfying assignment, including derived "
        "atoms) that agrees with the evidence. Its weight is the product of its literal "
        "weights (w or 1 − w; evidence, derived atoms and negated choices of annotated "
        "disjunctions count 1). Summing the weights gives Prob (divided by the total for "
        "a query), counting the models gives Count, and the largest weight gives Viterbi "
        "(the largest smallest literal weight gives Fuzzy). The circuit computes the same "
        "values without listing the models."
    )
    st.dataframe(pd.DataFrame(summary), hide_index=True)

    variables = sorted({v for m in models for v in m} - {"_F"}, key=lambda v: ("__" in v, v))
    table = pd.DataFrame([{v: m.get(v) for v in variables} for m in models], columns=variables)
    for query in queries:
        table[f"⊨ {label(query)}"] = [holds(m, query) for m in models]
    table["weight"] = weights
    table["share"] = [w / Z if Z else 0.0 for w in weights]
    table = table.sort_values("weight", ascending=False)
    st.dataframe(
        table,
        hide_index=True,
        column_config={
            "share": st.column_config.ProgressColumn(
                "share", help="weight / total weight", format="percent", min_value=0, max_value=1
            ),
        },
    )


## inference modes in the circuit panel, and those whose ⊕ is a max (their best branches are highlighted)
CIRCUIT_MODES = ["Prob", "LogProb", "Count", "Viterbi", "LogViterbi", "Fuzzy", "Formula"]
MAX_MODES = {"Viterbi", "LogViterbi", "Fuzzy"}

## largest circuits (in SDD nodes) drawn with values
MAX_CIRCUIT_NODES = 300

## longest node label in the circuit drawing
MAX_LABEL = 28

HIGHLIGHT = "#d62728"


def show_label(value) -> str:
    """The left component of a (product semiring) value, shortened for a node label."""
    if isinstance(value, tuple):
        value = value[0]
    text = show_value(value)
    return text if len(text) <= MAX_LABEL else text[: MAX_LABEL - 1] + "…"


def sdd_nodes(formula) -> list:
    """The nodes of an SDD (each once)."""
    seen, stack, out = set(), [formula], []
    while stack:
        node = stack.pop()
        if node.id in seen:
            continue
        seen.add(node.id)
        out.append(node)
        if node.is_decision():
            stack += [n for element in node.elements() for n in element]
    return out


def circuit_dot(formula, names: dict[int, str], trace: dict, mode: str, result, highlight: bool = True) -> str:
    """Draws an SDD as a circuit in graphviz dot, with the value of each node from a traced
    evaluation (see `trace` in `SddMapper.wmc_interface`): decision nodes are ⊕ over their
    elements, each element is a ⊗ of a prime and a sub. For max-based modes, the elements
    that achieve each maximum are highlighted, from the root down to the leaves."""
    plus, times = ALGEBRA[mode][1], ALGEBRA[mode][2]
    nodes, elements = trace["nodes"], trace["elements"]

    def score(value):
        value = value[0] if isinstance(value, tuple) else value
        return float(value)

    ## the elements on the best path (for max-based modes)
    best: set[tuple[int, int]] = set()
    on_path: set[int] = set()
    if mode in MAX_MODES and highlight:
        stack = [formula]
        while stack:
            node = stack.pop()
            if node.id in on_path:
                continue
            on_path.add(node.id)
            if node.is_decision():
                live = [(i, e) for i, e in enumerate(node.elements()) if not (e[0].is_false() or e[1].is_false())]
                if live:
                    i, (prime, sub) = max(live, key=lambda ie: score(elements[node.id][ie[0]]))
                    best.add((node.id, i))
                    stack += [prime, sub]

    lines = [
        "digraph circuit {",
        '  graph [rankdir=TB, nodesep=0.25, ranksep=0.35];',
        '  node [fontname="Helvetica", fontsize=11];',
        '  edge [arrowhead=none, color="#888888"];',
        f'  result [shape=box, style="rounded,filled", fillcolor="#eeeeee", label="{mode}\\n{show_label(result)}"];',
        f'  result -> n{formula.id};',
    ]
    seen = set()
    stack = [formula]
    while stack:
        node = stack.pop()
        if node.id in seen:
            continue
        seen.add(node.id)
        color = f', color="{HIGHLIGHT}", penwidth=2' if node.id in on_path else ""
        value = show_label(nodes.get(node.id, "")) if node.id in nodes else ""
        if node.is_decision():
            lines.append(f'  n{node.id} [shape=ellipse, label="{plus}\\n{value}"{color}];')
            for i, (prime, sub) in enumerate(node.elements()):
                if prime.is_false() or sub.is_false():
                    continue
                element = f"e{node.id}_{i}"
                on = (node.id, i) in best
                style = f', color="{HIGHLIGHT}", penwidth=2' if on else ""
                lines.append(
                    f'  {element} [shape=box, style=rounded, label="{times}\\n{show_label(elements[node.id][i])}"{style}];'
                )
                ## as drawn by the SDD library: dashed edges to primes, solid edges to subs
                lines.append(f"  n{node.id} -> {element} [style=solid{style}];")
                lines.append(f"  {element} -> n{prime.id} [style=dashed{style}];")
                lines.append(f"  {element} -> n{sub.id} [style=solid{style}];")
                stack += [prime, sub]
        elif node.is_literal():
            name = names[abs(node.literal)]
            literal = name if node.literal > 0 else f"¬{name}"
            lines.append(f'  n{node.id} [shape=box, style=filled, fillcolor="#f6f6f6", label="{literal}\\n{value}"{color}];')
        else:
            lines.append(f'  n{node.id} [shape=box, label="{"⊤" if node.is_true() else "⊥"}\\n{value}"{color}];')
    lines.append("}")
    return "\n".join(lines)


@st.fragment
def render_circuit(prog, language: str) -> None:
    """The compiled circuit, with the value of each node in a chosen semiring (a fragment,
    so changing the options reruns only this panel)."""
    try:
        draw_circuit(prog, language)
    except Exception:
        st.code(traceback.format_exc(), language="python")


def render_svg(dot: str) -> bytes | None:
    """Renders a dot graph as SVG, if the `graphviz` package and binaries are installed."""
    try:
        import graphviz
        return graphviz.Source(dot).pipe(format="svg")
    except Exception:
        return None


def circuit_files(solver, files: dict[str, str]) -> bytes:
    """A zip of the circuit representations: the given dot graphs (and their SVG renderings,
    when graphviz is available), the SDD and vtree in the SDD library's file formats, and the
    names of the SDD's variable ids."""
    import io
    import tempfile
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, dot in files.items():
            archive.writestr(f"{name}.dot", dot)
            svg = render_svg(dot)
            if svg is not None:
                archive.writestr(f"{name}.svg", svg)
        with tempfile.TemporaryDirectory() as tmp:
            sdd_path, vtree_path = f"{tmp}/circuit.sdd", f"{tmp}/circuit.vtree"
            solver._formula_obj.save(sdd_path.encode())
            solver._solver.manager.vtree().save(vtree_path.encode())
            archive.write(sdd_path, "circuit.sdd")
            archive.write(vtree_path, "circuit.vtree")
        names = sorted(solver._solver.symbol_map.items())
        archive.writestr("variables.txt", "".join(f"{i} {name}\n" for i, name in names))
    return buffer.getvalue()


def draw_circuit(prog, language: str) -> None:
    solver = prog.solver
    formula = solver._formula_obj
    size = len(sdd_nodes(formula))
    if size > MAX_CIRCUIT_NODES:
        st.info(f"This circuit has {size} nodes, too many to draw.")
        return

    annotated_tab, sdd_tab = st.tabs(["Annotated circuit", "SDD"])
    with annotated_tab:
        mode, annotated = draw_annotated_circuit(prog, language)
    with sdd_tab:
        sdd = solver.PrettyPrinting().dot
        st.graphviz_chart(sdd)
        st.caption(
            "The circuit as drawn by the SDD library: each + node is a decision node, a "
            "disjunction of its elements, and each × node is an element, the conjunction of a "
            "prime (dashed edge) and a sub (solid edge). Literals, ⊤ and ⊥ are drawn again "
            "wherever they are used, and elements with a ⊥ part are kept."
        )

    st.download_button(
        "Download circuit files",
        data=circuit_files(solver, {f"circuit_{mode}": annotated, "sdd": sdd}),
        file_name="circuit.zip",
        mime="application/zip",
        icon=":material/download:",
        on_click="ignore",
        key=f"circuit_download_{language}",
        help="The annotated circuit and the SDD as dot (and SVG), and the SDD and vtree files "
        "(with variables.txt naming their variable ids).",
    )


def draw_annotated_circuit(prog, language: str) -> tuple[str, str]:
    """Draws the circuit with the value of each node in a chosen semiring, and returns the
    semiring and the circuit as dot."""
    solver = prog.solver
    evidence, queries = prog._prepare_evidence()
    queries = sorted(queries)

    col_mode, col_query, col_symbolic = st.columns([2, 2, 1])
    mode = col_mode.selectbox("Semiring", CIRCUIT_MODES, key=f"circuit_mode_{language}")
    query = col_query.selectbox("Query", queries, key=f"circuit_query_{language}") if len(queries) > 1 else None
    symbolic = col_symbolic.toggle("symbolic", key=f"circuit_symbolic_{language}", disabled=mode == "Formula")

    trace: dict = {}
    out = solver.query(
        mode=mode,
        evidence=evidence,
        query={query} if query else set(queries),
        symbolic=symbolic,
        trace=trace,
    )
    names = solver._solver.symbol_map
    dot = circuit_dot(solver._formula_obj, names, trace, mode, out.inference, highlight=not symbolic)
    st.graphviz_chart(dot)

    root = trace["root"]
    caption = (
        "The circuit is evaluated bottom-up: leaves are literal weights, each ⊗ box combines "
        "a prime (dashed edge) and a sub (times the weights of variables missing from the branch, when the "
        "semiring needs smoothing), and each ⊕ node combines its boxes. Elements with a ⊥ part "
        "are not drawn (they contribute the semiring's 0)."
    )
    if mode in MAX_MODES and not symbolic:
        caption += " The highlighted path follows the best element of each ⊕, which gives the most likely model."
    if isinstance(root, tuple) and (queries or evidence or "_F" in (solver._aux_vars or set())):
        caption += (
            f" With conditioning, each node holds a pair (shown: the first component). At the "
            f"root the pair is ({show_label(root[0])}, {show_label(root[1])}): the value with "
            f"the query, and the normalizer, giving {show_label(out.inference)}."
        )
    st.caption(caption)
    return mode, dot


@st.cache_resource(max_entries=16, show_spinner="Sampling…")
def sample_results(code: str, language: str):
    """Samples of a program (cached, so reruns show the same samples)."""
    return parsed_program(code, language).sample(k=3000)


@st.cache_resource(max_entries=16, show_spinner="Building the polynomial…")
def polynomial_results(code: str, language: str):
    """The polynomials of a program (simplified, when small enough: simplifying large ones
    can take minutes) and the code of its PyTorch module."""
    prog = parsed_program(code, language)
    polynomials = [r.inference for r in as_list(prog.inference(mode="Prob", symbolic=True))]
    simplified = all(sp.count_ops(p) <= MAX_SIMPLIFY_OPS for p in polynomials)
    if simplified:
        polynomials = [p.simplify() for p in polynomials]
    return polynomials, simplified, prog.tensorize(simplify=simplified).gm.code


## python calls shown for each output option
SNIPPET_CALLS = {
    "execute": "## run the program once\nprint(prog.execute())",
    "sample": "## sample from the program\nprint(prog.sample(k=3000))",
    "inference": "## exact inference\nprint(prog.inference())",
    "circuit": (
        "## symbolic polynomial and compiled PyTorch module\n"
        "print(prog.inference(mode=\"Prob\", symbolic=True).inference)\n"
        "print(prog.tensorize(simplify=True).gm.code)"
    ),
    "algebra": (
        "## the same program in different semirings\n"
        f"for mode in {list(ALGEBRA)}:\n"
        "    try:\n"
        "        print(mode, prog.inference(mode=mode))\n"
        "    except NotImplementedError as e:  ## e.g., Entropy with queries\n"
        "        print(mode, \"n/a:\", e)\n"
        "print(prog.inference(mode=\"Viterbi\", symbolic=True))"
    ),
    "models": "## the models of the program\nfor model in prog.solver.models():\n    print(model)",
    "transformer": "## the program as a transformer: layers, attention heads and FFNs\nprint(prog.language.architecture)",
}


def brasp_outputs(output) -> dict[str, object]:
    """The output vectors of a B-RASP program, by label: one vector, a tuple, or a dict."""
    if isinstance(output, dict):
        return {str(k): v for k, v in output.items()}
    if isinstance(output, tuple):
        return {str(k): v for k, v in enumerate(output)}
    return {"output": output}


def brasp_output_sequence(outputs: dict[str, object], n: int) -> str:
    """The output of a sequence-to-sequence program at each position: the label of the output
    vector that is 1 there, `·` if none is, `{…}` if several are (labels are separated by
    spaces when they are longer than one character)."""
    symbols = []
    for i in range(1, n + 1):
        on = [label for label, vector in outputs.items() if vector[i]]
        symbols.append("·" if not on else on[0] if len(on) == 1 else "{" + ",".join(on) + "}")
    separator = "" if all(len(label) == 1 for label in outputs) else " "
    return separator.join(symbols)


def brasp_cell_style(value) -> str:
    return "background-color: rgba(44, 160, 44, 0.35)" if value == 1 else ""


def render_brasp(prog) -> None:
    """Runs a B-RASP program on each of its queries, and shows the trace: every vector
    (rows), at every position (columns), with the output vectors marked."""
    from symprog.lang.languages.brasp import Run

    language = prog.language
    if not language.queries:
        st.info('Add `query("...")` to the program to run it on a word.')
        return

    with st.expander("Python program", expanded=False):
        st.code(prog.python_prog, language="python")

    ## one expander per query (the first one open), headed by the word and its verdict
    for number, word in enumerate(language.queries):
        run = Run(language.alphabet, word)
        outputs = brasp_outputs(prog.prog(run))

        ## name the output vectors (they are vectors of the trace)
        labels = {}
        for label, vector in outputs.items():
            name = next((n for n, v in run.vectors.items() if v is vector), None)
            labels[name] = label

        header = f"Query: `{word or 'ε'}`"
        if len(outputs) == 1:
            ## a recognizer: accepted when its output is 1 at the last position
            (vector,) = outputs.values()
            accepted = len(word) > 0 and bool(vector[len(word)])
            header += " :green[accepted]" if accepted else " :red[rejected]"
        elif word:
            ## a sequence-to-sequence program: its output sequence
            header += f" → `{brasp_output_sequence(outputs, len(word))}`"

        with st.expander(header, expanded=number == 0):
            if not word:
                st.caption("The empty word has no positions.")
                continue
            if len(outputs) == 1:
                st.caption("Accepted when the output vector (▶) is 1 at the last position.")
            else:
                st.caption(
                    "The output sequence gives, at each position, the label of the output vector (▶) "
                    "that is 1 there: · if none is, {…} if several are."
                )

            rows = {
                (f"▶ {name}" if name in labels else name): [int(v) for v in vector]
                for name, vector in run.vectors.items()
            }
            columns = [f"{i} {symbol}" for i, symbol in enumerate(word, start=1)]
            table = pd.DataFrame.from_dict(rows, orient="index", columns=columns)
            st.dataframe(table.style.map(brasp_cell_style), width="stretch")
            if len(outputs) > 1:
                st.caption("Outputs: " + ", ".join(f"▶ {name} ({label})" for name, label in labels.items()))


## largest number of words a B-RASP program is run on, to check its circuit by brute force
MAX_BRASP_WORDS = 20000


def brasp_semantics_problem(prog) -> str | None:
    """Why a B-RASP program has no semantics (no length, or not a recognizer), if so."""
    language = prog.language
    if len(language.architecture.outputs) != 1:
        return ("Inference needs a recognizer, a program that returns one output vector (accepted "
                "when it is true at the last position). This program returns "
                f"{len(language.architecture.outputs)}: {', '.join(language.architecture.outputs)}.")
    if language.length is None:
        return ("Inference is over all words of a fixed length: add e.g. `length(6)` to the program. "
                "The circuit's models are then the accepted words of that length.")
    return None


def brasp_probability_style(value: float) -> str:
    return f"background-color: rgba(31, 119, 180, {0.6 * value:.2f})"


def render_brasp_inference(prog) -> None:
    """How many words of the declared length are accepted, and where each symbol occurs in
    the accepted words, computed on the circuit."""
    language = prog.language
    n, k = language.length, len(language.alphabet)
    count = int(round(prog.inference(mode="Count").inference.item()))
    prob = prog.inference(mode="Prob").inference.item()
    with st.container(horizontal=True, gap="medium"):
        st.metric("Accepted words", f"{count:,} of {k ** n:,}", help=f"Words of length {n} over {''.join(language.alphabet)}.")
        st.metric("P(accepted)", f"{prob:.4g}", help="With uniformly random, independent symbols: Count / |Σ|ⁿ.")
        st.metric("Circuit size", f"{len(sdd_nodes(prog.solver._formula_obj)):,}", help="Nodes of the compiled SDD.")
    if count == 0:
        st.warning(f"No word of length {n} is accepted, so conditional probabilities are undefined.")
        return

    ## P(symbol at position | accepted), one query per input variable
    table = pd.DataFrame(
        [[prog.inference(query=f"Q{a}_{i}").inference.item() for a in language.alphabet] for i in range(1, n + 1)],
        index=[f"position {i}" for i in range(1, n + 1)],
        columns=language.alphabet,
    )
    st.dataframe(table.style.map(brasp_probability_style).format("{:.3f}"), width="stretch")
    st.caption(
        f"P(symbol at position | accepted), over the words of length {n}: each value is the "
        f"query `Q<symbol>_<position>` on the circuit, e.g. `prog.inference(query=\"Q{language.alphabet[0]}_1\")`, "
        "without listing the accepted words."
    )


def render_brasp_models(prog) -> None:
    """The accepted words (the circuit's models), checked against running the program on
    every word of the declared length."""
    language = prog.language
    n, k = language.length, len(language.alphabet)
    prog.inference()  ## builds the circuit
    models = list(itertools.islice(prog.solver.models(), MAX_MODELS + 1))
    if len(models) > MAX_MODELS:
        st.info(f"More than {MAX_MODELS} words are accepted, too many to list.")
        return
    words = sorted(language.word(m) for m in models)

    if k ** n <= MAX_BRASP_WORDS:
        executed = set()
        for word in map("".join, itertools.product(language.alphabet, repeat=n)):
            (vector,) = brasp_outputs(prog.prog(word)).values()
            if vector[n]:
                executed.add(word)
        if executed == set(words):
            st.success(
                f"The circuit's models are exactly the words the program accepts when run on each "
                f"of the {k ** n:,} words of length {n}."
            )
        else:
            st.error(
                f"The circuit and the program disagree: {sorted(set(words) - executed)[:5]} are only "
                f"models, {sorted(executed - set(words))[:5]} are only accepted when run."
            )
    else:
        st.caption(f"Not checked by running the program: {k ** n:,} words is too many.")

    st.dataframe(
        pd.DataFrame({"accepted word": words, "probability": [1 / k ** n] * len(words)}),
        hide_index=True, width="stretch",
    )
    st.caption(f"{len(words):,} accepted words of length {n}, each with probability 1/{k}^{n}.")


def render_brasp_semantics(prog, output_options: list[str]) -> None:
    """Inference, accepted words and the circuit of a B-RASP program, over the words of the
    length the program declares."""
    if not {"inference", "models", "circuit"} & set(output_options):
        return
    problem = brasp_semantics_problem(prog)
    if problem:
        st.info(problem)
        return
    ## build the solver (the circuit) once, whichever panels are shown: it is built lazily, by
    ## the first inference, and the circuit panel reads `prog.solver` directly
    try:
        prog._build_semantics()
    except Exception:
        st.code(traceback.format_exc(), language="python")
        return
    if "inference" in output_options:
        with st.expander("Exact inference", expanded=True):
            try:
                render_brasp_inference(prog)
            except Exception:
                st.code(traceback.format_exc(), language="python")
    if "models" in output_options:
        with st.expander("Accepted words", expanded=True):
            try:
                render_brasp_models(prog)
            except Exception:
                st.code(traceback.format_exc(), language="python")
    if "circuit" in output_options:
        with st.expander("Compile circuit", expanded=False):
            render_circuit(prog, "brasp")


## colors of the transformer diagram
ATTENTION_FILL, FFN_FILL, EMBEDDING_FILL, LAYER_FILL = "#dbe9f6", "#fde9d0", "#e8f3e8", "#f7f7f7"


def dot_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def positional_encoding(predicate: str) -> str:
    from symprog.lang.languages.brasp import positional_encoding as encoding
    return encoding(predicate)


def transformer_dot(arch) -> str:
    """The transformer of a B-RASP program in graphviz dot, from the embedding (bottom) to the
    output (top): each layer has its attention heads and its FFN, and edges show which
    vectors each head reads, labeled by role (q: query, k: key, v: value, v@i: value at the
    query position, d: default, pos: positional encoding)."""
    lines = [
        "digraph transformer {",
        '  graph [rankdir=BT, nodesep=0.3, ranksep=0.45, fontname="Helvetica", fontsize=12];',
        '  node [fontname="Helvetica", fontsize=11, shape=box, style="rounded,filled"];',
        '  edge [color="#888888", fontname="Helvetica", fontsize=9, fontcolor="#555555", arrowsize=0.6];',
    ]
    unused = set(arch.unused)
    producer = {name: name for name in arch.symbols}  ## node computing each vector
    head_ids = {}
    for op in arch.operations:
        for k, head in enumerate(op.heads):
            head_ids[id(head)] = op.name if op.kind == "attention" else f"{op.name}__head{k + 1}"
        producer[op.name] = op.name

    ## layer 0: the embedding
    lines.append(f'  subgraph cluster_0 {{ label="Embedding (layer 0)"; style=filled; fillcolor="{LAYER_FILL}"; color="#cccccc";')
    for name in arch.symbols:
        lines.append(f'    "{name}" [fillcolor="{EMBEDDING_FILL}", label="{name}"];')
    for p in arch.positional:
        lines.append(f'    "{p}" [fillcolor="{EMBEDDING_FILL}", style="rounded,filled,dashed"];')
    for op in arch.ffn_in(0):
        lines.append(f'    "{op.name}" [fillcolor="{EMBEDDING_FILL}", label="{op.name}\\n{dot_text(op.source)[:40]}"{UNUSED_STYLE if op.name in unused else ""}];')
    lines.append("  }")

    ## layers 1..L: attention heads, then the FFN
    for layer in range(1, arch.layers + 1):
        lines.append(f'  subgraph cluster_{layer} {{ label="Layer {layer}"; style=filled; fillcolor="{LAYER_FILL}"; color="#cccccc";')
        heads = arch.heads_in(layer)
        if heads:
            lines.append(f'    subgraph cluster_{layer}_attention {{ label="attention ({len(heads)} head{'s' if len(heads) > 1 else ''})"; style="filled,dashed"; fillcolor=white;')
            for op, head in heads:
                arrow = "▶" if head.choose == "last" else "◀"
                title = op.name if op.kind == "attention" else f"{op.name} · head {op.heads.index(head) + 1}"
                lines.append(f'      "{head_ids[id(head)]}" [fillcolor="{ATTENTION_FILL}", label="{title}\\n{arrow} {head.mask}(i)"{UNUSED_STYLE if op.name in unused else ""}];')
            lines.append("    }")
        ffn = arch.ffn_in(layer)
        if ffn:
            lines.append(f'    subgraph cluster_{layer}_ffn {{ label="FFN"; style="filled,dashed"; fillcolor=white;')
            for op in ffn:
                lines.append(f'      "{op.name}" [fillcolor="{FFN_FILL}", label="{op.name}"{UNUSED_STYLE if op.name in unused else ""}];')
            lines.append("    }")
        lines.append("  }")

    ## edges: what each head and each FFN operation reads
    def edge(source: str, target: str, label: str = "") -> None:
        lines.append(f'  "{source}" -> "{target}"' + (f' [label="{label}"];' if label else ";"))

    for op in arch.operations:
        for head in op.heads:
            roles: dict[str, list[str]] = {}
            for role, names in [("q", head.query), ("k", head.key), ("v", head.value),
                                ("v@i", head.value_at_i), ("d", head.default)]:
                for name in names:
                    roles.setdefault(producer[name], []).append(role)
            for source, labels in roles.items():
                edge(source, head_ids[id(head)], ",".join(labels))
            for p in head.positional:
                edge(positional_encoding(p), head_ids[id(head)], "pos")
        if op.kind == "position-wise":
            for name in op.inputs:
                edge(producer[name], op.name)
            for head in op.heads:
                edge(head_ids[id(head)], op.name)
            for p in op.positional:
                edge(positional_encoding(p), op.name, "pos")

    ## the readout
    lines.append('  output [shape=box, style="rounded,filled", fillcolor="#eeeeee", label="output"];')
    for name in arch.outputs:
        edge(producer[name], "output")
    lines.append("}")
    return "\n".join(lines)


## dead code (vectors the outputs do not depend on): faded and dashed in the dataflow
UNUSED_STYLE = ', style="rounded,filled,dashed", color="#aaaaaa", fontcolor="#999999"'

## colors of the residual stream table, by the kind of stage that writes a coordinate
STAGE_FILL = {"embedding": EMBEDDING_FILL, "attention": ATTENTION_FILL, "ffn": FFN_FILL}
READ_FILL = "#fff3b0"


def residual_stream_table(arch) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The residual stream, stage by stage (rows) and coordinate by coordinate (columns):
    `add` where the stage writes the coordinate, `read` where it reads it, `·` where the
    coordinate holds a value that the stage leaves alone, and nothing before it is written.
    Also returns the cell colors."""
    stages = arch.stages()
    coordinates = arch.coordinates
    written_by = {name: stage.kind for stage in stages for name in stage.writes}
    unused = set(arch.unused)
    columns = [f"▶ {c}" if c in arch.outputs else f"{c} (unused)" if c in unused else c
               for c in coordinates]
    labels, rows, colors = [], [], []
    present: set[str] = set()
    for stage in stages:
        if stage.kind == "embedding":
            labels.append("embedding")
        elif stage.kind == "readout":
            labels.append("readout")
        else:
            labels.append(f"layer {stage.layer} · {'attention' if stage.kind == 'attention' else 'FFN'}")
        row, color = [], []
        for c in coordinates:
            if c in stage.writes:
                row.append("add")
                color.append(f"background-color: {STAGE_FILL[written_by[c]]}; font-weight: 600")
            elif c in stage.reads:
                row.append("read")
                color.append(f"background-color: {READ_FILL}")
            elif c in present:
                row.append("·")
                color.append("color: #999999")
            else:
                row.append("")
                color.append("")
        present |= set(stage.writes)
        rows.append(row)
        colors.append(color)
    return (pd.DataFrame(rows, index=labels, columns=columns),
            pd.DataFrame(colors, index=labels, columns=columns))


def render_residual_stream(arch) -> None:
    table, colors = residual_stream_table(arch)
    st.dataframe(table.style.apply(lambda _: colors, axis=None), width="stretch")
    st.caption(
        "The residual stream has one coordinate per column, 0 until a stage writes it. Each "
        "stage **adds** its results into their own coordinates (colored as in the dataflow: "
        "embedding green, attention blue, FFN orange), so the stream is the concatenation of "
        "everything computed so far. Later stages **read** coordinates by projection (the query, "
        "key and value matrices of a head, the input layer of an FFN), and · marks a value "
        "carried along untouched. ▶ marks the outputs, read by the readout."
    )


def render_transformer(prog) -> None:
    """The transformer of a B-RASP program: its size, a diagram, and its attention heads."""
    arch = prog.language.architecture
    with st.container(horizontal=True, gap="medium"):
        st.metric("Layers", arch.layers, help="The longest chain of attentions that read each other's outputs.")
        st.metric("Attention heads", len(arch.heads))
        st.metric("Max heads per layer", max((len(arch.heads_in(l)) for l in range(1, arch.layers + 1)), default=0))
        st.metric("Residual width", arch.width,
                  help="One coordinate per vector: the symbols, the positional encodings, the defined "
                       "vectors, and heads inside position-wise expressions.")

    if arch.unused:
        st.warning(
            f"Not used by the output: **{', '.join(arch.unused)}**. Nothing the output depends on "
            "reads these vectors, so a compiler would drop them (with their heads and residual "
            "coordinates). They are shown faded and dashed, as the program defines them."
        )

    dataflow, residual = st.tabs(["Dataflow", "Residual stream"])
    with dataflow:
        st.graphviz_chart(transformer_dot(arch))
        st.caption(
            "Read from the bottom up. Layer 0 is the embedding: the symbol vectors, the positional "
            "encodings, and position-wise operations of these alone. In each layer, the attention "
            "heads (blue) read the residual stream after the previous layer, and the FFN (orange) "
            "computes the position-wise operations. Edge labels give the role of each vector in a "
            "head: q (query: the score at i), k (key: the score at j), v (value at j), v@i (value at "
            "i, combined with the attended position), d (default), pos (positional encoding). "
            "▶ / ◀: rightmost / leftmost attention, with its mask. Edges that skip layers are not "
            "extra connections: they read the residual stream (see the next tab)."
        )
    with residual:
        render_residual_stream(arch)

    rows = []
    for op, head in arch.heads:
        rows.append({
            "layer": head.layer,
            "head": op.name if op.kind == "attention" else f"{op.name} · head {op.heads.index(head) + 1}",
            "attention": f"{'rightmost' if head.choose == 'last' else 'leftmost'}, {head.mask}(i)",
            "query (score at i)": ", ".join(head.query) or "—",
            "key (score at j)": ", ".join(head.key) or "—",
            "value": ", ".join(head.value + [f"{n}@i" for n in head.value_at_i]) or "constant",
            "default": ", ".join(head.default) or "constant",
            "positional": ", ".join(head.positional) or "—",
        })
    if rows:
        st.dataframe(pd.DataFrame(rows).sort_values("layer", kind="stable"), hide_index=True, width="stretch")
    ffn = [op for op in arch.operations if op.kind == "position-wise"]
    if ffn:
        st.dataframe(
            pd.DataFrame([{"layer": op.layer, "FFN operation": op.name, "computes": op.source} for op in ffn]),
            hide_index=True, width="stretch",
        )


def python_snippet(code: str, language: str, output_options: list[str]) -> str:
    """Returns *code* as an inline symprog program in Python, followed by the calls
    for the selected output options."""
    lines = code.splitlines()
    ## drop blank (or whitespace-only) lines at the start and end
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    body = "\n".join(("    " + line).rstrip() for line in lines)
    calls = [SNIPPET_CALLS[o] for o in SNIPPET_CALLS if o in output_options] or [SNIPPET_CALLS["inference"]]
    return (
        "from symprog import Parser\n\n"
        f"@Parser(language=\"{language}\")\n"
        f"def {LANGUAGE_SIGNATURES.get(language, 'prog()')}:\n"
        f"{body}\n\n"
        + "\n\n".join(calls)
        + "\n"
    )


def render_tab(language: str) -> None:
    examples = LANGUAGE_EXAMPLES[language]

    col_left, col_right = st.columns([1, 2])

    with col_left:
        choice = st.selectbox(
            "Example programs",
            list(examples.keys()),
            key=f"example_{language}",
        )

    # Sync editor when example changes
    editor_key = f"code_{language}"
    last_key = f"last_example_{language}"
    if st.session_state.get(last_key) != choice:
        st.session_state[editor_key] = examples[choice]
        st.session_state[last_key] = choice

    with col_right:
        output_options = st.multiselect(
            "Output options",
            ## "algebra" is disabled for now (see below)
            LANGUAGE_OUTPUT_OPTIONS.get(language, OUTPUT_OPTIONS),
            default=["execute"],
            key=f"output_{language}",
        )

    editor_out = code_editor(
        code=st.session_state.get(editor_key, examples[choice]),
        lang="python",
        allow_reset=True,
        response_mode="debounce",
        theme="dark",
        options={"showLineNumbers": True},
        key=f"editor_{language}",
    )
    code = editor_out["text"] or st.session_state.get(editor_key, examples[choice])

    ## buttons sized to their labels, side by side (fixed columns made the labels wrap)
    with st.container(horizontal=True, gap="small", vertical_alignment="center"):
        run = st.button("Execute", key=f"run_{language}", type="primary", icon=":material/play_arrow:", width="content")
        with st.popover("Python code", icon=":material/code:", width="content"):
            st.caption(
                "This program as inline Python (use the copy icon at the top right). "
                "Run it from a file or notebook: the decorator reads the function's source."
            )
            st.code(python_snippet(code, language, output_options), language="python")

    ## keep showing the results of an executed program (e.g., when options change), until its code changes
    executed_key = f"executed_{language}"
    if run:
        st.session_state[executed_key] = code
    if st.session_state.get(executed_key) == code:
        try:
            prog = parsed_program(code, language)
        except Exception:
            st.error("Parse error")
            st.code(traceback.format_exc(), language="python")
            return

        if "execute" in output_options and language == "brasp":
            try:
                render_brasp(prog)
            except Exception:
                st.code(traceback.format_exc(), language="python")

        if "transformer" in output_options and language == "brasp":
            with st.expander("Transformer", expanded=True):
                try:
                    render_transformer(prog)
                except Exception:
                    st.code(traceback.format_exc(), language="python")

        if language == "brasp":
            render_brasp_semantics(prog, output_options)
            return

        if "execute" in output_options:
            try:
                ## rejection sampling: retry runs that fail the program's assertions (evidence)
                for _ in range(MAX_EXECUTE_ATTEMPTS):
                    try:
                        result = prog.execute()
                        st.code(result)
                        break
                    except AssertionError:
                        pass
                else:
                    st.warning(
                        f"No run satisfied the evidence in {MAX_EXECUTE_ATTEMPTS} attempts; "
                        "the evidence may be impossible."
                    )
            except Exception:
                st.code(traceback.format_exc(), language="python")

        if "sample" in output_options:
            with st.expander("Sample", expanded=True):
                try:
                    out = sample_results(code, language)
                    attempts = float(out["attempts"])  # type: ignore[arg-type]
                    executed = float(out["executed"])  # type: ignore[arg-type]
                    return_values: dict[str, float] = out.get("return_values", {})  # type: ignore[assignment]

                    #st.metric("Execution rate", f"{executed / attempts:.1%}")

                    try:
                        with st.expander("Sampler", expanded=False):
                            ground = prog.python_prog
                            st.code(ground)
                    except Exception:
                        st.code(traceback.format_exc(), language="python")

                    ### print return values 
                    if return_values:
                        denom = sum(return_values.values())
                        df = pd.DataFrame(
                            {
                                "return value": list(return_values.keys()),
                                "probability": [v / denom for v in return_values.values()],
                            }
                        )
                        st.bar_chart(df.set_index("return value")["probability"])
                except Exception:
                    st.code(traceback.format_exc(), language="python")

        if "inference" in output_options or "circuit" in output_options:
            with st.expander("Exact inference", expanded=True) as inference:
                try:
                    result = prog.inference()

                    if not isinstance(result, list) and hasattr(result, "inference"):
                        marg = result.details.get("marginal", "unk")
                        if marg is None and language == "constraint": 
                            marg = "formula"
                        
                        prob = result.inference.item()

                        df = pd.DataFrame({"variable": [marg], "prob": [prob]})
                        df["prob_pct"] = (df["prob"] * 100).round(1)
                        inference.dataframe(
                            df[["variable", "prob_pct"]],
                            column_config={
                                "prob_pct": st.column_config.ProgressColumn(
                                    "probability", format="%.1f%%", min_value=0, max_value=100
                                    ),
                                    "class": "Class",
                            },
                            hide_index=True,
                        )

                    elif isinstance(result, list):
                        margs = []
                        probs = []
                        for r in result:
                            marg = r.details.get("marginal", "unk")
                            prob = r.inference.item()
                            margs.append(marg)
                            probs.append(prob)
                        
                        df = pd.DataFrame({"variable": margs, "prob": probs})
                        df["prob_pct"] = (df["prob"] * 100).round(1)
                        inference.dataframe(
                            df[["variable", "prob_pct"]],
                            column_config={
                                "prob_pct": st.column_config.ProgressColumn(
                                    "probability", format="%.1f%%", min_value=0, max_value=100
                                    ),
                                    "class": "Class",
                            },
                            hide_index=True,
                        )

                    else:
                        st.write(result)
                except Exception:
                    st.code(traceback.format_exc(), language="python")

        ## the algebra panel is disabled for now (the circuit panel covers much of it)
        # if "algebra" in output_options:
        #     with st.expander("Algebra: one circuit, many semirings", expanded=True):
        #         try:
        #             render_algebra(code, language)
        #         except Exception:
        #             st.code(traceback.format_exc(), language="python")

        if "models" in output_options:
            with st.expander("Models", expanded=True):
                try:
                    render_models(prog)
                except Exception:
                    st.code(traceback.format_exc(), language="python")

        if "circuit" in output_options:
            with st.expander("Compile circuit", expanded=False):
                try:
                    render_circuit(prog, language)
                except Exception:
                    st.code(traceback.format_exc(), language="python")

                with st.expander("Boolean Semantics", expanded=False):
                    try:
                        ### boolean encoding of the semantics
                        for result in as_list(prog.inference(mode='Formula')):
                            st.latex(sp.latex(result.inference))
                        #st.write(semantics)
                    except Exception:
                        st.code(traceback.format_exc(), language="python")

                with st.expander("Polynomial", expanded=False):
                    try:
                        polynomials, simplified, module_code = polynomial_results(code, language)
                        for polynomial in polynomials:
                            show_expression(polynomial)
                        if not simplified:
                            st.caption("Not simplified: the polynomial is too large to simplify quickly.")
                        with st.expander("Pytorch", expanded=False):
                            st.code(module_code, language="python")
                    except Exception:
                        st.code(traceback.format_exc(), language="python")

        # if "ground program" in output_options:
        #     with st.expander("Ground program", expanded=False):
        #         try:
        #             ground = prog.python_prog
        #             st.code(ground)
        #         except Exception:
        #             st.code(traceback.format_exc(), language="python")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="symprog Language Console", layout="wide")
    st.title("symprog Language Console")

    tabs = st.tabs([LANGUAGE_LABELS[lang] for lang in LANGUAGE_LABELS])
    for tab, language in zip(tabs, LANGUAGE_LABELS):
        with tab:
            render_tab(language)


if __name__ == "__main__":
    main()
