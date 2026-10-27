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
## a random 3-CNF: 20 variables, 70 clauses
## each line is a clause (a disjunction of literals), and the
## program is their conjunction, i.e., a formula in CNF
## the algebra panel gives Count (the number of satisfying assignments,
## out of 2^20) and Viterbi (the weight of the most likely one), and the
## models panel lists the satisfying assignments

## variable weights
x1: 0.59; x2: 0.63; x3: 0.15; x4: 0.43; x5: 0.75
x6: 0.72; x7: 0.61; x8: 0.48; x9: 0.71; x10: 0.55
x11: 0.84; x12: 0.37; x13: 0.74; x14: 0.27; x15: 0.46
x16: 0.27; x17: 0.22; x18: 0.89; x19: 0.42; x20: 0.78

## clauses
not x20 or not x5 or x10
not x11 or not x16 or x18
not x7 or x18 or not x16
x18 or not x1 or x3
not x1 or not x16 or x11
not x7 or x19 or x8
x15 or x3 or x19
x16 or x4 or x10
x18 or not x7 or x20
not x13 or not x11 or not x8
not x20 or x9 or not x16
x5 or x2 or x3
x13 or not x17 or not x9
not x19 or x14 or x9
x12 or not x3 or x11
x7 or not x8 or not x1
x6 or not x11 or not x14
x8 or not x2 or not x18
not x7 or not x19 or x4
x2 or not x1 or not x7
x7 or x2 or x1
x3 or not x8 or not x20
x17 or x15 or not x2
x9 or x12 or not x16
x7 or not x2 or not x6
not x20 or x15 or x6
not x17 or x10 or not x12
not x1 or not x15 or not x3
x8 or x16 or x12
not x20 or not x5 or x10
not x1 or not x7 or not x11
not x19 or x14 or not x2
not x2 or x6 or not x15
x16 or not x18 or not x1
not x15 or x2 or x14
not x3 or x5 or not x1
x7 or x1 or x19
x7 or not x4 or x20
x4 or not x16 or x13
x4 or x9 or x5
x4 or not x5 or not x9
x9 or x18 or x11
x2 or not x16 or x15
not x7 or not x13 or not x10
x12 or not x3 or not x11
not x5 or x19 or not x10
x4 or not x16 or x8
x3 or not x10 or not x13
not x18 or x16 or x19
x4 or not x16 or not x20
x5 or x6 or not x13
not x3 or not x7 or not x8
x17 or x10 or x15
not x7 or x14 or not x3
not x6 or not x14 or x7
x1 or not x17 or not x15
x9 or x7 or x2
not x4 or x7 or not x15
x20 or x16 or not x5
x17 or x16 or x11
not x7 or x18 or x8
x11 or x20 or x2
x5 or x13 or not x10
not x3 or not x17 or not x2
x15 or x11 or not x6
x12 or x17 or x13
x17 or not x3 or x14
x20 or x14 or x16
x8 or not x1 or x19
x11 or x3 or not x16
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

LANGUAGE_EXAMPLES = {
    "constraint": CONSTRAINT_EXAMPLES,
    "plp": PLP_EXAMPLES,
}

LANGUAGE_LABELS = {
    "constraint": "Probabilistic Logic",
    "plp": "Probabilistic Logic Programming",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_and_run(code: str, language: str):
    """Parse *code* and return a ProbabilisticProgram, or raise."""
    indent = "    "
    indented = "\n".join(indent + line for line in code.splitlines())
    src = f"def prog():\n{indented}\n"
    parser = Parser(language=language)
    return parser.parse(src)


@st.cache_resource(max_entries=16, show_spinner=False)
def parsed_program(code: str, language: str):
    """`parse_and_run`, reusing the program (and its compiled circuit) across reruns."""
    return parse_and_run(code, language)


## the inference modes shown in the algebra panel: (domain, ⊕, ⊗, 0, 1, weights of ℓ and ¬ℓ, smoothing)
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
}


def python_snippet(code: str, language: str, output_options: list[str]) -> str:
    """Returns *code* as an inline symprog program in Python, followed by the calls
    for the selected output options."""
    body = "\n".join(("    " + line).rstrip() for line in code.strip("\n").splitlines())
    calls = [SNIPPET_CALLS[o] for o in SNIPPET_CALLS if o in output_options] or [SNIPPET_CALLS["inference"]]
    return (
        "from symprog import Parser\n\n"
        f"@Parser(language=\"{language}\")\n"
        "def prog():\n"
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
            ["execute", "sample", "inference", "algebra", "models", "circuit"],
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

    col_run, col_python, _ = st.columns([1, 1, 6])
    with col_run:
        run = st.button("Execute", key=f"run_{language}")
    with col_python:
        with st.popover("Python code", icon=":material/code:"):
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

        if "algebra" in output_options:
            with st.expander("Algebra: one circuit, many semirings", expanded=True):
                try:
                    render_algebra(code, language)
                except Exception:
                    st.code(traceback.format_exc(), language="python")

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
    st.set_page_config(page_title="Probabilistic Language Console", layout="wide")
    st.title("Probabilistic Language Console")

    tabs = st.tabs([LANGUAGE_LABELS[lang] for lang in LANGUAGE_LABELS])
    for tab, language in zip(tabs, LANGUAGE_LABELS):
        with tab:
            render_tab(language)


if __name__ == "__main__":
    main()
