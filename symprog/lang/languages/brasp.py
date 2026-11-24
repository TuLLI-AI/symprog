"""B-RASP as an embedded `symprog` language.

B-RASP (Boolean RASP) is the programming language of Yang, Chiang and Angluin,
*Masked Hard-Attention Transformers Recognize Exactly the Star-Free Languages* (NeurIPS 2024).
Every value is a Boolean vector, with one bit per position of the input word.

A program is a Python function of a position variable (here `i`). Each statement defines a
new vector by its value at position `i`:

```python
@Parser(language="brasp")
def prog(i):
    alphabet("ab")

    prev_a    = last((Qa[j] for j in before(i)), default=False)
    seen_a    = last((True for j in before(i) if Qa[j]), default=False)
    b_after_a = Qb[i] and seen_a[i]
    x         = last((prev_a[j] for j in before(i) if Qb[j]), default=False)
    return x

    query("abba")
    query("ab")

prog.execute("abba")   # Vector(0 0 1 0): runs on one word
prog.execute()         # {'abba': Vector(0 0 1 0), 'ab': Vector(0 0)}: runs on the queries
```

`alphabet("ab")` declares the symbols (one character each; alternatively,
`@Parser(language="brasp", alphabet="ab")`), and each `query("...")` adds an input word.
`length(n)` declares the length of the words for inference: `prog.inference(mode="Count")`
counts the accepted words of length n (see `BraspLanguage`).

The constructs are:

| B-RASP                              | Python                                              |
|-------------------------------------|-----------------------------------------------------|
| symbol predicate Q_a                | `Qa[i]` (one vector `Q<symbol>` per symbol)         |
| position-wise operation             | `and`, `or`, `not`, `x if c else y`                 |
| MOD^r_m                             | `i % m == r`                                        |
| rightmost / leftmost attention      | `last(...)` / `first(...)`                          |
| mask j < i / j > i / none           | `for j in before(i)` / `after(i)` / `everywhere(i)` |
| mask j <= i / j >= i                | `for j in upto(i)` / `onwards(i)`                   |
| score S(i, j)                       | `if S` (optional: no `if` means every position)     |
| value V(i, j)                       | the generator's element                             |
| default D(i)                        | `default=D` (required)                              |

`BraspTransformer` checks that a program stays within B-RASP and rewrites it into plain
Python, which calls the runtime below. The same rewritten program runs on one word (with
the `Run` builder, Python Booleans) or on all words of a length (with the `Compile`
builder, formulas over input variables).
"""
import ast
import functools
import math
import sys
from dataclasses import dataclass, field
from types import ModuleType
from typing import Callable

from symprog.solvers.solver import Solver
from symprog.lang.language import ASTModelingLanguage, LanguageRegistry

__all__ = [
    "BraspLanguage",
    "Architecture",
    "Operation",
    "Head",
    "Stage",
    "BraspTransformer",
    "BraspSyntaxError",
    "Vector",
    "Builder",
    "Run",
    "Compile",
    "Step",
    "Automaton",
    "HeadSummary",
]

# ----------------------------------------------------------------------------
# Runtime: Boolean helpers that work on Python Booleans and on formula strings
# ----------------------------------------------------------------------------

def And(*xs):
    """Conjunction, simplifying constants away."""
    if any(x is False for x in xs):
        return False
    xs = [x for x in xs if x is not True]
    return True if not xs else xs[0] if len(xs) == 1 else f"And({', '.join(xs)})"

def Or(*xs):
    """Disjunction, simplifying constants away."""
    if any(x is True for x in xs):
        return True
    xs = [x for x in xs if x is not False]
    return False if not xs else xs[0] if len(xs) == 1 else f"Or({', '.join(xs)})"

def Not(x):
    """Negation of a Boolean or a formula."""
    return (not x) if isinstance(x, bool) else f"Not({x})"

def ITE(condition, then, otherwise):
    """If-then-else."""
    return Or(And(condition, then), And(Not(condition), otherwise))


class Vector:
    """A Boolean vector, indexed by positions 1..n. Values are Python Booleans (when
    running on a word) or formulas (when compiling)."""

    def __init__(self, values):
        self.values = list(values)

    def __getitem__(self, i):
        if not 1 <= i <= len(self.values):
            raise IndexError(f"position {i} is out of range 1..{len(self.values)}")
        return self.values[i - 1]

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        return iter(self.values)

    def __eq__(self, other):
        return isinstance(other, Vector) and self.values == other.values

    def __repr__(self):
        if all(isinstance(v, bool) for v in self.values):
            return f"Vector({' '.join('1' if v else '0' for v in self.values)})"
        return f"Vector({self.values})"


class Builder:
    """The B-RASP operations, used by rewritten programs. Subclasses provide `Q`
    (the symbol vectors), `positions`, and `define` (how a new vector is stored)."""

    def vector(self, name: str, at: Callable) -> Vector:
        """Defines the vector `name`, whose value at position i is `at(i)`."""
        return self.define(name, [at(i) for i in self.positions])

    ## masks: the candidate positions j for position i (`head` numbers the attention in the
    ## program, from 0, which only `Step` uses)
    def before(self, i, head=None):
        return range(1, i)

    def after(self, i, head=None):
        return range(i + 1, len(self.positions) + 1)

    def everywhere(self, i, head=None):
        return range(1, len(self.positions) + 1)

    def upto(self, i, head=None):
        return range(1, i + 1)

    def onwards(self, i, head=None):
        return range(i, len(self.positions) + 1)

    ## attention: an if-then-else chain over the candidates, nearest first
    def last(self, candidates, score: Callable, value: Callable, default):
        """The value at the rightmost candidate j where score(j) holds, else `default`."""
        expr = default
        for j in candidates:  ## later candidates wrap earlier ones, so the rightmost is tested first
            expr = ITE(score(j), value(j), expr)
        return expr

    def first(self, candidates, score: Callable, value: Callable, default):
        """The value at the leftmost candidate j where score(j) holds, else `default`."""
        expr = default
        for j in reversed(candidates):
            expr = ITE(score(j), value(j), expr)
        return expr

    def define(self, name: str, values: list) -> Vector:
        raise NotImplementedError


class Run(Builder):
    """Runs a program on one word, with Python Booleans. `vectors` records every vector, in
    the order they are computed (the symbol vectors `Q<symbol>` first): the program's trace."""

    def __init__(self, alphabet, word: str):
        unknown = set(word) - set(alphabet)
        if unknown:
            raise ValueError(f"symbols {sorted(unknown)} are not in the alphabet {list(alphabet)}")
        self.alphabet, self.word = list(alphabet), word
        self.positions = range(1, len(word) + 1)
        self.Q = {a: Vector([c == a for c in word]) for a in self.alphabet}
        self.vectors = {f"Q{a}": self.Q[a] for a in self.alphabet}

    def define(self, name, values):
        self.vectors[name] = Vector(values)
        return self.vectors[name]


class Compile(Builder):
    """Builds a program for all words of length n. Each symbol at each position is an
    input variable `Q<symbol>_<i>`, and each defined vector gets one variable per position,
    `<prefix><name>_<i>`, defined by a biconditional (constants stay constants)."""

    def __init__(self, alphabet, n: int, prefix: str = ""):
        self.alphabet, self.n, self.prefix = list(alphabet), n, prefix
        self.positions = range(1, n + 1)
        self.Q = {a: Vector([f"Q{a}_{i}" for i in self.positions]) for a in self.alphabet}
        self.inputs = [f"Q{a}_{i}" for i in self.positions for a in self.alphabet]
        ## each position holds exactly one symbol (`Xor` of several arguments means exactly one)
        self.formulas = [
            f"Xor({', '.join(f'Q{a}_{i}' for a in self.alphabet)})" if len(self.alphabet) > 1
            else f"Q{self.alphabet[0]}_{i}"
            for i in self.positions
        ]
        self.defined = []

    def define(self, name, values):
        out = []
        for i, value in zip(self.positions, values):
            if isinstance(value, bool):
                out.append(value)
                continue
            variable = f"{self.prefix}{name}_{i}"
            self.formulas.append(f"Biconditional({variable}, {value})")
            self.defined.append(variable)
            out.append(variable)
        return Vector(out)

    def word(self, model: dict) -> str:
        """The input word of a model (an assignment to the variables)."""
        return "".join(
            next(a for a in self.alphabet if model[f"Q{a}_{i}"]) for i in self.positions
        )


# ----------------------------------------------------------------------------
# Automata: programs whose attentions only look at the past, as DFAs
# ----------------------------------------------------------------------------

class FutureMask(ValueError):
    """A program with a mask that looks ahead (`after`, `onwards`, `everywhere`), which the
    left-to-right automaton construction does not handle."""


class Column:
    """A vector in a `Step`: its value at the new position, and at past positions (looked up
    in their profiles when read, since profiles only keep what attentions read at j).
    Indexed like a `Vector`."""

    def __init__(self, name: str, history: dict[int, dict], position: int, value):
        self.name, self.history, self.position, self.value = name, history, position, value

    def __getitem__(self, j):
        return self.value if j == self.position else self.history[j][self.name]


class Step(Builder):
    """Runs a program at one new position, `position`, given summaries of the positions before
    it (see `automaton`): `candidates[k]` lists the past positions that attention k may pick
    (in order), and `history` maps each to its profile (the values there of the vectors that
    attentions read at j). Masks only offer these (and the new position, for `upto`), so a
    program with a future mask raises `FutureMask`."""

    def __init__(self, alphabet, history: dict[int, dict], candidates: list[list[int]], position: int, symbol: str):
        self.history, self.candidates, self.position = history, candidates, position
        self.Q = {a: Column(f"Q{a}", history, position, symbol == a) for a in alphabet}
        self.vectors = {f"Q{a}": self.Q[a] for a in alphabet}

    def vector(self, name, at):
        self.vectors[name] = Column(name, self.history, self.position, at(self.position))
        return self.vectors[name]

    def before(self, i, head=None):
        return self.candidates[head]

    def upto(self, i, head=None):
        return self.candidates[head] + [i]

    def after(self, i, head=None):
        raise FutureMask("the automaton construction needs masks that look at the past (before, upto)")

    everywhere = onwards = after


@dataclass
class Automaton:
    """A DFA over `alphabet`: states are 0..len(delta)-1, `delta[q][a]` is the state after
    reading symbol a in state q, `start` is the initial state and `accepting` the set of
    accepting states."""
    alphabet: list[str]
    delta: list[dict[str, int]]
    start: int
    accepting: set[int]

    @property
    def size(self) -> int:
        return len(self.delta)

    def accepts(self, word: str) -> bool:
        q = self.start
        for a in word:
            q = self.delta[q][a]
        return q in self.accepting

    def minimize(self) -> "Automaton":
        """The minimal DFA (Moore's partition refinement), with the dead state, if any, last."""
        block = [int(q in self.accepting) for q in range(self.size)]
        while True:
            signature = [(block[q],) + tuple(block[self.delta[q][a]] for a in self.alphabet)
                         for q in range(self.size)]
            ids = {s: k for k, s in enumerate(dict.fromkeys(signature))}
            refined = [ids[s] for s in signature]
            if len(ids) == len(set(block)):
                break
            block = refined
        ## renumber in breadth-first order from the start (the dead state last)
        live = self.live_states()
        dead = {block[q] for q in range(self.size) if q not in live}
        order, queue = [], [block[self.start]]
        while queue:
            b = queue.pop(0)
            if b in order:
                continue
            order.append(b)
            q = block.index(b)
            queue += [block[self.delta[q][a]] for a in self.alphabet]
        order.sort(key=lambda b: b in dead)
        number = {b: k for k, b in enumerate(order)}
        delta = [{a: number[block[self.delta[block.index(b)][a]]] for a in self.alphabet} for b in order]
        accepting = {number[block[q]] for q in self.accepting}
        return Automaton(self.alphabet, delta, number[block[self.start]], accepting)

    def live_states(self) -> set[int]:
        """The states from which some word is accepted."""
        live = set(self.accepting)
        while True:
            more = {q for q in range(self.size) if q not in live
                    and any(self.delta[q][a] in live for a in self.alphabet)}
            if not more:
                return live
            live |= more

    def unrolled(self, n: int) -> list[dict[int, bool | dict[str, int]]]:
        """The automaton unrolled over the words of length n, as in `unroll`: node F(k, q) is
        "the rest of the word, from position k+1 on, leads from q to acceptance", with
        `F(n, q) = (q accepting)` and `F(k, q) = Or_a (x_{k+1} = a) And F(k+1, delta(q, a))`.
        Returns, for each k = 0..n, the states q whose node is reached from F(0, start), mapped
        to True (F(k, q) is ⊤: every suffix is accepted), False (⊥: none is) or the decision
        {a: delta(q, a)} over symbol x_{k+1}. Only the nodes that are not ⊤ or ⊥ are followed,
        so the result is what the circuit keeps (before merging equal nodes)."""
        values: list[dict[int, bool | dict[str, int]]] = [{} for _ in range(n + 1)]
        values[n] = {q: q in self.accepting for q in range(self.size)}
        for k in range(n - 1, -1, -1):
            for q in range(self.size):
                children = [values[k + 1][self.delta[q][a]] for a in self.alphabet]
                if all(c is True for c in children):
                    values[k][q] = True
                elif all(c is False for c in children):
                    values[k][q] = False
                else:
                    values[k][q] = {a: self.delta[q][a] for a in self.alphabet}
        reached = [{self.start}] + [set() for _ in range(n)]
        for k in range(n):
            for q in reached[k]:
                if isinstance(values[k][q], dict):
                    reached[k + 1] |= set(values[k][q].values())
        return [{q: values[k][q] for q in sorted(reached[k])} for k in range(n + 1)]

    def count(self, n: int) -> int:
        """The number of accepted words of length n (forward algorithm)."""
        weights = {self.start: 1}
        for _ in range(n):
            nxt: dict[int, int] = {}
            for q, w in weights.items():
                for a in self.alphabet:
                    nxt[self.delta[q][a]] = nxt.get(self.delta[q][a], 0) + w
            weights = nxt
        return sum(w for q, w in weights.items() if q in self.accepting)


@dataclass
class HeadSummary:
    """What `automaton` keeps of the past for one attention: `names` are the vectors it reads at
    j (in its score and value), `residue` whether it reads j mod the modulus, `choose` is
    "last" or "first", and `ordered` whether the order of the past positions matters (when the
    value reads j; otherwise, only which profiles occurred does)."""
    choose: str
    names: tuple[str, ...]
    residue: bool
    ordered: bool


def automaton(run: Callable, alphabet: list[str], heads: list[HeadSummary], modulus: int = 1,
              max_states: int = 100_000) -> Automaton:
    """The DFA of a recognizer whose attentions only look at the past, built by running the
    program one position at a time (`run(builder)` returns its output vector).

    The state after a prefix summarizes it exactly, attention by attention. A past position j
    matters to attention k only through its *profile* for k: the values at j of the vectors
    that k reads at j, and j mod `modulus` (the lcm of the moduli of `i % m == r`) if k reads
    it. Among positions with the same profile, a rightmost attention (`last`) can only pick
    the last one, and a leftmost attention (`first`) the first one, so for attention k the
    state keeps the last (or first) occurrence of each profile, in order, or only the set of
    profiles that occurred when its value does not read j. The state also holds the length of
    the prefix mod `modulus`, and whether the output is true at the last position (then the
    state is accepting). The states are then minimized.
    """
    def next_state(state, symbol):
        length, summaries, _ = state
        ## past positions, with residues as in the profiles, in order within each attention
        ## (`i % m == r` and `j % m == r` only see residues)
        history, candidates, j = {}, [], 0
        for head, summary in zip(heads, summaries):
            positions = []
            for residue, values in summary:
                j += 1 + (residue - j - 1) % modulus
                history[j] = dict(zip(head.names, values))
                positions.append(j)
            candidates.append(positions)
        position = j + 1 + ((length + 1) % modulus - j - 1) % modulus
        step = Step(alphabet, history, candidates, position, symbol)
        accepted = bool(run(step)[position])
        new = []
        for head, summary in zip(heads, summaries):
            item = ((length + 1) % modulus if head.residue else 0,
                    tuple(step.vectors[name][position] for name in head.names))
            if not head.ordered:
                summary = tuple(sorted(set(summary) | {item}))
            elif head.choose == "last":
                summary = tuple(x for x in summary if x != item) + (item,)
            elif item not in summary:
                summary = summary + (item,)
            new.append(summary)
        return ((length + 1) % modulus, tuple(new), accepted), accepted

    start = (0, tuple(() for _ in heads), False)
    index, delta, accepting, queue = {start: 0}, [{}], set(), [start]
    while queue:
        state = queue.pop()
        for a in alphabet:
            target, accepted = next_state(state, a)
            if target not in index:
                if len(index) >= max_states:
                    raise ValueError(f"the automaton has more than {max_states} states")
                index[target] = len(delta)
                delta.append({})
                queue.append(target)
                if accepted:
                    accepting.add(index[target])
            delta[index[state]][a] = index[target]
    return Automaton(list(alphabet), delta, 0, accepting).minimize()


def unroll(automaton: Automaton, n: int, manager, literal: Callable):
    """The words of length n that the automaton accepts, as an SDD: one node per state and
    position, `node(k, q) = Or_a (exactly a at position k+1) And node(k+1, delta(q, a))`, so
    the SDD has at most n x states decision nodes (with a right-linear vtree in position
    order, it is the automaton's OBDD). `literal(a, k)` is the SDD literal of symbol a at k."""
    nodes = {q: manager.true() if q in automaton.accepting else manager.false()
             for q in range(automaton.size)}
    for k in range(n, 0, -1):
        symbols = {a: literal(a, k) for a in automaton.alphabet}
        exactly = {a: symbols[a] for a in automaton.alphabet}
        for a in automaton.alphabet:
            for b in automaton.alphabet:
                if b != a:
                    exactly[a] = manager.conjoin(exactly[a], manager.negate(symbols[b]))
        nodes = {q: functools.reduce(manager.disjoin, [
                    manager.conjoin(exactly[a], nodes[automaton.delta[q][a]]) for a in automaton.alphabet])
                 for q in range(automaton.size)}
    return nodes[automaton.start]


# ----------------------------------------------------------------------------
# The structure of a program, as a transformer
# ----------------------------------------------------------------------------

@dataclass
class Head:
    """An attention, as an attention head. Each field lists the vectors it reads, by role:
    `query` and `key` are the vectors of the score at i and at j, `value` the vectors of the
    value at j, `value_at_i` the vectors of the value at i (combined with the attended
    position after attention), and `default` the vectors of the default (at i). `positional`
    lists the positional predicates it uses, e.g. `j % 2 == 1` (and `key_positional` and
    `value_positional` whether the score or the value uses one of the attended position j)."""
    choose: str
    mask: str
    query: list[str] = field(default_factory=list)
    key: list[str] = field(default_factory=list)
    value: list[str] = field(default_factory=list)
    value_at_i: list[str] = field(default_factory=list)
    default: list[str] = field(default_factory=list)
    positional: list[str] = field(default_factory=list)
    key_positional: bool = False    ## the score reads `j % m == r`
    value_positional: bool = False  ## the value reads `j % m == r`
    layer: int = 0

    def reads(self) -> list[str]:
        return unique(self.query + self.key + self.value + self.value_at_i + self.default)


@dataclass
class Operation:
    """A statement `name = expression`. It is computed by an attention head (`kind` is
    "attention": the expression is one attention), or position-wise ("position-wise": by an
    FFN, possibly combining the outputs of attention heads inside the expression, `heads`).
    `inputs` lists the vectors read outside of attentions."""
    name: str
    kind: str
    source: str
    heads: list[Head] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    positional: list[str] = field(default_factory=list)
    layer: int = 0


@dataclass
class Architecture:
    """The transformer of a B-RASP program, layer by layer:

    - layer 0 is the **embedding**: the symbol vectors, the positional encodings
      (`i % m == r`), and the position-wise operations that only read these (a function of a
      symbol and its position is part of the embedding);
    - in each layer, the **attention heads** read the residual stream after the previous
      layer, and the **FFN** computes the position-wise operations of the layer;
    - an attention is in layer 1 + the largest layer of the vectors it reads, and a
      position-wise operation in the largest layer of its inputs (and heads), so the number of
      layers is the longest chain of dependent attentions;
    - the **residual stream** has one coordinate per vector, per positional encoding, and per
      attention head inside a position-wise expression (whose result the FFN reads). Each
      layer *adds* its results into their own coordinates, which are 0 until then, so the
      stream is the concatenation of everything computed so far (see `stages`).

    `positional` lists the positional encodings, as `pos % m == r` (the same encoding serves
    `i % m == r` and `j % m == r`).
    """
    symbols: list[str]
    operations: list[Operation]
    outputs: list[str]
    layers: int
    positional: list[str]

    @property
    def heads(self) -> list[tuple[Operation, Head]]:
        return [(op, head) for op in self.operations for head in op.heads]

    @property
    def width(self) -> int:
        """The number of coordinates of the residual stream."""
        return len(self.coordinates)

    @property
    def coordinates(self) -> list[str]:
        """The coordinates of the residual stream, in the order they are written."""
        return [name for stage in self.stages() for name in stage.writes]

    def stages(self) -> list["Stage"]:
        """The embedding, the attention and FFN of each layer, and the readout, with the
        coordinates of the residual stream that each writes and reads."""
        stages = [Stage(0, "embedding", self.symbols + self.positional + [op.name for op in self.ffn_in(0)], [])]
        for layer in range(1, self.layers + 1):
            heads = self.heads_in(layer)
            if heads:
                reads = [name for op, head in heads
                         for name in head.reads() + [positional_encoding(p) for p in head.positional]]
                stages.append(Stage(layer, "attention", [head_name(op, head) for op, head in heads], unique(reads)))
            ffn = self.ffn_in(layer)
            if ffn:
                reads = [name for op in ffn
                         for name in op.inputs + [head_name(op, head) for head in op.heads]
                         + [positional_encoding(p) for p in op.positional]]
                stages.append(Stage(layer, "ffn", [op.name for op in ffn], unique(reads)))
        stages.append(Stage(self.layers + 1, "readout", [], list(self.outputs)))
        return stages

    @property
    def unused(self) -> list[str]:
        """The defined vectors that the outputs do not depend on (dead code: a compiler would
        drop them, but they are kept here to show the program as written)."""
        reads = {op.name: unique(op.inputs + [n for head in op.heads for n in head.reads()])
                 for op in self.operations}
        live, todo = set(), list(self.outputs)
        while todo:
            name = todo.pop()
            if name not in live:
                live.add(name)
                todo.extend(reads.get(name, []))
        return [op.name for op in self.operations if op.name not in live]

    def layer(self, name: str) -> int:
        return 0 if name in self.symbols else next(op.layer for op in self.operations if op.name == name)

    def heads_in(self, layer: int) -> list[tuple[Operation, Head]]:
        return [(op, head) for op, head in self.heads if head.layer == layer]

    def ffn_in(self, layer: int) -> list[Operation]:
        """The position-wise operations of a layer (of the embedding, for layer 0)."""
        return [op for op in self.operations if op.kind == "position-wise" and op.layer == layer]


@dataclass
class Stage:
    """A step of the transformer (`kind` is "embedding", "attention", "ffn" or "readout"),
    with the residual-stream coordinates it writes and reads."""
    layer: int
    kind: str
    writes: list[str]
    reads: list[str]


def head_name(op: Operation, head: Head) -> str:
    """The name of a head's result: the vector it defines, or `name · head k` for a head
    inside a position-wise expression."""
    return op.name if op.kind == "attention" else f"{op.name} · head {op.heads.index(head) + 1}"


def positional_encoding(predicate: str) -> str:
    """`j % 2 == 1` -> `pos % 2 == 1`: the same encoding, at whichever position it is read."""
    return "pos % " + predicate.split(" % ", 1)[1]


def unique(names: list[str]) -> list[str]:
    return list(dict.fromkeys(names))

def add(names: list[str], name: str) -> None:
    if name not in names:
        names.append(name)


def architecture(symbols: list[str], operations: list[Operation], outputs: list[str]) -> Architecture:
    """Assigns layers to the heads and operations of a program (see `Architecture`)."""
    layers = {name: 0 for name in symbols}
    for op in operations:
        for head in op.heads:
            head.layer = 1 + max((layers[name] for name in head.reads()), default=0)
        op.layer = max([head.layer for head in op.heads] + [layers[name] for name in op.inputs], default=0)
        layers[op.name] = op.layer
    positional = unique([positional_encoding(p) for op in operations
                         for p in op.positional + [q for h in op.heads for q in h.positional]])
    return Architecture(
        symbols=list(symbols),
        operations=operations,
        outputs=outputs,
        layers=max((op.layer for op in operations), default=0),
        positional=positional,
    )


# ----------------------------------------------------------------------------
# The transformer
# ----------------------------------------------------------------------------

class BraspSyntaxError(SyntaxError):
    """A program that is not valid B-RASP."""


def brasp_queries(run: Callable, word, queries: list[str]):
    """Runs a program on `word` if given, else on each query (as a dict from words to outputs)."""
    if word is not None:
        return run(word)
    if not queries:
        raise ValueError('no input: run the program on a word, `prog.execute("abba")`, '
                         'or add `query("abba")` to the program')
    return {q: run(q) for q in queries}


ATTENTIONS = {"last", "first"}
MASKS = {"before", "after", "everywhere", "upto", "onwards"}
DECLARATIONS = {"alphabet", "query", "length"}
BUILDER = "_p"   ## the builder inside a rewritten program
WORD = "_word"   ## the argument of a rewritten program
RUN = "_run"     ## the program on one word, inside a rewritten program
RESERVED = ATTENTIONS | MASKS | DECLARATIONS | {
    "And", "Or", "Not", "ITE", "brasp_builder", "brasp_queries", BUILDER, WORD, RUN,
}


class BraspTransformer(ast.NodeTransformer):
    """Checks that a program is B-RASP and rewrites it into plain Python.

    For example, the program

    ```python
    def prog(i):
        alphabet("ab")
        prev_a = last((Qa[j] for j in before(i)), default=False)
        y = Qb[i] and prev_a[i]
        return y
        query("abba")
    ```

    becomes

    ```python
    def prog(_word=None):

        def _run(_word):
            _p = brasp_builder(_word)
            Qa = _p.Q['a']
            Qb = _p.Q['b']
            prev_a = _p.vector('prev_a', lambda i: _p.last(_p.before(i), lambda j: True, lambda j: Qa[j], False))
            y = _p.vector('y', lambda i: And(Qb[i], prev_a[i]))
            return y
        return brasp_queries(_run, _word, ['abba'])
    ```

    so that `prog(word)` runs the program on `word`, and `prog()` on each query.

    The rules it enforces:

    - the body is a sequence of assignments `name = expression`, with one `return`;
    - `alphabet("...")` (at most once, before the assignments) declares the symbols,
      `query("...")` (anywhere) adds an input word, and `length(n)` (at most once, anywhere)
      the length of the words for inference;
    - each name is defined once, before it is used (straight-line code);
    - vectors are always indexed by a position variable in scope: `x[i]`, or `x[j]` inside
      an attention;
    - expressions use `and`, `or`, `not`, `if`-`else`, `True`, `False`, `i % m == r`, and
      attention `last(...)` / `first(...)`, which may not be nested.
    """

    def __init__(self, alphabet=None):
        ## the alphabet given to the Parser, if any (the program may also declare it)
        self.alphabet: list[str] | None = list(alphabet) if alphabet else None
        self.queries: list[str] = []
        self.length: int | None = None
        self.symbol_vectors: dict[str, str] = {}
        self.vectors: set[str] = set()
        self.bound: list[str] = []  ## position variables in scope: [i], or [i, j] in an attention
        ## the structure of the program: its operations, and what each reads (see `Architecture`)
        self.operations: list[Operation] = []
        self.outputs: list[str] = []
        self.operation: Operation | None = None  ## the operation being defined
        self.head: Head | None = None            ## the attention being visited
        self.role: str | None = None             ## "score", "value" or "default", in an attention
        self.heads = 0                           ## attentions so far (numbering them)

    ## errors -----------------------------------------------------------------

    def error(self, node: ast.AST, message: str):
        line = getattr(node, "lineno", None)
        where = f" (line {line} of the program)" if line is not None else ""
        try:
            code = f": `{ast.unparse(node)}`"
        except Exception:
            code = ""
        raise BraspSyntaxError(f"{message}{where}{code}")

    ## the program -------------------------------------------------------------

    def visit_Module(self, node: ast.Module):
        functions = [s for s in node.body if isinstance(s, ast.FunctionDef)]
        if len(node.body) != 1 or len(functions) != 1:
            self.error(node, "a B-RASP program is a single function")
        node.body = [self.visit(functions[0])]
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef):
        args = node.args
        if (len(args.args) != 1 or args.posonlyargs or args.kwonlyargs or args.vararg
                or args.kwarg or args.defaults):
            self.error(node, "a B-RASP program takes exactly one argument, its position variable, e.g. `def prog(i)`")
        position = args.args[0].arg
        if position in self.vectors or position in RESERVED:
            self.error(node, f"`{position}` cannot be the position variable")

        statements = list(node.body)
        ## drop a docstring
        if statements and isinstance(statements[0], ast.Expr) and isinstance(statements[0].value, ast.Constant) \
                and isinstance(statements[0].value.value, str):
            statements = statements[1:]
        self.declare_alphabet(node, statements)

        body = [ast.parse(f"{BUILDER} = brasp_builder({WORD})").body[0]]
        body += [ast.parse(f"{name} = {BUILDER}.Q[{symbol!r}]").body[0]
                 for name, symbol in self.symbol_vectors.items()]
        returned = False
        for statement in statements:
            declaration = self.declaration(statement)
            if declaration == "alphabet":
                if len(body) > 1 + len(self.symbol_vectors):
                    self.error(statement, "declare the alphabet before defining vectors")
            elif declaration == "query":
                self.query(statement)
            elif declaration == "length":
                self.declare_length(statement)
            elif isinstance(statement, ast.Assign):
                if returned:
                    self.error(statement, "vectors are defined before `return`")
                body.append(self.define(statement, position))
            elif isinstance(statement, ast.Return):
                if returned:
                    self.error(statement, "a B-RASP program has one `return`")
                body.append(self.output(statement))
                returned = True
            else:
                self.error(statement, "each statement defines a vector (`name = expression`), "
                                      "or is `alphabet(...)`, `query(...)` or `return`")
        if not returned:
            self.error(node, "a B-RASP program must `return` its output vector(s)")

        ## def prog(_word=None):
        ##     def _run(_word): <body>
        ##     return brasp_queries(_run, _word, [queries])
        run = ast.FunctionDef(name=RUN, args=arguments([WORD]), body=body, decorator_list=[], returns=None,
                              **({"type_params": []} if "type_params" in ast.FunctionDef._fields else {}))
        dispatch = ast.Return(value=ast.Call(
            func=ast.Name(id="brasp_queries", ctx=ast.Load()),
            args=[ast.Name(id=RUN, ctx=ast.Load()), ast.Name(id=WORD, ctx=ast.Load()),
                  ast.List(elts=[ast.Constant(q) for q in self.queries], ctx=ast.Load())],
            keywords=[],
        ))
        node.args = arguments([WORD], defaults=[ast.Constant(None)])
        node.body = [run, dispatch]
        node.decorator_list = []
        node.returns = None
        return node

    ## declarations ------------------------------------------------------------

    @staticmethod
    def declaration(statement: ast.stmt) -> str | None:
        """`alphabet` or `query`, if the statement is `alphabet(...)` or `query(...)`."""
        if (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
                and isinstance(statement.value.func, ast.Name) and statement.value.func.id in DECLARATIONS):
            return statement.value.func.id
        return None

    def string_argument(self, statement: ast.Expr) -> str:
        call = statement.value
        if len(call.args) != 1 or call.keywords or not isinstance(call.args[0], ast.Constant) \
                or not isinstance(call.args[0].value, str):
            self.error(statement, f"write `{call.func.id}(\"...\")`, with one string")
        return call.args[0].value

    def declare_alphabet(self, node: ast.FunctionDef, statements: list[ast.stmt]) -> None:
        """Sets the alphabet, from `alphabet("...")` in the program or from the Parser, and
        the symbol vectors `Q<symbol>`."""
        declared = [s for s in statements if self.declaration(s) == "alphabet"]
        if len(declared) > 1:
            self.error(declared[1], "declare the alphabet once")
        if declared:
            symbols = self.string_argument(declared[0])
            if not symbols or len(set(symbols)) != len(symbols):
                self.error(declared[0], "the alphabet is a nonempty string of distinct symbols, e.g. `alphabet(\"ab\")`")
            if self.alphabet is not None and self.alphabet != list(symbols):
                self.error(declared[0], f"the program declares a different alphabet than the Parser ({''.join(self.alphabet)})")
            self.alphabet = list(symbols)
        if self.alphabet is None:
            self.error(node, 'declare the alphabet, e.g. `alphabet("ab")` in the program '
                             '(or `Parser(language="brasp", alphabet="ab")`)')
        bad = [a for a in self.alphabet if not f"Q{a}".isidentifier()]
        if bad:
            self.error(declared[0] if declared else node,
                       f"symbols {bad} do not give valid vector names `Q<symbol>`")
        self.symbol_vectors = {f"Q{a}": a for a in self.alphabet}
        self.vectors = set(self.symbol_vectors)

    def query(self, statement: ast.Expr) -> None:
        word = self.string_argument(statement)
        unknown = sorted(set(word) - set(self.alphabet))
        if unknown:
            self.error(statement, f"symbols {unknown} are not in the alphabet ({''.join(self.alphabet)})")
        self.queries.append(word)

    def declare_length(self, statement: ast.Expr) -> None:
        """`length(n)`: the length of the words that inference is over."""
        call = statement.value
        if self.length is not None:
            self.error(statement, "declare the length once")
        if len(call.args) != 1 or call.keywords or not is_int(call.args[0]) or call.args[0].value < 1:
            self.error(statement, "write `length(n)`, with a whole number n >= 1")
        self.length = call.args[0].value

    def define(self, node: ast.Assign, position: str) -> ast.Assign:
        """`name = e`  ->  `name = _p.vector('name', lambda i: e')`."""
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            self.error(node, "define one vector per statement: `name = expression`")
        name = node.targets[0].id
        if name in self.vectors:
            self.error(node, f"`{name}` is already defined (vectors cannot be redefined)")
        if name in RESERVED or name == position:
            self.error(node, f"`{name}` is a reserved name")

        is_attention = (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                        and node.value.func.id in ATTENTIONS)
        self.operation = Operation(name=name, kind="attention" if is_attention else "position-wise",
                                   source=ast.unparse(node.value))
        self.bound = [position]
        value = self.visit(node.value)
        self.bound = []
        self.vectors.add(name)  ## only after its definition: no self-reference
        self.operation.inputs = unique(self.operation.inputs)
        self.operations.append(self.operation)
        self.operation = None

        call = ast.Call(
            func=ast.Attribute(value=ast.Name(id=BUILDER, ctx=ast.Load()), attr="vector", ctx=ast.Load()),
            args=[ast.Constant(name), lambda_(position, value)],
            keywords=[],
        )
        return ast.copy_location(ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=call), node)

    def output(self, node: ast.Return) -> ast.Return:
        """`return x`, `return x, y` or `return {"key": x, ...}`, with defined vectors."""
        def check(value):
            if isinstance(value, ast.Name) and value.id in self.vectors:
                self.outputs.append(value.id)
                return
            if isinstance(value, ast.Tuple):
                return [check(v) for v in value.elts]
            if isinstance(value, ast.Dict) and all(isinstance(k, ast.Constant) for k in value.keys):
                return [check(v) for v in value.values]
            self.error(node, "return a vector, a tuple of vectors, or a dict of vectors (without indexing)")
        if node.value is None:
            self.error(node, "a B-RASP program must return its output vector(s)")
        check(node.value)
        return node

    ## expressions -------------------------------------------------------------

    def generic_visit(self, node):
        self.error(node, f"`{type(node).__name__}` is not part of B-RASP")

    def visit_Constant(self, node: ast.Constant):
        if not isinstance(node.value, bool):
            self.error(node, "the only constants are `True` and `False`")
        return node

    def visit_BoolOp(self, node: ast.BoolOp):
        op = "And" if isinstance(node.op, ast.And) else "Or"
        return call(op, [self.visit(v) for v in node.values], node)

    def visit_UnaryOp(self, node: ast.UnaryOp):
        if not isinstance(node.op, ast.Not):
            self.error(node, "the only unary operator is `not`")
        return call("Not", [self.visit(node.operand)], node)

    def visit_IfExp(self, node: ast.IfExp):
        return call("ITE", [self.visit(node.test), self.visit(node.body), self.visit(node.orelse)], node)

    def visit_Name(self, node: ast.Name):
        if node.id in self.vectors:
            self.error(node, f"index the vector at a position, e.g. `{node.id}[{self.bound[-1]}]`")
        if node.id in self.bound:
            self.error(node, f"positions can only index vectors, or appear in `{node.id} % m == r`")
        self.error(node, f"`{node.id}` is not defined (vectors must be defined before they are used)")

    def visit_Subscript(self, node: ast.Subscript):
        """`x[i]` or `x[j]`: a defined vector, at a position variable in scope."""
        if not (isinstance(node.value, ast.Name) and node.value.id in self.vectors):
            name = node.value.id if isinstance(node.value, ast.Name) else ast.unparse(node.value)
            self.error(node, f"`{name}` is not defined (vectors must be defined before they are used)")
        if not (isinstance(node.slice, ast.Name) and node.slice.id in self.bound):
            self.error(node, f"vectors are indexed by a position variable in scope ({', '.join(self.bound)})")
        self.record(node.value.id, at_j=node.slice.id != self.bound[0])
        return node

    def record(self, name: str, at_j: bool) -> None:
        """Records that the operation being defined reads `name`, in the role it is read."""
        if self.head is None:
            self.operation.inputs.append(name)
        elif self.role == "score":
            add(self.head.key if at_j else self.head.query, name)
        elif self.role == "value":
            add(self.head.value if at_j else self.head.value_at_i, name)
        else:
            add(self.head.default, name)

    def visit_Compare(self, node: ast.Compare):
        """`i % m == r`: a positional predicate, which stays a Python comparison (a constant)."""
        ok = (
            len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq)
            and isinstance(node.left, ast.BinOp) and isinstance(node.left.op, ast.Mod)
            and isinstance(node.left.left, ast.Name) and node.left.left.id in self.bound
            and is_int(node.left.right) and is_int(node.comparators[0])
        )
        if not ok:
            self.error(node, "the only comparison is the positional predicate `i % m == r`")
        m, r = node.left.right.value, node.comparators[0].value  # type: ignore
        if not (m >= 1 and 0 <= r < m):
            self.error(node, "in `i % m == r`, we need m >= 1 and 0 <= r < m")
        predicate = f"{node.left.left.id} % {m} == {r}"  # type: ignore
        add(self.operation.positional if self.head is None else self.head.positional, predicate)
        if self.head is not None and node.left.left.id != self.bound[0]:  # type: ignore
            if self.role == "score":
                self.head.key_positional = True
            elif self.role == "value":
                self.head.value_positional = True
        return node

    def visit_Call(self, node: ast.Call):
        """`last((V for j in before(i) if S), default=D)`
        ->  `_p.last(_p.before(i), lambda j: S, lambda j: V, D)`."""
        if not (isinstance(node.func, ast.Name) and node.func.id in ATTENTIONS):
            self.error(node, "the only calls are the attentions `last(...)` and `first(...)`")
        choose = node.func.id
        if len(self.bound) > 1:
            self.error(node, "attention cannot be nested: define the inner attention as its own vector first")
        if len(node.args) != 1 or not isinstance(node.args[0], ast.GeneratorExp):
            self.error(node, f"write attention as `{choose}((value for j in before(i) if score), default=...)`")
        defaults = [k for k in node.keywords if k.arg == "default"]
        if len(defaults) != 1 or len(node.keywords) != 1:
            self.error(node, f"`{choose}` needs exactly one keyword argument, `default=...`")

        generator = node.args[0]
        if len(generator.generators) != 1:
            self.error(node, "attention ranges over one position variable")
        loop = generator.generators[0]
        if not isinstance(loop.target, ast.Name):
            self.error(node, "attention ranges over one position variable, e.g. `for j in before(i)`")
        j, i = loop.target.id, self.bound[-1]
        if j in self.bound or j in self.vectors or j in RESERVED:
            self.error(node, f"`{j}` cannot be the attended position variable")
        mask = loop.iter
        if not (isinstance(mask, ast.Call) and isinstance(mask.func, ast.Name) and mask.func.id in MASKS
                and len(mask.args) == 1 and not mask.keywords
                and isinstance(mask.args[0], ast.Name) and mask.args[0].id == i):
            self.error(node, f"the mask is one of `before({i})`, `after({i})`, `upto({i})`, `onwards({i})` "
                             f"or `everywhere({i})`")

        self.head = Head(choose=choose, mask=mask.func.id)
        self.role = "default"
        default = self.visit(defaults[0].value)  ## evaluated at i
        self.bound = [i, j]
        self.role = "score"
        score = self.visit(ast.BoolOp(op=ast.And(), values=loop.ifs)) if len(loop.ifs) > 1 \
            else self.visit(loop.ifs[0]) if loop.ifs else ast.Constant(True)
        self.role = "value"
        value = self.visit(generator.elt)
        self.bound = [i]
        self.operation.heads.append(self.head)
        self.head, self.role = None, None

        builder = ast.Name(id=BUILDER, ctx=ast.Load())
        candidates = ast.Call(func=ast.Attribute(value=builder, attr=mask.func.id, ctx=ast.Load()),
                              args=[ast.Name(id=i, ctx=ast.Load()), ast.Constant(self.heads)], keywords=[])
        self.heads += 1
        return ast.copy_location(ast.Call(
            func=ast.Attribute(value=builder, attr=choose, ctx=ast.Load()),
            args=[candidates, lambda_(j, score), lambda_(j, value), default],
            keywords=[],
        ), node)


def call(name: str, args: list, node: ast.AST) -> ast.Call:
    return ast.copy_location(ast.Call(func=ast.Name(id=name, ctx=ast.Load()), args=args, keywords=[]), node)

def arguments(names: list[str], defaults: list | None = None) -> ast.arguments:
    return ast.arguments(posonlyargs=[], args=[ast.arg(arg=n) for n in names], kwonlyargs=[],
                         kw_defaults=[], defaults=defaults or [])

def lambda_(variable: str, body: ast.expr) -> ast.Lambda:
    return ast.Lambda(args=arguments([variable]), body=body)

def is_int(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and type(node.value) is int


# ----------------------------------------------------------------------------
# The language
# ----------------------------------------------------------------------------

## how `BraspLanguage` builds circuits: through CNF, directly, or from the program's DFA
## (see `BraspLanguage`)
COMPILATIONS = ("cnf", "direct", "automaton")


@LanguageRegistry(name="brasp")
class BraspLanguage(ASTModelingLanguage):
    """B-RASP programs. The alphabet is declared in the program, `alphabet("ab")`, or given
    to the parser, `@Parser(language="brasp", alphabet="ab")`.

    `prog.execute(word)` runs the program on a word and returns its output vector(s), and
    `prog.execute()` runs it on each `query("...")` of the program, as a dict from words to
    outputs. Calling the underlying program with a builder, `prog.prog(Run(alphabet, word))`,
    keeps the builder's trace of all vectors (`.vectors`); with a `Compile` builder,
    `prog.prog(Compile(alphabet, n))` builds its formulas for all words of length `n`.

    **Inference** is over the words of a fixed length n, declared with `length(n)` in the
    program (or `Parser(..., length=n)`), for a recognizer (a program with one output
    vector): the solver's models are the words of length n whose output is true at the last
    position, each symbol of a word being uniformly random and independent. So
    `prog.inference(mode="Count")` counts the accepted words, `mode="Prob"` gives the
    fraction of words that are accepted, and `query="Qa_3"` gives P(a at position 3 | accepted).
    The input variables are `Q<symbol>_<position>`, and each vector `x` of the program has
    variables `x_<position>` (where it is not a constant).

    **Compilation** (`Parser(..., compilation=...)`) builds the circuit either through CNF
    (`"cnf"`, the default: the formulas are translated to CNF with Tseitin variables, which are
    then quantified away, and the vtree is minimized once compiled) or directly (`"direct"`:
    the formulas are applied as SDDs one by one, minimizing the vtree along the way, which is
    slower on B-RASP's attention chains). With `project=True` (the default), the variables of
    the defined vectors are then quantified away too: each is a function of the input, so this
    leaves the same words and probabilities in a smaller circuit, over the input variables
    only (with `project=False` they stay, and can be queried).

    With `"automaton"`, the program is first turned into its minimal DFA (`automaton()`, for
    programs whose masks only look at the past: `before` and `upto`), which is then unrolled
    over the n positions: the circuit is the DFA's OBDD, with at most n x states decision
    nodes, over the input variables only.
    """

    def __init__(self, alphabet=None, length: int | None = None, compilation: str = "cnf",
                 project: bool = True, **solver_kwargs) -> None:
        if compilation not in COMPILATIONS:
            raise ValueError(f"compilation is one of {', '.join(map(repr, COMPILATIONS))}, not {compilation!r}")
        super().__init__(**solver_kwargs)
        self.compilation = compilation
        self.project = project
        self.alphabet: list[str] | None = list(alphabet) if alphabet else None
        self.queries: list[str] = []
        self.length: int | None = length
        self.architecture: Architecture | None = None  ## the program as a transformer (see `Architecture`)
        self.compiled: Compile | None = None           ## the program compiled by `semantics`
        self._automata: dict = {}                      ## DFAs, by program and output

    @property
    def glbs(self) -> dict[str, ModuleType | Callable]:
        def brasp_builder(word):
            ## a word runs the program; anything else is a builder (`Run`, `Compile`). This checks
            ## for a string rather than for a `Builder`: when the module is reloaded (e.g., by
            ## Streamlit), builders from the new module are not instances of the old `Builder`
            return Run(self.alphabet, word) if isinstance(word, str) else word
        return {"And": And, "Or": Or, "Not": Not, "ITE": ITE,
                "brasp_builder": brasp_builder, "brasp_queries": brasp_queries}

    def parse(self, tree: ast.AST) -> ast.AST:
        transformer = BraspTransformer(self.alphabet)
        tree = transformer.visit(tree)
        ast.fix_missing_locations(tree)
        self.alphabet, self.queries = transformer.alphabet, transformer.queries
        if transformer.length is not None:
            if self.length is not None and self.length != transformer.length:
                raise BraspSyntaxError(f"the program declares length({transformer.length}), but the Parser length={self.length}")
            self.length = transformer.length
        self.architecture = architecture(
            [f"Q{a}" for a in self.alphabet], transformer.operations, unique(transformer.outputs)
        )
        return tree

    def semantics(self, program: str) -> Solver:
        """The solver over the words of the declared length (see the class docstring)."""
        if self.length is None:
            raise ValueError('inference is over the words of a fixed length: add e.g. `length(10)` to the program')
        return self.compile(program, self.length)

    def compile(self, program: str, n: int, output: str | None = None) -> Solver:
        """A solver whose models are the words of length n that the program accepts (its
        output is true at the last position), with uniform, independent symbols.

        Arguments:
            program: the rewritten program (`prog.python_prog`).
            n: the length of the words.
            output: for a program that returns a dict of vectors, the key of the vector that
                decides acceptance (by default, the program must return one vector).
        """
        if output is None and len(self.architecture.outputs) != 1:
            raise ValueError(
                "inference needs a recognizer, a program that returns one output vector "
                f"(this one returns {len(self.architecture.outputs)}: {', '.join(self.architecture.outputs)})"
            )
        if n < 1:
            raise ValueError("the length of the words is at least 1")

        if self.compilation == "automaton":
            return self.compile_automaton(program, n, output)

        ## run the rewritten program with a `Compile` builder
        run = self.runner(program, output)
        compiled = Compile(self.alphabet, n)
        accept = run(compiled)[n]

        formulas = list(compiled.formulas)
        if accept is False:  ## no word is accepted: a contradiction
            formulas.append(f"And({compiled.inputs[0]}, Not({compiled.inputs[0]}))")
        elif accept is not True:
            formulas.append(accept)

        options = dict(
            var_list=compiled.inputs + compiled.defined,
            var_weights={v: 1 / len(self.alphabet) for v in compiled.inputs},
            aux_vars=set(compiled.defined),
            categoricals=set(compiled.inputs),  ## exactly one symbol per position, w(not Qa_i) = 1
            **self.solver_kwargs,
        )
        if self.compilation == "cnf":
            solver = Solver.from_cnf(formulas=[str(f) for f in formulas], **options)
        else:
            solver = Solver.from_kwargs(**options)
            solver.add(formulas)
        if self.project:
            project(solver, compiled.defined)
        self.compiled = compiled
        return solver

    def runner(self, program: str, output: str | None = None) -> Callable:
        """The rewritten program as a function from a builder to its output vector (the one
        named `output`, for a program that returns a dict)."""
        locals_: dict = {}
        exec(compile(program, "<brasp>", "exec"), self.glbs, locals_)
        function = next(f for f in locals_.values() if callable(f))

        def run(builder):
            returned = function(builder)
            if output is not None:
                if not isinstance(returned, dict) or output not in returned:
                    raise ValueError(f"the program does not return a dict with the key {output!r}")
                return returned[output]
            (vector,) = returned.values() if isinstance(returned, dict) \
                else returned if isinstance(returned, tuple) else (returned,)
            return vector
        return run

    def automaton(self, program: str, output: str | None = None) -> Automaton:
        """The minimal DFA of the program, for programs whose masks only look at the past
        (see `automaton`)."""
        key = (program, output)
        if key not in self._automata:
            heads = [head for _, head in self.architecture.heads]
            moduli = [int(p.split(" % ")[1].split(" == ")[0]) for p in self.architecture.positional]
            summaries = [HeadSummary(
                choose=head.choose,
                names=tuple(unique(head.key + head.value)),
                residue=head.key_positional or head.value_positional,
                ordered=bool(head.value) or head.value_positional,
            ) for head in heads]
            self._automata[key] = automaton(
                self.runner(program, output), self.alphabet, summaries, modulus=math.lcm(1, *moduli),
            )
        return self._automata[key]

    def compile_automaton(self, program: str, n: int, output: str | None = None) -> Solver:
        """The solver of `compile`, built by unrolling the program's DFA (see `unroll`), with a
        right-linear vtree in position order."""
        from pysdd.sdd import SddManager, Vtree
        from symprog.solvers.mapper import MapperRegistry

        dfa = self.automaton(program, output)
        compiled = Compile(self.alphabet, n)
        inputs = compiled.inputs  ## in position order: Qa_1, Qb_1, Qa_2, ...
        mapper = MapperRegistry.build_instance("sdd", var_list=inputs)
        vtree = Vtree(var_count=len(inputs), var_order=list(range(1, len(inputs) + 1)), vtree_type="right")
        mapper.manager = SddManager.from_vtree(vtree)
        mapper._sym_map = {name: mapper.manager.literal(k + 1) for k, name in enumerate(inputs)}
        root = unroll(dfa, n, mapper.manager, lambda a, k: mapper._sym_map[f"Q{a}_{k}"])
        ## the SDD library's iterator recurses along the vtree, which is right-linear (deep)
        sys.setrecursionlimit(max(sys.getrecursionlimit(), 20 * len(inputs) + 1000))
        options = dict(self.solver_kwargs)
        options.setdefault("no_simplify", True)  ## keep the OBDD (minimizing would change its vtree)
        solver = Solver.from_kwargs(
            solver=mapper,
            formulas=root,
            var_weights={v: 1 / len(self.alphabet) for v in inputs},
            categoricals=set(inputs),
            **options,
        )
        self.compiled = compiled
        return solver

    def word(self, model: dict) -> str:
        """The word of a model of the solver."""
        return self.compiled.word(model)


def project(solver: Solver, names: list[str]) -> None:
    """Existentially quantifies the variables `names` out of the solver's circuit. They must be
    functions of the other variables (as the defined vectors are of the input), so that each
    model of the rest has exactly one extension: the circuit then has the same models over the
    other variables, with the same weights. The SDD library keeps them in its manager, as free
    variables, so they are marked as hidden (as the Tseitin variables of the CNF route are):
    model counting, enumeration and smoothing leave them out."""
    sdd = solver._solver
    root = solver._formula_obj
    quantified = [name for name in names if name in sdd._sym_map]
    for name in quantified:
        root = sdd.manager.exists(sdd._sym_map[name].literal, root)
    root.ref()  ## keep it from the manager's garbage collection
    sdd._quantified_vars = sdd._hidden_vars | set(quantified)
    solver._formula_obj = root
    solver._formulas = [root]  ## so that rebuilding the formula keeps the projection
