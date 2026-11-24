"""Compiling a B-RASP program into the weights of a transformer (a Tracr for B-RASP).

`compile_transformer(prog)` turns a parsed B-RASP program into a `CompiledTransformer`, a
PyTorch module with the layers of `prog.language.architecture`, whose weights compute the
program exactly:

```python
model = compile_transformer(prog, max_length=16)
model.run("abba")      # {"Qa": Vector(1 0 0 1), ..., "x": Vector(0 0 1 0)}: as `Run`
model.reading("abba")  # every vector as a probability at every position, and each head's attention
model.cpt("y")         # the FFN computing `y`, as a conditional probability table
```

The module is an ordinary transformer (embeddings, attention heads with query, key, value
and output matrices, ReLU FFNs, residual connections; no layer normalization), so it can be
trained from these weights. Read as probabilities, it is a **stochastic B-RASP program** with
the structure of the original:

- each **attention head** is a categorical choice of the attended position j (its attention
  weights), with "no match" as attending to BOS, whose probability is that of the default;
- each **FFN operation** is a Bernoulli, Pr(z = 1) = sigmoid(beta * coordinate), whose
  conditional probability table (`cpt`) is read off by clamping its inputs.

The compiled weights give probabilities close to 0 and 1, which reproduce the program.

**The residual stream.** One coordinate per Boolean signal, +1 for true and -1 for false, 0
while not yet written (and at BOS, which the FFNs never write). The coordinates are

- the embedding: `one` (1 everywhere), `bos` (1 at BOS), `pos` (j / max_length), the symbols
  `Q<a>` and the positional encodings `pos % m == r`;
- for each attention head: `<head>.found` (whether some position matched) and a copy of the
  vectors that its value reads at the attended position, `<head>.<vector>@j`;
- for each operation: its vector, written by the FFN of its layer (for an attention, from
  the head's signals: its value if found, else its default);
- for each head, the features of its score, written by the FFN of the layer before:
  `<head>.score[...]` (read as keys) and `<head>.query[...]` (read as queries).

**Attention.** The score S(i, j) is a Boolean function of vectors at i (the query) and at j
(the key). The FFN before the head computes, at each position, the key features
k_a = S(a, key bits) for each assignment a of the query bits, and the query features
q_a = [query bits = a], so that S(i, j) = sum_a q_a(i) k_a(j) is bilinear. The logits are

    lambda * S(i, j)  +/-  mu * j / max_length  +  (lambda / 2) * bos(j)

(+ for rightmost, - for leftmost), with mu = gap * max_length and lambda = 4 mu: a matching
position beats BOS, BOS beats the other positions, and among the matches the rightmost (or
leftmost) wins by `gap` per position. The mask (before, upto, ...) is the head's attention
mask; BOS is always visible.

**FFNs.** Each operation of a layer, and each score feature for the next layer, is a
Boolean function of coordinates written before the FFN (composing the operations of the same
layer). It becomes a lookup table: one ReLU unit per row, i.e., per assignment b of its k
inputs, ReLU(sum_k s_k x_k - (k - 1) - 2 bos) with s_k = +1 or -1 the sign of b, which is 1
exactly when the inputs match b. The output weight of a unit is the table's value there. When
a head found no match, the bits it copied are not read (they are 0), so its rows leave them
out.
"""
from __future__ import annotations

import ast
import itertools
import math
from dataclasses import dataclass, field

import torch
from torch import nn

from symprog.lang.languages.brasp import Run, Vector, head_name, unique

__all__ = ["compile_transformer", "CompiledTransformer", "Coordinate", "HeadSpec", "Reading"]

ONE, BOS, POS = "one", "bos", "pos"
MAX_INPUTS = 16  ## inputs of one lookup table (2^16 rows at most)
MASK = -1e9      ## the attention logit of positions outside the mask


# ----------------------------------------------------------------------------
# The program: operations and attentions, from their source
# ----------------------------------------------------------------------------

def _calls(node: ast.expr) -> list[ast.Call]:
    """The attentions of an expression, in the order the parser numbers them."""
    if isinstance(node, ast.Call):
        return [node]
    if isinstance(node, ast.BoolOp):
        return [c for v in node.values for c in _calls(v)]
    if isinstance(node, ast.UnaryOp):
        return _calls(node.operand)
    if isinstance(node, ast.IfExp):
        return _calls(node.test) + _calls(node.body) + _calls(node.orelse)
    return []


def _reads(node: ast.expr) -> list[tuple[str, str]]:
    """The signals an expression reads outside of attentions, as (position variable, name):
    a vector `x[i]` is ("i", "x"), and `i % m == r` is ("i", "pos % m == r")."""
    if isinstance(node, ast.Subscript):
        return [(node.slice.id, node.value.id)]
    if isinstance(node, ast.Compare):
        return [(node.left.left.id, f"pos % {node.left.right.value} == {node.comparators[0].value}")]
    if isinstance(node, ast.BoolOp):
        return unique([r for v in node.values for r in _reads(v)])
    if isinstance(node, ast.UnaryOp):
        return _reads(node.operand)
    if isinstance(node, ast.IfExp):
        return unique(_reads(node.test) + _reads(node.body) + _reads(node.orelse))
    return []


def _evaluate(node: ast.expr, at, attend=None) -> bool:
    """The value of an expression, given the value of each signal, `at(variable, name)`, and
    of each attention, `attend(call)`."""
    if isinstance(node, ast.Constant):
        return bool(node.value)
    if isinstance(node, ast.BoolOp):
        values = [_evaluate(v, at, attend) for v in node.values]
        return all(values) if isinstance(node.op, ast.And) else any(values)
    if isinstance(node, ast.UnaryOp):
        return not _evaluate(node.operand, at, attend)
    if isinstance(node, ast.IfExp):
        return _evaluate(node.body if _evaluate(node.test, at, attend) else node.orelse, at, attend)
    if isinstance(node, (ast.Subscript, ast.Compare)):
        ((variable, name),) = _reads(node)
        return at(variable, name)
    if isinstance(node, ast.Call):
        return attend(node)
    raise ValueError(f"not a B-RASP expression: {ast.unparse(node)}")


@dataclass
class HeadSpec:
    """An attention head of the compiled transformer: the attention `last(...)`/`first(...)`
    of operation `op`, with its score, value and default, and the signals each reads (at the
    query position i, or at the attended position j)."""
    name: str
    op: str
    layer: int
    choose: str
    mask: str
    i: str
    j: str
    score: ast.expr | None
    value: ast.expr
    default: ast.expr
    score_i: list[str]
    score_j: list[str]
    value_i: list[str]
    value_j: list[str]
    default_i: list[str]

    @property
    def found(self) -> str:
        return f"{self.name}.found"

    def copied(self, name: str) -> str:
        """The coordinate where the head writes the signal `name` of the attended position."""
        return f"{self.name}.{name}@j"

    def query_features(self) -> list[tuple[tuple[bool, ...], str]]:
        if not self.score_i:
            return []
        return [(bits, f"{self.name}.query[{''.join('1' if b else '0' for b in bits)}]")
                for bits in itertools.product([False, True], repeat=len(self.score_i))]

    def key_features(self) -> list[tuple[tuple[bool, ...], str]]:
        if not self.score_i:
            return [((), f"{self.name}.score")]
        return [(bits, f"{self.name}.score[{''.join('1' if b else '0' for b in bits)}]")
                for bits in itertools.product([False, True], repeat=len(self.score_i))]


def _head_spec(op, head, call: ast.Call) -> HeadSpec:
    generator = call.args[0]
    loop = generator.generators[0]
    i, j = loop.iter.args[0].id, loop.target.id
    score = (None if not loop.ifs else loop.ifs[0] if len(loop.ifs) == 1
             else ast.BoolOp(op=ast.And(), values=loop.ifs))
    default = call.keywords[0].value
    split = lambda node, var: [name for v, name in (_reads(node) if node is not None else []) if v == var]
    return HeadSpec(
        name=head_name(op, head), op=op.name, layer=head.layer, choose=head.choose, mask=head.mask,
        i=i, j=j, score=score, value=generator.elt, default=default,
        score_i=split(score, i), score_j=split(score, j),
        value_i=split(generator.elt, i), value_j=split(generator.elt, j),
        default_i=split(default, i),
    )


# ----------------------------------------------------------------------------
# The residual stream
# ----------------------------------------------------------------------------

@dataclass
class Coordinate:
    """A coordinate of the residual stream: written by the embedding, an attention head or an
    FFN (`kind`: "embedding", "attention", "ffn" for a vector of the program, "feature" for
    a score feature) of `layer`. `encoding` is "±1" (true is +1, false -1) or "0/1"."""
    name: str
    kind: str
    layer: int
    encoding: str
    description: str = ""


@dataclass
class Table:
    """A Boolean function as a lookup table: `rows` maps assignments of `inputs` (tuples of
    (input, value), leaving out inputs that do not matter in that row) to the output."""
    output: str
    inputs: list[str]
    rows: dict[tuple[tuple[str, bool], ...], bool]
    encoding: str


# ----------------------------------------------------------------------------
# The modules
# ----------------------------------------------------------------------------

class Attention(nn.Module):
    """One attention head: logits x W_Q (x W_K)^T under its mask, then softmax, then the
    values x W_V, written back by W_O."""

    def __init__(self, spec: HeadSpec, d: int, d_qk: int, d_v: int):
        super().__init__()
        self.spec = spec
        self.W_Q = nn.Parameter(torch.zeros(d, d_qk))
        self.W_K = nn.Parameter(torch.zeros(d, d_qk))
        self.W_V = nn.Parameter(torch.zeros(d, d_v))
        self.W_O = nn.Parameter(torch.zeros(d_v, d))

    def mask(self, T: int) -> torch.Tensor:
        """Additive mask over (query i, key j), positions 0 (BOS) to T - 1: BOS is always
        visible, and BOS itself only sees BOS."""
        i = torch.arange(T).unsqueeze(1)
        j = torch.arange(T).unsqueeze(0)
        allowed = {
            "before": j < i, "upto": j <= i, "after": j > i, "onwards": j >= i,
            "everywhere": torch.ones(T, T, dtype=torch.bool),
        }[self.spec.mask] & (j >= 1) & (i >= 1)
        allowed = allowed | (j == 0)
        allowed[0, 1:] = False
        return torch.where(allowed, 0.0, MASK)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = (x @ self.W_Q) @ (x @ self.W_K).transpose(-1, -2) + self.mask(x.shape[-2]).to(x)
        pattern = torch.softmax(logits, dim=-1)
        return (pattern @ (x @ self.W_V)) @ self.W_O, pattern


class FFN(nn.Module):
    """ReLU(x W_1 + b_1) W_2."""

    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.W_1 = nn.Parameter(torch.zeros(d, hidden))
        self.b_1 = nn.Parameter(torch.zeros(hidden))
        self.W_2 = nn.Parameter(torch.zeros(hidden, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x @ self.W_1 + self.b_1) @ self.W_2


@dataclass
class Reading:
    """A run of the transformer on a word, read as a stochastic B-RASP program:
    `probability[name][i - 1]` is Pr(vector `name` is true at position i), for the symbols,
    positional encodings and vectors of the program; `attention[head]` is the head's
    attention, a (n + 1) x (n + 1) matrix over positions 0 (BOS, i.e., no match) to n;
    `residual` lists the residual stream after each stage, (n + 1) x d."""
    word: str
    probability: dict[str, list[float]]
    attention: dict[str, torch.Tensor]
    residual: list[tuple[str, torch.Tensor]]


class CompiledTransformer(nn.Module):
    """A transformer compiled from a B-RASP program (see the module docstring). Layer 0 is
    the embedding and its FFN; each layer 1..L has attention heads, then an FFN."""

    def __init__(self, alphabet: list[str], layout: list[Coordinate], heads: list[HeadSpec],
                 tables: list[list[Table]], layers: int, outputs: list[str], max_length: int,
                 gap: float, beta: float):
        super().__init__()
        self.alphabet, self.layout, self.heads = list(alphabet), layout, heads
        self.tables, self.layers, self.outputs = tables, layers, outputs
        self.max_length, self.gap, self.beta = max_length, gap, beta
        self.index = {c.name: k for k, c in enumerate(layout)}
        d = len(layout)
        self.token = nn.Parameter(torch.zeros(len(alphabet) + 1, d))   ## row 0: BOS
        self.position = nn.Parameter(torch.zeros(max_length + 1, d))   ## row 0: BOS
        self.attention = nn.ModuleList(nn.ModuleList() for _ in range(layers + 1))  ## [0] unused
        self.ffn = nn.ModuleList()

    @property
    def width(self) -> int:
        return len(self.layout)

    def vectors(self) -> list[str]:
        """The Boolean (±1) coordinates: symbols, positional encodings and vectors."""
        return [c.name for c in self.layout if c.encoding == "±1"]

    ## running ------------------------------------------------------------------

    def tokens(self, words: list[str]) -> torch.Tensor:
        """BOS followed by the symbols, for words of the same length."""
        if len({len(w) for w in words}) > 1:
            raise ValueError("the words of a batch have the same length")
        if any(len(w) > self.max_length for w in words):
            raise ValueError(f"the transformer was compiled for words of length at most {self.max_length}")
        return torch.tensor([[0] + [self.alphabet.index(a) + 1 for a in w] for w in words])

    def forward(self, tokens: torch.Tensor, trace: bool = False):
        """The residual stream after the last layer, (batch, 1 + n, d). With `trace`, also the
        residual stream after each stage and the attention of each head."""
        T = tokens.shape[-1]
        x = self.token[tokens] + self.position[:T]
        stages, patterns = [("embedding", x)], {}
        for layer in range(self.layers + 1):
            if layer > 0 and len(self.attention[layer]):
                delta = 0
                for head in self.attention[layer]:
                    out, pattern = head(x)
                    delta = delta + out
                    patterns[head.spec.name] = pattern
                x = x + delta
                stages.append((f"layer {layer} · attention", x))
            x = x + self.ffn[layer](x)
            stages.append((f"layer {layer} · FFN" if layer else "embedding · FFN", x))
        return (x, stages, patterns) if trace else x

    def probabilities(self, x: torch.Tensor, names: list[str] | None = None) -> dict[str, torch.Tensor]:
        """Pr(vector is true) at each position, sigmoid(beta * coordinate), without BOS."""
        names = names if names is not None else self.vectors()
        return {name: torch.sigmoid(self.beta * x[..., 1:, self.index[name]]) for name in names}

    @torch.no_grad()
    def run(self, word: str) -> dict[str, Vector]:
        """The program's vectors on `word`, read from the residual stream (true when the
        coordinate is positive), as `Run(alphabet, word).vectors`."""
        if not word:
            return {name: Vector([]) for name in self.vectors()}
        x = self(self.tokens([word]))[0]
        return {name: Vector([bool(v > 0) for v in x[1:, self.index[name]]]) for name in self.vectors()
                if not name.startswith("pos % ")}

    @torch.no_grad()
    def reading(self, word: str) -> Reading:
        """The run on `word` as a stochastic B-RASP program (see `Reading`)."""
        x, stages, patterns = self(self.tokens([word]), trace=True)
        probability = {name: p[0].tolist() for name, p in self.probabilities(x).items()}
        return Reading(word, probability, {h: p[0] for h, p in patterns.items()},
                       [(label, s[0]) for label, s in stages])

    ## agreement with the program -------------------------------------------------

    @torch.no_grad()
    def check(self, prog, words) -> list[tuple[str, str]]:
        """The (word, vector) pairs where the transformer and the program disagree."""
        wrong = []
        for word in words:
            run = Run(self.alphabet, word)
            prog.prog(run)
            mine = self.run(word)
            wrong += [(word, name) for name, vector in run.vectors.items() if mine.get(name) != vector]
        return wrong

    @torch.no_grad()
    def fidelity(self, prog, words) -> dict[str, float]:
        """For each vector of the program, the mean probability that the transformer gives to
        the program's value, over the positions of `words`. A compiled transformer has
        fidelity close to 1; for a trained one, it measures how far it computes the program."""
        totals: dict[str, list[float]] = {}
        for word in words:
            if not word:
                continue
            run = Run(self.alphabet, word)
            prog.prog(run)
            p = self.probabilities(self(self.tokens([word])))
            for name, vector in run.vectors.items():
                if name in p:
                    for k, value in enumerate(vector):
                        q = p[name][0, k].item()
                        totals.setdefault(name, []).append(q if value else 1 - q)
        return {name: sum(v) / len(v) for name, v in totals.items()}

    ## FFNs as conditional probability tables ------------------------------------------

    def table(self, name: str) -> tuple[int, Table]:
        for layer, tables in enumerate(self.tables):
            for t in tables:
                if t.output == name:
                    return layer, t
        raise KeyError(f"no FFN computes {name!r}")

    @torch.no_grad()
    def cpt(self, name: str) -> list[dict]:
        """The FFN output `name` as a conditional probability table: for each row of its
        lookup table (an assignment of its inputs; inputs that do not matter are None), the
        program's value and Pr(name = 1) = sigmoid(beta * output), with the inputs clamped to
        ±1 and the other coordinates 0 (`one` is 1)."""
        layer, t = self.table(name)
        rows = []
        for row, value in t.rows.items():
            x = torch.zeros(self.width)
            x[self.index[ONE]] = 1.0
            for signal, v in row:
                x[self.index[signal]] = 1.0 if v else -1.0
            out = self.ffn[layer](x)[self.index[name]].item()
            given = dict(row)
            rows.append({**{s: (int(given[s]) if s in given else None) for s in t.inputs},
                         "program": int(value),
                         f"Pr({name} = 1)": 1 / (1 + math.exp(-self.beta * out)) if t.encoding == "±1"
                         else min(max(out, 0.0), 1.0)})
        return rows

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ----------------------------------------------------------------------------
# The compiler
# ----------------------------------------------------------------------------

class _Compiler:
    def __init__(self, prog, max_length: int, gap: float, beta: float):
        language = prog.language
        self.arch = arch = language.architecture
        self.alphabet = list(language.alphabet)
        self.max_length, self.gap, self.beta = max_length, gap, beta
        self.ops = {op.name: op for op in arch.operations}
        self.trees: dict[str, ast.expr] = {}
        self.op_heads: dict[str, list[HeadSpec]] = {}
        self.calls: dict[int, HeadSpec] = {}
        for op in arch.operations:
            tree = ast.parse(op.source, mode="eval").body
            calls = _calls(tree)
            if len(calls) != len(op.heads):
                raise ValueError(f"cannot match the attentions of {op.name!r} to its heads")
            specs = [_head_spec(op, head, call) for head, call in zip(op.heads, calls)]
            self.trees[op.name], self.op_heads[op.name] = tree, specs
            self.calls.update({id(call): spec for call, spec in zip(calls, specs)})
        self.heads = [h for specs in self.op_heads.values() for h in specs]
        ## the head whose copy each coordinate `<head>.<name>@j` is
        self.copies = {h.copied(name): h for h in self.heads for name in h.value_j}

    ## the layout ---------------------------------------------------------------

    def layout(self) -> list[Coordinate]:
        arch = self.arch
        coords = [
            Coordinate(ONE, "embedding", 0, "0/1", "1 at every position"),
            Coordinate(BOS, "embedding", 0, "0/1", "1 at BOS (position 0)"),
            Coordinate(POS, "embedding", 0, "0/1", "the position j, as j / max_length"),
        ]
        coords += [Coordinate(s, "embedding", 0, "±1", "symbol") for s in arch.symbols]
        coords += [Coordinate(p, "embedding", 0, "±1", "positional encoding") for p in arch.positional]
        for layer in range(arch.layers + 1):
            for h in self.heads_in(layer):
                coords.append(Coordinate(h.found, "attention", layer, "±1", "some position matched the score"))
                coords += [Coordinate(h.copied(n), "attention", layer, "±1", f"{n} at the attended position")
                           for n in h.value_j]
            for op in self.ops_in(layer):
                coords.append(Coordinate(op.name, "ffn", layer, "±1", op.source))
            for h in self.heads_in(layer + 1):
                coords += [Coordinate(name, "feature", layer, "0/1", f"query features of {h.name}")
                           for _, name in h.query_features()]
                coords += [Coordinate(name, "feature", layer, "0/1", f"score of {h.name} at this key")
                           for _, name in h.key_features()]
        return coords

    def heads_in(self, layer: int) -> list[HeadSpec]:
        return [h for h in self.heads if h.layer == layer]

    def ops_in(self, layer: int):
        return [op for op in self.arch.operations if op.layer == layer]

    ## Boolean functions of an FFN's inputs ----------------------------------------------

    def computed(self, name: str, layer: int) -> bool:
        """Whether the FFN of `layer` computes the vector `name` (else it reads it)."""
        return name in self.ops and self.ops[name].layer == layer

    def dependencies(self, name: str, layer: int) -> set[str]:
        """The coordinates, written before the FFN of `layer`, that `name` depends on."""
        if not self.computed(name, layer):
            return {name}
        out = set()
        for _, n in _reads(self.trees[name]):
            out |= self.dependencies(n, layer)
        for h in self.op_heads[name]:
            out.add(h.found)
            out |= {h.copied(n) for n in h.value_j}
            for n in h.value_i + h.default_i:
                out |= self.dependencies(n, layer)
        return out

    def value(self, name: str, layer: int, a: dict[str, bool]) -> bool:
        """The value of `name` at a position, given the coordinates `a` written before the
        FFN of `layer` there."""
        if not self.computed(name, layer):
            return a[name]
        here = lambda _, n: self.value(n, layer, a)

        def attend(call):
            h = self.calls[id(call)]
            if a[h.found]:
                return _evaluate(h.value, lambda v, n: a[h.copied(n)] if v == h.j else self.value(n, layer, a))
            return _evaluate(h.default, here)

        return _evaluate(self.trees[name], here, attend)

    def score(self, h: HeadSpec, query: tuple[bool, ...], layer: int, a: dict[str, bool]) -> bool:
        """The score of head `h` with the query bits `query` at i, and this position as j."""
        if h.score is None:
            return True
        bits = dict(zip(h.score_i, query))
        return _evaluate(h.score, lambda v, n: bits[n] if v == h.i else self.value(n, layer, a))

    def possible(self, a: dict[str, bool]) -> bool:
        """Whether an assignment can occur: one symbol per position."""
        symbols = [a[s] for s in self.arch.symbols if s in a]
        return sum(symbols) <= 1 and (len(symbols) < len(self.arch.symbols) or sum(symbols) == 1)

    def table(self, output: str, inputs: set[str], function, encoding: str) -> Table:
        order = {c.name: k for k, c in enumerate(self.coords)}
        inputs = sorted(inputs, key=order.__getitem__)
        if len(inputs) > MAX_INPUTS:
            raise ValueError(f"{output!r} depends on {len(inputs)} coordinates, too many for a lookup table")
        rows: dict = {}
        for bits in itertools.product([False, True], repeat=len(inputs)):
            a = dict(zip(inputs, bits))
            if not self.possible(a):
                continue
            ## the bits a head copied do not matter when it found no match (they are 0)
            row = tuple((n, v) for n, v in a.items() if not (n in self.copies and not a[self.copies[n].found]))
            value = function(a)
            assert rows.get(row, value) == value, f"inconsistent table for {output}"
            rows[row] = value
        return Table(output, inputs, rows, encoding)

    def tables(self, layer: int) -> list[Table]:
        """The lookup tables of the FFN of `layer`: its vectors, then the score features of
        the heads of the next layer."""
        out = [self.table(op.name, self.dependencies(op.name, layer),
                          lambda a, n=op.name: self.value(n, layer, a), "±1")
               for op in self.ops_in(layer)]
        for h in self.heads_in(layer + 1):
            key_inputs = set().union(*[self.dependencies(n, layer) for n in h.score_j])
            for bits, name in h.key_features():
                out.append(self.table(name, key_inputs, lambda a, b=bits, h=h: self.score(h, b, layer, a), "0/1"))
            query_inputs = set().union(*[self.dependencies(n, layer) for n in h.score_i])
            for bits, name in h.query_features():
                out.append(self.table(name, query_inputs,
                                      lambda a, b=bits, h=h: all(self.value(n, layer, a) == v
                                                                 for n, v in zip(h.score_i, b)), "0/1"))
        return out

    ## the weights ----------------------------------------------------------------

    def build(self) -> CompiledTransformer:
        self.coords = self.layout()
        tables = [self.tables(layer) for layer in range(self.arch.layers + 1)]
        model = CompiledTransformer(self.alphabet, self.coords, self.heads, tables, self.arch.layers,
                                    list(self.arch.outputs), self.max_length, self.gap, self.beta)
        ix = model.index
        d = model.width
        with torch.no_grad():
            ## embeddings
            for k, a in enumerate(self.alphabet, start=1):
                for b in self.alphabet:
                    model.token[k, ix[f"Q{b}"]] = 1.0 if a == b else -1.0
            model.token[0, ix[BOS]] = 1.0
            model.position[:, ix[ONE]] = 1.0
            for j in range(1, self.max_length + 1):
                model.position[j, ix[POS]] = j / self.max_length
                for p in self.arch.positional:
                    m, r = (int(x) for x in p.split(" % ")[1].split(" == "))
                    model.position[j, ix[p]] = 1.0 if j % m == r else -1.0

            ## attention heads
            mu = self.gap * self.max_length
            lam = 4 * mu
            for layer in range(1, self.arch.layers + 1):
                for h in self.heads_in(layer):
                    queries = h.query_features() or [((), ONE)]
                    keys = h.key_features()
                    copied = [h.copied(n) for n in h.value_j]
                    head = Attention(h, d, len(keys) + 2, 1 + len(copied))
                    for k, ((_, q), (_, key)) in enumerate(zip(queries, keys)):
                        head.W_Q[ix[q], k] = lam
                        head.W_K[ix[key], k] = 1.0
                    head.W_Q[ix[ONE], len(keys)] = mu if h.choose == "last" else -mu
                    head.W_K[ix[POS], len(keys)] = 1.0
                    head.W_Q[ix[ONE], len(keys) + 1] = lam / 2
                    head.W_K[ix[BOS], len(keys) + 1] = 1.0
                    ## values: found = one - 2 bos, and a copy of each vector the value reads at j
                    head.W_V[ix[ONE], 0], head.W_V[ix[BOS], 0] = 1.0, -2.0
                    head.W_O[0, ix[h.found]] = 1.0
                    for k, (name, target) in enumerate(zip(h.value_j, copied), start=1):
                        head.W_V[ix[name], k] = 1.0
                        head.W_O[k, ix[target]] = 1.0
                    model.attention[layer].append(head)

            ## FFNs: one unit per row of each lookup table
            for layer, layer_tables in enumerate(tables):
                units = [(t, row, value) for t in layer_tables for row, value in t.rows.items()
                         if t.encoding == "±1" or value]
                ffn = FFN(d, len(units))
                for u, (t, row, value) in enumerate(units):
                    for signal, v in row:
                        ffn.W_1[ix[signal], u] = 1.0 if v else -1.0
                    ffn.W_1[ix[BOS], u] = -2.0
                    ffn.b_1[u] = -(len(row) - 1)
                    ffn.W_2[u, ix[t.output]] = (1.0 if value else -1.0) if t.encoding == "±1" else 1.0
                model.ffn.append(ffn)
        return model


def compile_transformer(prog, max_length: int | None = None, gap: float = 20.0,
                        beta: float = 10.0) -> CompiledTransformer:
    """Compiles a parsed B-RASP program (`Parser(language="brasp")`) into a transformer.

    Arguments:
        prog: the program.
        max_length: the longest word the transformer runs on (the size of its positional
            embedding); by default the program's `length(n)`, or its longest query, or 16.
        gap: how much the attention logits of neighboring positions differ, so that the
            attention is hard up to about exp(-gap).
        beta: the scale of the readout, Pr(true) = sigmoid(beta * coordinate).
    """
    language = prog.language
    if max_length is None:
        max_length = max([language.length or 0] + [len(q) for q in language.queries]) or 16
    return _Compiler(prog, max_length, gap, beta).build()
