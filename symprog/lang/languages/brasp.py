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

    ## masks: the candidate positions j for position i
    def before(self, i):
        return range(1, i)

    def after(self, i):
        return range(i + 1, len(self.positions) + 1)

    def everywhere(self, i):
        return range(1, len(self.positions) + 1)

    def upto(self, i):
        return range(1, i + 1)

    def onwards(self, i):
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
# The structure of a program, as a transformer
# ----------------------------------------------------------------------------

@dataclass
class Head:
    """An attention, as an attention head. Each field lists the vectors it reads, by role:
    `query` and `key` are the vectors of the score at i and at j, `value` the vectors of the
    value at j, `value_at_i` the vectors of the value at i (combined with the attended
    position after attention), and `default` the vectors of the default (at i). `positional`
    lists the positional predicates it uses, e.g. `j % 2 == 1`."""
    choose: str
    mask: str
    query: list[str] = field(default_factory=list)
    key: list[str] = field(default_factory=list)
    value: list[str] = field(default_factory=list)
    value_at_i: list[str] = field(default_factory=list)
    default: list[str] = field(default_factory=list)
    positional: list[str] = field(default_factory=list)
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
                              args=[ast.Name(id=i, ctx=ast.Load())], keywords=[])
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
    """

    def __init__(self, alphabet=None, length: int | None = None, **solver_kwargs) -> None:
        super().__init__(**solver_kwargs)
        self.alphabet: list[str] | None = list(alphabet) if alphabet else None
        self.queries: list[str] = []
        self.length: int | None = length
        self.architecture: Architecture | None = None  ## the program as a transformer (see `Architecture`)
        self.compiled: Compile | None = None           ## the program compiled by `semantics`

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

        ## run the rewritten program with a `Compile` builder
        locals_: dict = {}
        exec(compile(program, "<brasp>", "exec"), self.glbs, locals_)
        function = next(f for f in locals_.values() if callable(f))
        compiled = Compile(self.alphabet, n)
        returned = function(compiled)
        if output is not None:
            if not isinstance(returned, dict) or output not in returned:
                raise ValueError(f"the program does not return a dict with the key {output!r}")
            vector = returned[output]
        else:
            (vector,) = returned.values() if isinstance(returned, dict) \
                else returned if isinstance(returned, tuple) else (returned,)
        accept = vector[n]

        formulas = list(compiled.formulas)
        if accept is False:  ## no word is accepted: a contradiction
            formulas.append(f"And({compiled.inputs[0]}, Not({compiled.inputs[0]}))")
        elif accept is not True:
            formulas.append(accept)

        solver = Solver.from_kwargs(
            var_list=compiled.inputs + compiled.defined,
            var_weights={v: 1 / len(self.alphabet) for v in compiled.inputs},
            aux_vars=set(compiled.defined),
            categoricals=set(compiled.inputs),  ## exactly one symbol per position, w(not Qa_i) = 1
            **self.solver_kwargs,
        )
        solver.add(formulas)
        self.compiled = compiled
        return solver

    def word(self, model: dict) -> str:
        """The word of a model of the solver."""
        return self.compiled.word(model)
