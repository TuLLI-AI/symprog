# symprog

**Symbolic programming in Python: a fast prototyping toolkit for
working with weighted Boolean formulas and discrete (probabilistic) programs.**

> ⚠️ **Work in progress.** `symprog` is under active development. APIs may
> change between releases, and some features are experimental. Feedback, bug
> reports and contributions are very welcome. Please open an
> [issue](https://github.com/TuLLI-AI/symprog/issues).

`symprog` lets you write small symbolic programs (Boolean constraints,
probabilistic logic programs, weighted formulas) directly inside Python. You can
then reason about them on the fly: count models, compute exact probabilities,
find the most likely assignment, take gradients, or look at the underlying
symbolic equations.

It is designed with two uses in mind:

- **Rapid prototyping.** Sketch a new modeling idea involving weighed
  Boolean formulas or discrete programming in a few lines of Python. Then switch the reasoning backend (SDDs, BDDs, SAT, SMT) and the
  numerical backend (PyTorch, symbolic algebra) without rewriting your program.
- **Teaching.** The library stays small and readable. Each piece (languages,
  solvers, semirings, knowledge compilers) matches a concept from logic,
  probabilistic reasoning and automated reasoning, and the
  [notebooks](notebooks/) walk through them step by step.

## Installation

`symprog` requires Python 3.10 or newer.

```bash
pip install symprog
```

To get the latest development version from GitHub:

```bash
pip install "git+https://github.com/TuLLI-AI/symprog.git"
```

Optional extras:

```bash
pip install "symprog[console]"    # interactive Streamlit console
pip install "symprog[notebooks]"  # Jupyter and graphviz for the tutorials
pip install "symprog[jax]"        # experimental JAX backend
pip install "symprog[klay]"       # klay tensorizaiton backend (https://github.com/ML-KULeuven/klay)
```

For development, clone the repository and install it in editable mode:

```bash
git clone https://github.com/TuLLI-AI/symprog.git
cd symprog
pip install -e ".[dev,notebooks,console]"
pytest
```

## Quick tour

### Languages

`symprog` can embed small probabilistic programming languages *inline* in
Python, with on-the-fly inference and differentiable computation.

Below is a small program in probabilistic logic. It has a set of constraints
over weighted variables:

```python
from symprog import Parser

@Parser(language="constraint")
def prog():
    ## variable weights
    a: 0.9; b: 0.7; c: 0.65

    ## Boolean constraints
    (not a) or b
    c or b
    (not b) or c

    ## conditioning
    condition: a
```

You can then query the exact probability that the formula is satisfied:

```python
prog.inference(mode="Prob").inference
```

The next example uses a small probabilistic logic programming language, inspired
by [ProbLog](https://dtai.cs.kuleuven.be/problog/):

```python
from symprog import Parser

@Parser(language="plp")
def inh():
    ## probabilistic facts
    c1: 0.5; c2: 0.5

    ## weighted clauses
    e1[0.3] :- c1
    e1[0.2, False] :- c2

    e2[0.4] :- c2
    e2[0.2, False] :- c1

    e :- e1
    e :- e2

    ## evidence or conditions
    evidence(c1, True)
    evidence(c2, True)

    ## queries
    query(e)
```

The same kinds of inference work here. For more examples see
[`notebooks/languages.ipynb`](notebooks/languages.ipynb).

### Interactive console

To try programs interactively, launch the console from a clone of the
repository (it requires the `console` extra):

```bash
streamlit run console.py
```

The console comes with example programs in both languages (graph coloring,
fault diagnosis, Monty Hall, a hidden Markov model and more). Besides running,
sampling and exact inference, it can show:

- the same program evaluated in each semiring, with the symbolic expressions;
- the program's models, with a brute-force check of the solver's results;
- the compiled circuit, with the value of each node in a chosen semiring, and
  the files to download it;
- each program as inline Python code, ready to copy.

### Solvers

Inference is driven by `Solver`s. A `Solver` gives you one simple interface to
several reasoning tools, such as
[pysdd](https://pysdd.readthedocs.io/en/latest/),
[z3](https://ericpony.github.io/z3py-tutorial/guide-examples.htm)  and
[python-sat](https://pysathq.github.io/), and to tensor libraries such as
[PyTorch](https://pytorch.org/). With these it performs exact inference over
(weighted) Boolean formulas.

```python
from symprog import Solver

solver = Solver()

## add propositional formulas and variable weights
solver.add([
    "Implies(a, b)",
    "And(Or(a, b), Implies(a, c))",
])
solver.add_weights({"a": 0.9, "b": 0.64, "c": 0.34})

## probabilistic inference (weighted model counting)
print(solver.query())
# => SolverOut(inference_type=Prob, inference=0.2598400115966797)
```

Solvers support many kinds of inference through an extensible *algebraic*
(semiring) interface. See
[`notebooks/solver_tutorial.ipynb`](notebooks/solver_tutorial.ipynb) for details:

```python
## Viterbi-style inference (most likely assignment of variables)
solver.query(mode="Viterbi").inference
solver.query(mode="ViterbiAssignment").variables
# => {'c': True, 'b': True, 'a': True}
solver.query(mode="LogViterbi").inference

## semantic entropy (entropy of interpretations given the formulas)
solver.query(mode="Entropy").inference

## probability and log probability (more numerically stable)
solver.query(mode="Prob").inference
solver.query(mode="LogProb").inference

## fuzzy inference
solver.query(mode="Fuzzy").inference

## satisfiability testing (SAT)
solver.query(mode="SAT").inference
```

Conditioning on evidence:

```python
solver.query(mode="ViterbiAssignment", evidence="~a").variables
# => {'c': False, 'a': False, 'b': True}
```

Gradients with respect to the weights:

```python
solver.gradient(mode="Viterbi", key="weights")
# => tensor([0.2176, 0.3060, 0.5760])
```

Model enumeration:

```python
for model in solver.models():
    print(model)
# {'a': True, 'c': True, 'b': True}
# {'a': False, 'c': False, 'b': True}
# {'a': False, 'c': True, 'b': True}
```

### Backend support

Not every backend supports every feature yet:

| Backend | SAT / model counting / model enumeration | Weighted inference (`Prob`, `Viterbi`, …) | Gradients |
| --- | --- | --- | --- |
| `sdd` solver (default) | ✅ | ✅ | ✅ |
| `bdd`, `z3`, `sympy`, `pysat` solvers | ✅ | ❌ not yet implemented | n/a |
| `pytorch` weights (default) | | ✅ | ✅ |
| `python` weights | | ✅ | ❌ |
| `sympy` / `symengine` symbolic weights | | ✅ | ✅ (symbolic) |
| `jax` weights (**experimental**, `pip install "symprog[jax]"`) | | ✅ | ✅ |

With the JAX backend, `Solver.as_function` returns inference as a pure function
of the weight vector (ordered as in `solver.weight_names`). You can then
transform it with `jax.grad`, `jax.jit` and `jax.vmap`, for example to compute
batched gradients for many weight settings at once:

```python
import jax, jax.numpy as jnp

solver.add_weights({"a": 0.9, "b": 0.64, "c": 0.34}, container_type="jax")
f = solver.as_function(mode="LogProb", evidence="c")
batched_grads = jax.jit(jax.vmap(jax.grad(f)))(jnp.array([[0.9, 0.64, 0.34],
                                                          [0.2, 0.50, 0.70]]))
```

The `Product` inference mode (product t-norm with probabilistic sum) is also
**experimental**. It is not a true semiring, so its results depend on the
structure of the compiled circuit.

**Tensorization** (`solver.tensorize`), which compiles an inference into a
PyTorch module for fast batched evaluation, is **experimental** as well. It has
two backends: `fx` (default), which supports all inference modes, evidence,
queries and conditionals, and `klay`, which uses
[KLay](https://github.com/ML-KULeuven/klay) to evaluate circuits layer by layer
and supports `Prob`, `LogProb`, `Viterbi`, `LogViterbi` and `Fuzzy`, without
evidence or conditioning (`pip install "symprog[klay]"`, Python ≥ 3.11).

Experimental features emit a `symprog.ExperimentalWarning`, which can be
silenced with:

```python
import warnings, symprog
warnings.filterwarnings("ignore", category=symprog.ExperimentalWarning)
```

### Symbolic equations

Solvers also connect to computer algebra systems
([SymPy](https://www.sympy.org/),
[SymEngine](https://github.com/symengine/symengine.py)). You can produce and
manipulate the exact symbolic equations behind any of the inferences above:

```python
## the symbolic polynomial behind the entropy computation
solver.query(mode="Entropy", symbolic=True).inference
# => log(a*b*c + b*(1 - a)) - (a*b*c*log(a) + a*(b*c*log(b) + b*c*log(c)) + ...

## its symbolic gradient vector
solver.gradient(mode="Entropy", symbolic=True, key="weights")
```

## Tutorials

The [`notebooks/`](notebooks/) directory contains step-by-step tutorials, in
suggested reading order:

| Notebook | Topic |
| --- | --- |
| [`solver_tutorial.ipynb`](notebooks/solver_tutorial.ipynb) | **Start here:** formulas, weighted model counting, conditioning, and the algebraic (semiring) view of inference |
| [`languages.ipynb`](notebooks/languages.ipynb) | Writing inline constraint programs and probabilistic logic programs (ProbLog-style) |
| [`symbolic_and_gradients.ipynb`](notebooks/symbolic_and_gradients.ipynb) | Symbolic expressions, numerical/symbolic/Boolean gradients, and learning weights |
| [`tensorization_and_jax.ipynb`](notebooks/tensorization_and_jax.ipynb) | Fast batched inference: compiling to PyTorch modules, and JAX (`jit`, `vmap`, `grad`) |
| [`solver_interfaces.ipynb`](notebooks/solver_interfaces.ipynb) | The solver backends (SDD, BDD, z3, sympy, PySAT) |
| [`weight_interface.ipynb`](notebooks/weight_interface.ipynb) | The weight backends (PyTorch, Python, symbolic, JAX) |

## Contributing

`symprog` is meant to grow with its users. Bug reports, examples, new languages
and new solver backends are all welcome. Please open an issue to discuss larger
changes before submitting a pull request.

## License

`symprog` is released under the [Apache License 2.0](LICENSE).
