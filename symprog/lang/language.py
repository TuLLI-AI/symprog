import inspect
import ast
import textwrap
import warnings
from collections import defaultdict
from types import ModuleType
from abc import abstractmethod
from typing import Generic, Any, TypeVar, Callable
from dataclasses import dataclass

from ..base import UtilBase
from ..registry import Registry
from ..solvers import Solver
from ..solvers.solver_utils import SolverOut

__all__ = ["ModelingLanguage", "LanguageRegistry", "ProbabilisticProgram", "Parser"]

warnings.filterwarnings("ignore", category=DeprecationWarning)

T = TypeVar("T")

class ModelingLanguage(UtilBase, Generic[T]):
    """A `ModelingLanguage` is a class that represents a modeling language. It is used to 
    define the syntax and semantics of a modeling language. 

    Attributes: 
        glbs (dict[str, ModuleType | Callable]): 
            A dictionary of global variables to be injected into the program during execution.
        variable_weights (dict[str, float]):
            The core variables in the program along with (optional) weights. 
    
    Methods:
        parse(self, tree: T) -> T:
            Parses a program string into the underlying representation of the modeling language.
        semantics(self, program) -> Solver:
            Main method for building semantic representations and other computational graphs of the 
            program. Builds a `Solver` instance.    

    """
    def __init__(self,**solver_kwargs):
        self._solver_kwargs = solver_kwargs  

    @property 
    def solver_kwargs(self): 
        return self._solver_kwargs

    @property 
    def glbs(self) -> dict[str, ModuleType | Callable]:
        """Global variables to be injected into the program during execution.
        
        Returns: 
            A dictionary of global variables to be injected into the program during execution.
        """
        return {}
    
    @property 
    def variable_weights(self) -> dict[str, float]: 
        """Variable weights for the modeling language. This can be used to specify the 
        relative importance of different variables in the program. 
        
        Returns: 
            A dictionary of variable weights for the modeling language, or an empty diction if no 
            variable weights are specified.

        """
        return {}
    
    @property 
    def evidence_literals(self) -> set[str]:
        return set()
    @property 
    def query_literals(self) -> set[str]:
        return set()
    

    @abstractmethod 
    def parse(self, tree: T) -> T:
        """Parses a program string into the underlying representation of the modeling language. 

        Arguments: 
            tree: The abstract syntax tree (AST) representation of the program to be parsed. 
        """
        raise NotImplementedError
    
    @abstractmethod 
    def semantics(self, program) -> Solver:
        """Main method for building semantic representations and other computational graphs of the 
        program. Builds a `Solver` instance. 
        
        Arguments: 
            program: The raw python native program. 
        """
        raise NotImplementedError

class ASTModelingLanguage(ModelingLanguage[ast.AST]):
    """A `ModelingLanguage` that operates on the abstract syntax tree (AST) representation of the 
    input program. This is a common base class for modeling languages that perform transformations 
    on the AST of the input program. 

    Attributes: 
        glbs (dict[str, ModuleType | Callable]): 
            A dictionary of global variables to be injected into the program during execution.
        variable_weights (dict[str, float]):
            The core variables in the program along with (optional) weights. 
    
    Methods:
        parse(self, tree: T) -> T:
            Parses a program string into the underlying representation of the modeling language.
        semantics(self, program) -> Solver:
            Main method for building semantic representations and other computational graphs of the 
            program. Builds a `Solver` instance.    


    """
    def _parse_ast(self, program: str) -> ast.AST:
        """Parses the input program string into an AST representation. 

        Arguments: 
            program: The input program string to be parsed. 
        Returns: 
            The AST representation of the input program string.
        Raises: 
            ValueError: If there is an error parsing the input program string into an AST.

        """
        try: 
            tree = ast.parse(program) 
            return tree
        except Exception as e: 
            raise ValueError(f"Error parsing program string: {e}")


### language registry 
class LanguageRegistry(Registry[ModelingLanguage]):
    """A `LanguageRegistry` is a registry for modeling languages. It is used to register 
    implementations of modeling languages. 
    """ 
    @classmethod 
    def build_instance(cls, name: str, **solver_kwargs) -> ModelingLanguage:
        """Builds a `ModelingLanguage` instance from name and solver keywords. 
        
        Arguments: 
            name: The name of the modeling language in the registry. 
            **solver_kwargs: The solver keywords. 
        Raises: 
            ValueError when unknown modeling language name is provided. 
        
        """
        implementation_tuple = cls._IMPLEMENTATIONS.get(name)
        if implementation_tuple is None:
            impl = list(cls._IMPLEMENTATIONS.keys())
            raise ValueError(
                f"Resource not found registered, value={name}, available classes={', '.join(impl)}"
            )
        impl_cls, _ = implementation_tuple
        return impl_cls(**solver_kwargs) 


### program 
@dataclass 
class ProbabilisticProgram: 
    """A representation of a program in a modeling language.
    
    Attributes: 
        solver (Solver | None): 
            The underlying solver object used by the program that runs inference. 
            This is typically built by the  modeling language's `semantics` method.
        prog_str (str): 
            The string representation of the program. This is typically the transformed
            AST of the input program, converted back into a string.
        prog (Callable): 
            The underlying callable program that can be executed. This is typically a wrapper
            around the transformed AST of the input program.

    Methods: 
        execute(self) -> Any: 
            Executes the program and returns the result. This typically involves executing the 
            underlying callable program and returning the result. 
        sample(self, k: int = 3000) -> dict[str, float | dict[str, float]]: 
            Samples the program `k` times and returns the distribution of return values. This is a 
            common method for performing inference on probabilistic programs. 
        inference(self, mode: str = 'prob', query: Any = None, **kwargs) -> Any: 
            Main method for performing inference on the program. This can be implemented in various
            ways depending on the modeling language and solver used. The `mode` argument can be used
            to specify different modes of reasoning (e.g., probabilistic inference, MAP inference, etc.).
        grad(self, mode: str = 'prob', query: Any = None, **kwargs) -> Any: 
            Main method for performing gradient-based inference on the program. This can be implemented in various
            ways depending on the modeling language and solver used. The `mode` argument can be used
            to specify different modes of reasoning (e.g., probabilistic inference, MAP inference, etc). 
        tensorize(self, mode: str = 'prob', query: Any = None, **kwargs) -> Any:
            Main method for performing tensorization of the program. This can be used to convert the 
            program into a tensor-based representation that can be used for gradient-based optimization 
            and inference. This can be implemented in various ways depending on the modeling language and 
            solver used. The `mode` argument can be used to specify different modes of reasoning (e.g., 
            probabilistic inference, MAP inference, etc).

    Example usage:

    ```python
    @Parser(language="constraint")
    def prog():
        a: 0.9; b: 0.7
        (not a) or b
        a or b 
        
    program = prog() # executes the underlying program 
    print(program.python_prog)  # prints the transformed program string
    print(program.sample(k=1000))  # samples the program 1000 times 
    ```

    """
    _language: ModelingLanguage
    _prog: Callable 
    _prog_str: str
    _solver: Solver | None = None

    @property 
    def solver(self) -> Solver | None: 
        """Returns the underlying solver object used by the program."""
        return self._solver

    @property 
    def prog_str(self) -> str: 
        """Returns the string representation of the program."""
        return self._prog_str
    @property 
    def python_prog(self) -> str: 
        """Returns the string representation of the underlying Python program."""
        return self._prog_str
    
    @property 
    def prog(self) -> Callable: 
        """Returns the underlying callable program."""
        return self._prog

    def execute(self):
        """Execute the program."""
        return self._prog()

    def __call__(self):
        return self.execute()

    def _build_semantics(self): 
        """Build the semantic backbone of the program. This can be used to build a symbolic 
        representation of the program, or to build a computational graph for the program. 
        
        """
        if self._solver is None:
            self._solver = self._language.semantics(self._prog_str)

    def sample(self, k: int = 3000) -> dict[str, float | dict[str, float]]:
        """Samples the program `k` times and returns the distribution of return values.
        
        Arguments:
            k: The number of samples to draw from the program.
        Returns:
            A dictionary containing the number of attempts, number of successful 
            executions, and a distribution of return values
        """ 
        attempts = 0.0 
        executed = 0.0
        return_values = defaultdict(float) 

        for _ in range(k): 
            attempts += 1 
            try: 
                r = self.execute()
                return_values[str(r)] += 1.0
                executed += 1.0 
            except AssertionError as e:
                continue 
        return {
            "attempts": attempts, 
            "executed": executed, 
            "return_values": dict(return_values),
        }
    
    ### inference operations 
    ########################
    ########################
    
    def _prepare_evidence(self, **kwargs) -> tuple[set[str], set[str]]:
        """Prepares evidence for different modes of inference.
        
        Arguments: 
            **kwargs: the raw kwargs passed to the inference methods. 
        Raises: 
            ValueError if the modeling language and solver are not built correctly.
        """
        if self._language is None or self._solver is None:
            raise ValueError(
                "Modeling language and solver must be built before preparing evidence."
            )
        
        ## program evidence and provided evidence 
        evidence = kwargs.get("evidence", set())
        if not isinstance(evidence, set):
            evidence = set([e.strip() for e in evidence.split(" ")])
        program_evidence = self._language.evidence_literals
        evidence_literals = set(str(e) for e in evidence) | set(
            str(e) for e in program_evidence
        )

        ## program queries and provided queries (as `query` or `queries`)
        program_queries = self._language.query_literals
        queries = set()
        for key in ("query", "queries"):
            provided = kwargs.get(key) or set()
            if isinstance(provided, str):
                provided = set(q.strip() for q in provided.split(" ") if q.strip())
            queries |= set(provided)
        query_literals = set(str(q) for q in queries) | set(
            str(q) for q in program_queries
        )

        return evidence_literals, query_literals

    def _per_query(self, fn: Callable, mode: str, **kwargs) -> Any:
        """Calls `fn` (e.g., `solver.query`) with the program and provided evidence, once per
        query literal. Returns a single result for at most one query, otherwise a list."""
        evidence_literals, query_literals = self._prepare_evidence(**kwargs)
        kwargs.pop("queries", None)
        kwargs["evidence"] = evidence_literals

        if len(query_literals) < 2:
            kwargs["query"] = query_literals
            return fn(mode=mode, **kwargs)

        ## one result per query, in sorted order (see `details["marginal"]` for `Prob`)
        query_out = []
        for query in sorted(query_literals):
            kwargs["query"] = query
            query_out.append(fn(mode=mode, **kwargs))
        return query_out

    def grad(self, mode: str = 'Prob', **kwargs) -> Any:
        """Main method for performing gradient-based inference on the program. This relies on the underlying `solver`
        to perform inference. 

        Arguments: 
            mode: 
                Specified the particular mode of reasoning to perform over the probabilistic
                program (e.g., probabilistic inference, MAP inference, etc.). The particular 
                modes of reasoning available will depend on the modeling language and solver used.
            query: 
                An optional query to perform inference on. E.g, a particular variable or set of 
                variables to perform inference over. 
            **kwargs:
                Additional keyword arguments to pass to the underlying solver's inference method.
        """ 
        self._build_semantics()
        assert self.solver is not None, "Solver not built correctly"

        return self._per_query(self.solver.gradient, mode, **kwargs)
    
    def inference(self, mode: str = 'Prob', **kwargs) -> SolverOut | list[SolverOut]:
        """Main method for performing inference on the program. Relies on the underlying `solver`
        to perform inference. 

        Arguments: 
            mode: 
                Specified the particular mode of reasoning to perform over the probabilistic
                program (e.g., probabilistic inference, MAP inference, etc.). The particular 
                modes of reasoning available will depend on the modeling language and solver used.
            query: 
                An optional query to perform inference on. E.g, a particular variable or set of 
                variables to perform inference over. 
            **kwargs:
                Additional keyword arguments to pass to the underlying solver's inference method.

        Returns:
            The result of performing inference on the program.

        """
        self._build_semantics()
        assert self.solver is not None, "Solver not built correctly"

        return self._per_query(self.solver.query, mode, **kwargs)
    
    def tensorize(self, mode: str = 'Prob', **kwargs) -> Any:
        """Main method for performing tensorization of the program. This can be used to convert the 
        program into a tensor-based representation that can be used for gradient-based optimization 
        and inference. Relies on the underlying `solver` to perform tensorization. 

        Arguments: 
            mode (str): 
                Specified the particular mode of reasoning to perform over the probabilistic
                program (e.g., probabilistic inference, MAP inference, etc.). The particular 
                modes of reasoning available will depend on the modeling language and solver used.
            **kwargs:
                Additional keyword arguments to pass to the underlying solver's tensorization method.
        """
        self._build_semantics()
        assert self.solver is not None, "Solver not built correctly"
        evidence_literals, query_literals = self._prepare_evidence(**kwargs)
        kwargs.pop("queries", None)
        kwargs["evidence"] = evidence_literals
        kwargs["query"] = query_literals
        return self.solver.tensorize(mode=mode, **kwargs)

def safe_getsource(f):
    """Get the underlying source code.

    Arguments:
        f: The program and code to parse.
    """
    try:
        return inspect.getsource(f)
    except (OSError, TypeError, IOError):
        try:
            import dill.source as ds # type: ignore

            return ds.getsource(f)
        except Exception:
            raise RuntimeError(
                "Cannot retrieve source code. "
                "Is this code being run interactively (REPL/Jupyter) or after reload? "
                "Move your function to a script file for @Parser."
            )

class Parser(UtilBase):
    """Decorator class for parsing a program string into a `ProbabilisticProgram` object.
    
    Attributes:
        language: The modeling language to use for parsing.
    """ 
    def __init__(self, language: str = "constraint", **kwargs): 
        """Initializes the `Parser` with the specified modeling language.

        Arguments:
            language: The modeling language to use for parsing. Defaults to "constraint".
            **kwargs: Additional keyword arguments to pass to the language constructor.
        """
        self._solver_kwargs = kwargs
        self._language = LanguageRegistry.build_instance(
            name=language,**kwargs
        )

    @property 
    def language(self) -> ModelingLanguage:
        """Returns the underlying modeling language object used by the parser."""
        return self._language

    def parse(self, f: str | Callable) -> ProbabilisticProgram:
        """Parses a program string into a `ProbabilisticProgram` object. 

        Arguments: 
            f: The program and code to parse. Can be a string or a callable function.
        Returns:
            A `ProbabilisticProgram` object representing the parsed program.
        """
        if not isinstance(f, str):
            orig_source = safe_getsource(f)
            func_name = f.__name__
        else:
            orig_source = f
            func_name = "prog"

        lines = textwrap.dedent(orig_source).lstrip().splitlines()
        lines = [
            line for line in lines if not line.strip().startswith("@Parser")
        ]
        source = "\n".join(lines)

        try: 
            tree = ast.parse(source) 
        except Exception as e: 
            raise ValueError(f"Error parsing program string: {e}")
    
        ## compute native python program via transformations 
        tree = self._language.parse(tree)
        py_prog: str = ast.unparse(tree) #astor.to_source(tree) 
        code = compile(tree, filename="<ast>", mode="exec") # type: ignore

        def wrapper(*args, **kwargs):
            glbs = self._language.glbs
            locs = {}
            exec(code, glbs, locs)
            return locs[func_name](*args, **kwargs)
        
        return ProbabilisticProgram(
            _language=self._language,
            _prog=wrapper, 
            _prog_str=py_prog,
        )

    def __call__(self, f: str | Callable, language: str = "constraint") -> ProbabilisticProgram:
        ### parses the program 
        return self.parse(f)
