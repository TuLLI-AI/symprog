import warnings
from typing import TypeVar, Callable, Generic, Any
from dataclasses import dataclass

from .solver_utils import build_from_cnf, extract_vars, build_formula, Formula, SolverOut
from .tensorization import symbolic_to_fx
from .klay_interface import klay_module, KLAY_MODES
from ..base import ExperimentalWarning
from .mapper import SymbolicMapper, MapperRegistry 
from .semirings import *
from .semiring import SemiringRegistry
from .numerical import WeightManager, WeightRegistry

__all__ = ["Solver"]

T = TypeVar("T")
I = TypeVar("I")
O = TypeVar("O")

@dataclass 
class Solver(Generic[T, I, O]):
    """Interface for accessing symbolic solvers, performing inference over Boolean formulas
    with weights, and storing those formulas and weights. 

    Specifically designed to work with different symbolic backends (e.g., SDDs, BDDs, etc.), 
    various weight managers (e.g., PyTorch, NumPy, etc.), and different semirings for performing 
    weighted model counting and conditional inference. 

    Attributes: 
        solver_type (str)
            The type of solver backend being used to perform inference in the solver
            (e.g., 'sdd', 'bdd', etc.). Will automatically construct formulas within 
            the specified solver type. 
        solver (SymbolicMapper)
            The underlying symbolic solver/mapper object (i.e., an instance of an sdd or bdd
            solver). 
        weights (WeightManager): 
            The weight manager for storing solver parameters, performing operations on weights, 
                and computing gradients.
        formulas (list[any]): 
            An optional list of formulas stored in the solver. By default this is empty and
                formulas can be added via the `add_formulas` method.
        conditional (any): 
            An optional conditional formula to condition on during inference. Empty by default. 
        aux_vars (set[str | int]): An optional set of auxiliary unweighted variables to ignore 
            during inference and weighted search (used for encoding conditionals).
        no_simplify (bool): Whether to skip simplification of formulas in the solver. Defaults to False
            and only applies to certain solver backends that support simplification (e.g., SDDs).

    Methods:
        add_formulas(formulas: str | list[any]) -> None:
            Method for adding a formula to the solver's formula list. Multiple formulas
            can be added at once by providing a list of formulas, and a final formula is constructed 
            as the conjunction of all added formulas.   
        add_conditional(formula: str | any) -> None:
            Adds a formula to condition on when performing inference with 
            the solver. 
        remove_conditional() -> None:
            Removes any existing conditional formula from the solver.
        reset() -> None:
            Resets the solver state, clearing formulas and weights, and even 
            the solver itself. 
        pop() -> None:
            Removes the last added formula from the solver's formula list.
        inference(
            formula: Optional[any] = None, 
            mode: Optional[str] = "Prob", 
            fast: Optional[bool] = False, 
            evidence: Optional[set[str | int] | str] = None,
            conditional: any = None, 
            **kwargs
        ) -> any:
            Method for performing inference over the given formula (with optional conditioning). 
                When no formula is provided, uses the formula added via `add_formulas`. If `fast` 
                is set to True, will use fast counting methods (if available) for counting mode.
        gradient(f: any, key: Optional[str] = None) -> any:
            Method for computing weight gradients.

    Example quick usage: 

    >>> from symprog.solvers import Solver
    >>> s = Solver() 
    >>> s.add("And(Or(a, b), Implies(c, d))")
    >>> s.add_weights({"a" : 0.4, "b" : 0.45, "c" : 0.76, "d" : 0.91})
    >>> s.query(mode='Prob').inference 
    tensor(0.6242)
    >>> s.add_conditional("Or(a, ~d)")
    >>> s.query(mode='Prob').inference
    tensor(0.8336)

    """ 
    _stype: str = 'sdd'
    _solver: SymbolicMapper[T] | None = None
    _formulas: list[str | T] | None = None
    _formula_obj: T | None = None
    _conditional: T | str | None = None
    _aux_vars: set[str | int] | None = None
    _weights: WeightManager[I, O] | None = None
    _categoricals: set[str | int] | None = None
    _no_simplify: bool = False

    def __repr__(self) -> str:
        return (
            "Solver(\n"
            f"  num_formulas={len(self.formulas) if self.formulas else 0},\n"
            f"  weight_backend={type(self._weights).__name__ if self._weights else None},\n"
            f"  solver_backend={type(self._solver).__name__ if self._solver else None},\n"
            f"  conditional={True if self._conditional is not None else False},\n"
            ")"
        )

    ## TODO: figure out how to properly replace weights in-place
    @property
    def weights(self) -> WeightManager[I, O] | None:
        """The weight manager used for weighted inference (the weight values are
        in `weights.weights`, ordered as in `weight_names`)."""
        return self._weights
    @property 
    def no_simplify(self) -> bool:
        """Whether to skip simplification of formulas in the solver."""
        return self._no_simplify      
    @property 
    def solver_type(self) -> str | None:
        """The type of the underlying solver."""
        return self._stype
    @property 
    def formulas(self) -> list[str | T] | None:
        """The list of formulas stored in the solver."""
        return self._formulas
    
    @property 
    def formula(self) -> Formula:
        """The joint formula constructed from all added formulas.
        
        Raises: 
            ValueError: If no formulas have been added to the solver.
        """
        if self._formula_obj is None or self._solver is None: 
            raise ValueError("No formulas added to the solver.")

        return self._solver.BuildFormula(self._formula_obj)
    
    @property 
    def solver(self) -> SymbolicMapper[T] | None:
        """The underlying symbolic solver/mapper."""
        return self._solver

    def __post_init__(self) -> None:
        """Post-initialization processing for the Solver dataclass. Builds formulas and conditionals if 
        provided during initialization. Otherwise initializes empty formula list and auxiliary variable set.
        """
        self._aux_vars = self._aux_vars if self._aux_vars is not None else set()
        self._formulas = self._formulas if self._formulas is not None else [] 
        self._categoricals = self._categoricals if self._categoricals is not None else set()
        self._formula_obj = None

        if self._formulas:
            self._add_formulas(self._formulas)
            if self._conditional is not None:
                self._add_conditional(self._conditional)
            self._build_formula_obj()

    def _build_formula_obj(self) -> None:
        """Helper method for building the joint formula object from the list of added 
        formulas and optional conditional formulas.

        When conditioning is provided, an auxiliary variable `_F` is added to the formula to
        represent the conjunction of all formulas, which is then used in the biconditional with
        the conditional formula.

        Raises:
            ValueError: If no underlying solver is available.
        """
        if self._solver is None:
            raise ValueError(
                "No underlying solver available to build formula object."
            )

        con_formula = None 
        if self._formulas: 
            for formula in self._formulas:
                con_formula = (
                    formula
                    if con_formula is None
                    else self._solver.And(con_formula, formula) ## type: ignore
                )
        if self._conditional is not None \
            and con_formula is not None \
            and not isinstance(con_formula, str) \
            and not isinstance(self._conditional, str):
            try: 
                ind_var = self._solver["_F"]
            except (KeyError, ValueError):
                self._solver.add_symbol("_F")
                ind_var = self._solver["_F"]

            self._aux_vars = self._aux_vars or set()
            self._aux_vars.add("_F")      
            indicator = self._solver.Biconditional(
                ind_var, con_formula
            )
            con_formula = self._solver.And(
                indicator, self._conditional
            )

        if con_formula is not None and not isinstance(con_formula, str):
            self._formula_obj = con_formula

        if not self._no_simplify and self._formula_obj is not None:
                self._formula_obj = self._solver.simplify(self._formula_obj)

    ### reset and formula management methods

    def reset(self) -> None:
        """Resets the solver state, clearing formulas and weights, and even 
        the solver itself. 
        """
        self._formulas = []
        self._formula_obj = None
        self._weights = None
        self._solver = None 
        self._conditional = None
        if self._aux_vars and "_F" in self._aux_vars:
            self._aux_vars.remove("_F")

    def remove_conditional(self) -> None:
        """Removes any existing conditional formula from the solver."""
        self._conditional = None
        ## NOTE: `_F` stays in `_aux_vars`, since it remains a (now unused) solver
        ## variable that must be ignored during inference and model enumeration
        self._build_formula_obj()
        
    def pop(self) -> None:
        """Removes the last added formula from the solver's formula list.
        
        Raises:
            ValueError: If there are no formulas to pop.
        """
        if not self._formulas:
            raise ValueError(
                "No formulas to pop, must provide at least one formula."
            )
        self._formulas.pop()
        self._formula_obj = None
        self._build_formula_obj()

    def _add_formulas(self, formulas: str | list[str | T] | T) -> None:
        """Method for adding a formula to the solver's formula list. Multiple formulas
        can be added at once by providing a list of formulas, and a final formula is constructed 
        as the conjunction of all added formulas.

        Arguments: 
            formul (str | list[any]): 
                The formula or list of formulas to add. Can take the form of a string, or 
                    a list of strings or formula objects.
        """
        formulas = [formulas] if not isinstance(formulas, list) else formulas

        if isinstance(formulas[0], str):
            var_list: set[str] = extract_vars(formulas)

            if self._solver is None:
                self._solver = MapperRegistry.build_instance(
                    name=self._stype, var_list=set(var_list)
                )
            for var in var_list:
                if var not in self._solver.var_list:
                    self._solver.add_symbol(var)
                    
            formulas = build_formula(
                formulas=formulas, mapper=self._solver
            )
        if self._formulas is not None:
            self._formulas.extend(formulas)


    def _add_conditional(self, formula: str | T) -> None:
        """Adds a formula to condition on when performing inference with 
        the solver. 

        Arguments:
            formula (str | any): 
                An arbitrary proposition formula, either as a string or
                    as a formula object compatible with the underlying solver.
        Raises:
            ValueError: If no base formulas have been added to the solver yet.

        """
        if self._conditional is not None:
            warnings.warn(
                "Overwriting existing conditional formula...."
            )
        if self._solver is None or not self._formulas:
            raise ValueError(
                "Must add base formulas before adding conditionals."
            )
        
        if isinstance(formula, str):
            var_list: set[str] = extract_vars(formula)
            if not var_list.issubset(set(self._solver.var_list)):
                # warnings.warn(
                #     "Conditioning formula contains variables not in the base formulas..."
                # )
                for var in var_list:
                    if var not in self._solver.var_list:
                        self._solver.add_symbol(var)
            formula = build_formula(
                formulas=[formula], mapper=self._solver
            )[0]

        self._conditional = formula

    ### public methods for adding formulas and conditionals

    def add_formulas(self, formulas: str | list[str | T]) -> None:
        """Public method for adding formulas to the solver. Wraps the internal 
        `_add_formulas` method and rebuilds the joint formula object after adding.

        Arguments: 
            formul (str | list[any]): 
                The formula or list of formulas to add. Can take the form of a string, or 
                    a list of strings or formula objects.
        """
        self._add_formulas(formulas)
        self._build_formula_obj()
        
    add = add_formulas  # alias for _add_formulas method

    def add_conditional(self, formula: str | T) -> None:
        """Public method for adding a conditional formula to the solver. 
        Wraps the internal `_add_conditional` method and rebuilds the joint 
        formula object after adding.

        Arguments:
            formula (str | any): 
                An arbitrary proposition formula, either as a string or
                as a formula object compatible with the underlying solver. 
        """
        self._add_conditional(formula)
        self._build_formula_obj()

    ### solver convenience functions for accessing underlying solver features

    def BoolOps(self, op_names: str | list[str | int]) -> list[Callable[..., T]]: 
        """Helper method for getting boolean operations from the underlying solver. Useful
        for building formula. 

        Arguments: 
            op_names (str | list[str | int]): 
                A string of operation names separated by spaces or a list.

        Example usage: 

        >>> from symprog import Solver
        >>> s = Solver.from_kwargs(solver='sdd', var_list=['a', 'b'])
        >>> And, Or, Not = s.BoolOps('And Or Not')
        
        Returns:
            A list of callable boolean operations from the underlying solver.
        """
        if self._solver is None:
            raise ValueError(
                "No underlying solver available to retrieve boolean operations."
            )
        bool_names = (
            op_names.split() if isinstance(op_names, str) else op_names
        )
        return [getattr(self._solver, str(name).strip()) for name in bool_names]

    def Variables(self, var_str: str | list[str | int]) -> list[T]:
        """Helper method for getting variables from the underlying solver.

        Arguments: 
            var_str (str | list[str | int]): 
                A string of variable names separated by spaces or a 
                    list of variable names.

        Example usage: 

        >>> from symprog import Solver
        >>> s = Solver.from_kwargs(solver='sdd', var_list=['a', 'b'])
        >>> a, b = s.Variables('a b')

        Returns:
            A list of variable objects from the underlying solver.
        Raises: 
            ValueError: If no underlying solver is available.
        """
        if self._solver is None:
            raise ValueError(
                "No underlying solver available to retrieve variables."
            )
        var_names = (
            var_str.split() if isinstance(var_str, str) else var_str
        )
        return [self._solver[name] for name in var_names]

    def PrettyPrinting(self, formula: T | None = None) -> Formula:
        """Helper method for pretty-printing a formula.

        Arguments: 
            formula: The formula string to pretty-print.
        """
        formula = formula or self._formula_obj
        if formula is None or self._solver is None: 
            raise ValueError(
                "No solver/formula available for pretty-printing."
            )        
        return self._solver.BuildFormula(formula)
    
    ## inferences and model counting methods 
    
    def models(self,formula: T | None = None, **kwargs) -> Any:
        """Method for enumerating the models of a formula. 

        Arguments:
            formula: The (optional) formula to retrieve models from. If not provided,
                uses the formula added via `add_formulas`.

        Raises:
            ValueError: If no valid formula is provided or a solver is not initialized.
        """
        formula = formula or self._formula_obj
        if formula is None or self._solver is None:
            raise ValueError("Must specify a valid formula and solver to retrieve models")
        if not self._aux_vars:
            return self._solver.Models(formula, **kwargs)
        return self._project_models(self._solver.Models(formula, **kwargs))

    def _project_models(self, models: Any) -> Any:
        """Removes auxiliary variables from models, skipping duplicate projections."""
        seen = set()
        for model in models:
            projected = {k: v for k, v in model.items() if str(k) not in self._aux_vars}
            key = frozenset(projected.items())
            if key not in seen:
                seen.add(key)
                yield projected
    
    def _prepare_conditional(
        self, evidence: set[str | int] | str, query: set[str | int] | str
    ) -> tuple[set[str | int], set[str | int], set[str | int], bool]:  
        """Helper method for preparing the solver input to compute different types of conditionals.

        In the solver, a load formula is stored as `_formula_obj` and when conditioning is added, 
        an auxiliary variable `_F` is added to the formula to represent the conjunction of all formulas, 
        which is then used in the biconditional with the conditional formula. 
        When performing inference with conditioning, the solver conditions on the auxiliary variable `_F` 
        to compute the desired conditional probabilities. 
        
        """
        ## determines if normalization constant is needed for conditional inference
        right_comp = False 
        conditionals = set() 
        
        ## map evidence and query literal strings to sets 
        ## (copies user-provided collections, which are modified below)
        if isinstance(evidence, str):
            evidence = set([l.strip() for l in evidence.split()])
        else:
            evidence = set(evidence)
        if isinstance(query, str):
            query = set([l.strip() for l in query.split()])
        else:
            query = set(query)
                    
        ### if conditional is added, enforce main formula via `_F``
        if self._conditional is not None: 
            evidence.add("_F")
            right_comp = True
            if query:
                conditionals.add(f"_F")
                for e in evidence: 
                    conditionals.add(e)

        ### if there's a query and evidence, add evidence to right side. 
        if query and evidence and self._conditional is None: 
            right_comp = True 
            for e in evidence: conditionals.add(e)
        if query and not evidence and self._conditional is None:
            ## normalize by P(F), so that queries give P(q | F) (not the joint P(F, q))
            right_comp = True
        if not query and evidence and self._conditional is None: 
            right_comp = False
        if not query and not evidence and self._conditional is None: 
            right_comp = False 

        return (query, evidence, conditionals, right_comp)
                

    def inference(
        self, 
        formula: T | list[T | str] | None | str = None, 
        conditional: T | str | None = None, 
        mode: str = "Prob", 
        fast: bool = False, 
        evidence: set[str | int] | str | None = None, 
        query: set[str | int] | str | None = None,
        symbolic: bool = False,
        symbolic_backend: str = 'sympy',
        weights_mgr: WeightManager | None = None,
        incremental_simplify: bool = False,
        simplify_threshold: int = 4,
        **kwargs
    ) -> SolverOut:
        """Method for performing inference over the given formula. When no formula is provided, 
        uses the formula added via `add_formulas`. When a formula is provided, it will be processed 
        on the fly and overrides any existing formulas in the solver, as well as any existing 
        conditional formulas. If `fast` is set to True, will use fast counting methods (if available).

        When conditioning is provided, an auxiliary variable `_F` is added to the formula to 
        represent the conditional.

        Arguments: 
            mode (str): 
                The type of inference to perform (e.g. probabilistic, viterbi, entropy). 
            formula (any): 
                The (optional) formula to perform inference over. If not provided, 
                uses the formula added via `add_formulas` and stored in the solver.
            fast (bool): 
                Whether to use fast counting methods (if available) for counting mode. These 
                are typically native solver methods built into the different solver interfaces
                that avoid manual semiring computation. Defaults to False.
            evidence (set[str | int] | str): 
                An optional set of evidence literals to condition on.
            symbolic (bool):
                Whether to compute inferences symbolically, which results in symbolic expressions
                rather than numerical values. Defaults to False. Note: this might result in very
                large expressions for complex formulas and should be used carefully. 
            symbolic_backend (str):
                The backend to use for symbolic computation (default='sympy').
            conditional (T | str | None): 
                An optional conditional formula to condition on during inference. Overrides any
                existing conditional formula added via `add_conditional`.
            query (set[str | int] | str | None):
                An optional set of query literals to compute. When provided, the solver will compute the
                probability of the query literals conditioned on the formulas, the evidence and any
                conditional formulas, i.e., P(query | F, evidence, conditional), rather than computing
                the probability of the entire formula.
            
        Raises: 
            ValueError: If weights are not provided for weighted inference or 
                if no valid formulas are provided.
        """
        evidence = evidence or set()
        conditionals = set()
        query = query or set()
        if isinstance(query, str):
            marginal_query = query
        else: 
            marginal_query = " & ".join([str(l) for l in query]) if query else None
        
        ## build new formulas on the fly, if provided
        self._set_formulas(formula, conditional)

        if self._formula_obj is None or self._solver is None:
            raise ValueError(
                "No formulas or solver available for inference. Please call with `formula`" \
                " argument or add formulas via `add_formulas` method."
            )
        
        ### build evidence
        query, evidence, conditionals, compute_cond = self._prepare_conditional(
            evidence, query
        )
        for literal in evidence | query:
            name = str(literal).replace("~", "").strip()
            try:
                self._solver[name]
            except (KeyError, ValueError):
                raise ValueError(
                    f"Unknown variable in evidence or query: {name}"
                ) from None

        ### perform inference
        if mode.lower() == "sat":
            if not (evidence or query):
                return self._solver.SAT(self._formula_obj)
            ## conjoin evidence (and query) literals with the formula
            sat_formula = self._formula_obj
            for literal in evidence | query:
                name = str(literal).replace("~", "").strip()
                symbol = self._solver.Symbol(name, return_val=True)
                if "~" in str(literal):
                    symbol = self._solver.Not(symbol)
                sat_formula = self._solver.And(sat_formula, symbol)
            return self._solver.SAT(sat_formula)
        elif fast and mode.lower() == "count" and not (evidence or query or self._aux_vars):
            return self._solver.Count(self._formula_obj)
        else: 
            if self._weights is None and symbolic is False:
                raise ValueError("Weights must be provided for weighted inference.")

            if (symbolic is True and weights_mgr is None) or (mode == "Formula" and weights_mgr is None):
                boolean = True if mode == "Formula" else False
                
                ## build dummpy weights for symbolic inference
                weights = self._process_weights(
                    weights=None,
                    var_list=self._solver.var_list,
                    symbolic=True,
                    container_type=symbolic_backend,
                    boolean=boolean,
                )
            elif weights_mgr is not None: 
                weights = weights_mgr
            else: 
                weights = self._weights 

            semiring = SemiringRegistry.build_instance(
                name=mode,
                weight_mgr=weights, 
            )
            wmc_output = self._solver.wmc_interface(
                semiring=semiring, 
                formula=self._formula_obj, 
                evidence=evidence,
                aux_vars=self._aux_vars,
                conditioning=conditionals, 
                categoricals=self._categoricals,
                query=query,
                incremental_simplify=incremental_simplify,
                simplify_threshold=simplify_threshold,
                **kwargs
            )
            return semiring.process_output(
                wmc_output,
                conditional=compute_cond,
                marginal=marginal_query,
                **kwargs
            )

    query = inference  # alias for inference method

    def symbolic_expr(
        self,
        formula: T | list[T | str] | None | str = None, 
        conditional: T | str | None = None, 
        mode: str = "Prob", 
        evidence: set[str | int] | str | None = None, 
        query: set[str | int] | str | None = None,
        symbolic_backend: str = 'sympy',
        simplify: bool = False,
        incremental_simplify: bool = False,
        simplify_threshold: int = 4,
        **kwargs
    ) -> SolverOut: 
        """Method for retrieving the symbolic expression of a formula computation. For example,
        this can be used for finding the symbolic polynomial representation of a weighted model count,
        which might be used for further analysis or manipulation.   

        Arguments:
            formula (T | None): 
                The (optional) formula to retrieve the symbolic expression from. If not provided,   
                uses the formula added via `add_formulas`. 
            mode (str): 
                The type of inference to perform for generating the symbolic expression. Defaults to 'Prob'
                for probabilistic inference and WMC.
            evidence (set[str | int] | str | None): 
                An optional set of evidence literals to condition on.
            symbolic_backend (str): 
                The backend compute algebra framework to use for building a symbolic expression. 
                Defaults to using sympy.  
            conditional (T | str | None): 
                An optional conditional formula to condition on during inference. Overrides any
                existing conditional formula added via `add_conditional`.  

        Returns:
            A SolverOut object containing the symbolic expression of the inference.
        
        """
        output = self.inference(
            formula=formula,
            conditional=conditional,
            mode=mode,
            fast=False,
            evidence=evidence,
            symbolic=True,
            symbolic_backend=symbolic_backend,
            query=query,
            incremental_simplify=incremental_simplify,
            simplify_threshold=simplify_threshold,
            **kwargs
        ) 
        if simplify and hasattr(output.inference, 'simplify'):
            output.inference = output.inference.simplify()
        return output
    
    def gradient(
        self, 
        formula: T | list[T | str] | None | str = None, 
        conditional: T | str | None = None, 
        mode: str = "Prob", 
        fast: bool = False, 
        evidence: set[str | int] | str | None = None, 
        query: set[str | int] | str | None = None,
        symbolic: bool = False,
        symbolic_backend: str = 'sympy',
        key: str | None = None,
        boolean: bool = False,
        **kwargs
    ) -> Any:
        """Method for computing the gradients vector of weight parameters based on a solver 
        inference and inference output. 

        Arguments:
            f (any): 
                The output of a function computation (e.g., a WMC count tensor or symbolic formula)
            key (Optional[str]): 
                An optional key to retrieve a specific gradient component.
        
        Raises:
            ValueError: If no weights are available for gradient computation.
        """
        if formula is None and self._formula_obj is None:
            raise ValueError(
                "Must provide a formula for gradient computation."
            )
        if (self._weights is None) or (self._solver is None):
            raise ValueError(
                "Weights and solver must be provided for gradient computation."
            )

        weight_mgr = None 
        if symbolic is True or boolean is True: 
            weight_mgr = self._process_weights(
                weights=None,
                var_list=self._solver.var_list,
                symbolic=True,
                container_type=symbolic_backend,
                boolean=boolean,
            )
        else: 
            weight_mgr = self._weights

        if weight_mgr is not None:
            weight_mgr.requires_grad = True

        if boolean is True:
            mode = "Formula"

        ## backends such as JAX differentiate functions of the weights rather than computed values
        if getattr(weight_mgr, "functional_gradients", False):
            f = self.as_function(
                formula=formula,
                conditional=conditional,
                mode=mode,
                evidence=evidence,
                query=query,
                **kwargs
            )
            return weight_mgr.gradient(f, key=key, **kwargs)

        output = self.inference(
            formula=formula,
            conditional=conditional,
            mode=mode,
            fast=fast,
            evidence=evidence,
            query=query,
            symbolic=symbolic,
            symbolic_backend=symbolic_backend,
            weights_mgr=weight_mgr,
            **kwargs
        )
        return weight_mgr.gradient(
            output.inference, 
            key=key,
            boolean=boolean,
            **kwargs
        )
    
    grad = gradient

    def as_function(
        self,
        formula: T | list[T | str] | None | str = None,
        conditional: T | str | None = None,
        mode: str = "Prob",
        evidence: set[str | int] | str | None = None,
        query: set[str | int] | str | None = None,
        **kwargs
    ) -> Callable[[Any], Any]:
        """Returns the inference as a function of the weight vector, `f(weights) -> value`,
        where `weights` is ordered as in `weight_names`. Useful for function transformations
        such as `jax.grad`, `jax.jit` and `jax.vmap` with the (experimental) JAX backend:

        >>> s = Solver()
        >>> s.add(["Implies(a, b)", "And(Or(a, b), Implies(a, c))"])
        >>> s.add_weights({"a": 0.9, "b": 0.64, "c": 0.34}, container_type="jax")
        >>> f = s.as_function(mode="Prob")
        >>> jax.jit(jax.grad(f))(jnp.array([0.9, 0.64, 0.34]))
        Array([-0.4224, 0.406, 0.576], dtype=float32)

        Arguments:
            formula, conditional, mode, evidence, query:
                As in `inference`; formulas and conditionals are built once, when the
                function is created.

        Raises:
            ValueError: If no weights have been added to the solver.
        """
        if self._weights is None:
            raise ValueError("Weights must be added before building an inference function.")
        self._set_formulas(formula, conditional)
        weight_mgr = self._weights

        def f(weights: Any) -> Any:
            original = weight_mgr.parameters["weights"]
            weight_mgr.parameters["weights"] = weights
            try:
                return self.inference(
                    mode=mode, evidence=evidence, query=query, **kwargs
                ).inference
            finally:
                weight_mgr.parameters["weights"] = original

        return f

    @property
    def weight_names(self) -> list[str]:
        """The variable names corresponding to each position of the weight vector
        (e.g., in gradients and in `as_function`)."""
        if self._weights is None:
            return []
        wmap = self._weights.wmap
        return sorted(wmap, key=wmap.get)

    def _set_formulas(
        self, formula: T | list[T | str] | None | str = None, conditional: T | str | None = None
    ) -> None:
        """Replaces the formulas and/or conditional formula, if provided, and rebuilds
        the joint formula object."""
        if formula is not None or conditional is not None:
            if formula is not None:
                self._formulas = []
                self._add_formulas(formula)
            if conditional is not None:
                self._add_conditional(conditional)
            self._build_formula_obj()

    ## weight management methods

    @staticmethod 
    def _process_weights(
        weights: dict[str | int, Any] | list[tuple[str | int, Any]] | None,
        weight_map: dict[str | int, int] | None = None, 
        container_type: str = 'fast_pytorch',
        symbolic: bool = False,
        var_list: list[str | int] | None = None,
        boolean: bool = False,
        **kwargs,
    ) -> WeightManager[I,O]:
        """Helper class for processing incoming weights.

        Arguments: 
            weights (dict[str | int, any] | list[tuple[str | int, any]]): 
                The incoming weights, either as a dict mapping variable names to weights, 
                or as a list of (variable name, weight) tuples.
            weight_map (Optional[dict[str | int, int]]):
                The (optional) map from variable names to weight positions.
            container_type (Optional[str]):
                The type of backup numerical framework to use (default='pytorch')
            symbolic (Optional[bool]):
                Whether the weights are symbolic for performing symbolic computation
                  (default=False). 
        
        Raises: 
            ValueError: If both weights as a dict and a weight map are provided or 
            if weights are provided as a list but no weight map is given.
        """
        if isinstance(weights, dict): 
            if weight_map is not None: 
                raise ValueError("Cannot provide both weights as a dict and a weight map.")
            
            sorted_vars = sorted(weights.keys())
            weight_map = {k: v for v, k in enumerate(sorted_vars)}
            weights = [weights[k] for k in sorted_vars]

        elif isinstance(weights, list) and weight_map is None:
            updated_weights = [] 
            weight_map = {}

            for weight in weights: 
                if isinstance(weight, (tuple, list)) and len(weight) == 2: 
                    lit, w = weight 
                    weight_map[lit] = len(updated_weights)
                    updated_weights.append(w)
                else: 
                    raise ValueError(
                        "Weights provided as a list must be in the form of (literal, weight) tuples."
                    )
                
            sorted_vars = sorted(weight_map.keys())
            weights = [updated_weights[weight_map[k]] for k in sorted_vars]

        elif symbolic is True and var_list is not None:
            ### build dummy weights for symbolic computation 
            updated_weights = [] 
            weight_map = {}

            for var in var_list: 
                weight_map[var] = len(updated_weights)
                updated_weights.append(1.0)

            sorted_vars = sorted(weight_map.keys())
            weights = [updated_weights[weight_map[k]] for k in sorted_vars]

        elif weights is None or not weight_map: 
            raise ValueError(
                "Weight map must be provided when weights are given as a list."
            )
        
        ### check that weights and weight map are aligned
        if weights is not None: 
            for k, v in weight_map.items(): 
                if v >= len(weights):
                    raise ValueError(
                        f"Weight index {v} for literal {k} exceeds the length of the weights list."
                    )   

        return WeightRegistry.build_instance(
            name=container_type, 
            weights=weights, 
            wmap=weight_map, 
            symbolic=symbolic, 
            boolean=boolean,
            **kwargs
        )   

    def add_weights(
        self, 
        weights: Any, 
        weight_map: dict[str | int, int] | None = None, 
        container_type: str = 'pytorch', 
    ) -> None:
        """Assign weights to the current formula variables.  

        Arguments: 
            weights (any): 
                The incoming weights (expressed in a particular container type and format). 
            weight_map (Optional[dict[str | int, int]]): 
                The (optional) map from variable names to weight positions. 
            container_type (Optional[str]): 
                The type of backup numerical framework to use (default='pytorch')
            requires_grad (Optional[bool]): 
                Whether the weights require gradient computation (default=False)

        Raises: 
            ValueError whenever unknown variables are encountered or the value list is 
                not complete or not initial formula is provided. 
        """
        if self._solver is None or not self._formulas: 
            raise ValueError(f'Must provide formula before adding weights')
        if self._weights is not None:
            warnings.warn(
                "Overwriting existing weights. Consider using `object.weights = ...` " \
                "instead to modify weights in place."
            )
        self._weights = self._process_weights(
            weights, weight_map, container_type
        )
        # if self._solver is None and set(self._solver.var_list) != set(self._weights._wmap.keys()):
        #     raise ValueError("Weight map keys do not align with solver variables")

    ### class methods for object construction 

    @classmethod 
    def from_cnf(
        cls,
        stype: str = 'sdd',
        weight_type: str = 'pytorch',
        var_list: list[str | int] | None = None,
        var_weights: dict[str | int, Any] | None = None,
        formulas: list[str] | str | None = None,
        conditional: str | None = None,
        aux_vars: set[str | int] | None = None,
        categoricals: set[str | int] | None = None,
        requires_grad: bool = False,
        tactic: tuple[str, ...] = ('tseitin-cnf',),
        no_simplify: bool = False,
        **kwargs,
    ): 
        """Class method for constructing a Solver instance through initial translation
        into CNF format. Specifically designed for large formulas that require careful 
        formula or other structure optimization (e.g., variable ordering, decomposition). 

        Arguments:
            stype (str):
                The type of the underlying solver (default='sdd').
            weight_type (str):
                The type of weight manager to build (default='pytorch').    
            var_list (list | None):
                An optional list of variables to initialize the solver with.
            formulas (list | str | None):
                An optional list of formulas to initialize the solver with.
            conditional (str | None):
                An optional conditional formula to add.
            no_simplify (bool):
                Whether to skip formula simplification (default=False).

        Raises: 
            ValueError: If no formulas are provided for CNF translation.
        """
        if formulas is None: 
            raise ValueError("Initialization through CNF requires specifying formulas upfront.")
        if var_list is None: 
            var_list = list(extract_vars(formulas))

        # if conditional is not None: 
        #     warnings.warn(
        #         "Conditionals not supported for CNF translation, skipping..." \
        #         "Consider adding conditionals after initialization."
        #     )
        #     conditional = None 

        solver, formula, new_aux_vars = build_from_cnf(
            stype=stype,
            formulas=formulas, 
            variables=var_list, 
            tactic=tactic
        )
        if new_aux_vars:
            aux_vars = aux_vars or set()
            aux_vars.update(new_aux_vars)

        return cls.from_kwargs(
            weight_type=weight_type,
            solver=solver,
            formulas=formula,
            conditional=conditional,
            no_simplify=no_simplify, 
            aux_vars=aux_vars,
            categoricals=categoricals,
            requires_grad=requires_grad,
            var_weights=var_weights,
            **kwargs,
        )

    @classmethod
    def from_kwargs(
        cls, 
        stype: str = 'sdd',
        weight_type: str = 'pytorch',
        solver: SymbolicMapper[T] | None = None, 
        var_list: list[str | int] | None = None,
        var_weights: dict[str | int, Any] | None = None,
        weights: WeightManager[I, O] | None = None,
        formulas: list[str | T] | str | None = None,
        conditional: T | str | None = None,
        aux_vars: set[str | int] | None = None,
        categoricals: set[str | int] | None = None,
        requires_grad: bool = False,
        no_simplify: bool = False,
        **kwargs,
    ):
        """Class method for constructing a Solver instance from keyword arguments. Builds solvers 
        and weight managers on-the-fly if not provided, otherwise leaves formula and conditional 
        formula processing to the constructor. 

        Arguments:
            stype (str): 
                The type of the underlying solver (default='sdd').
            solver (SymbolicMapper): 
                The underlying symbolic solver/mapper.
            weights (WeightManager): 
                The weight manager for performing weighted inference.
            formulas (list | str): 
                An optional list of formulas stored in the solver.
            conditional (any): 
                An optional conditional formula to add.
            aux_vars (set[str | int]): 
                An optional set of auxiliary unweighted variables to ignore 
                    during inference and weighting (used for encoding conditionals).
            weight_type (str):
                The type of weight manager to build (default='pytorch').

        Example quick usage: 

        >>> from symprog import Solver
        >>> s = Solver.from_kwargs(solver='sdd', var_list=['a', 'b', 'c'])
        
        Returns:
            An instance of the Solver class.
        
        """
        if var_list:
            if solver is not None: 
                warnings.warn(
                    "Both solver and var_list provided. Ignoring var_list and using " \
                    "provided solver."
                )
            else: 
                solver = MapperRegistry.build_instance(
                    name=stype, var_list=set(var_list), **kwargs
                )   

        if var_weights:
            if weights is not None:
                warnings.warn(
                    "Both weights and var_weights provided. Ignoring var_weights "
                )
            weights = cls._process_weights(
                weights=var_weights, 
                container_type=weight_type,
                requires_grad=requires_grad,
                **kwargs
            )

        formulas = formulas if isinstance(formulas, list) \
                else [formulas] if formulas is not None else None
        
        #if solver is None and 
        
        return cls(
            _stype=stype,
            _solver=solver,
            _weights=weights, 
            _formulas=formulas,
            _conditional=conditional,
            _aux_vars=aux_vars,
            _categoricals=categoricals,
            _no_simplify=no_simplify,
        )
    
    ### tensorization methods
    
    def tensorize(
        self,
        formula: T | list[T | str] | None | str = None,
        conditional: T | str | None = None,
        mode: str = "Prob",
        evidence: set[str | int] | str | None = None,
        query: set[str | int] | str | None = None,
        simplify: bool = False,
        try_horner: bool = False,
        symbolic_backend: str = 'symengine',
        incremental_simplify: bool = False,
        simplify_threshold: int = 4,
        backend: str = "fx",
        **kwargs
    ):
        """EXPERIMENTAL: Compiles an inference into a PyTorch module for efficient batched 
        computation. The module is called as `mod(x)` with `x[..., D]` ordered as 
        `mod.input_symbols`, or as `mod(a, b, ...)` or `mod(a=..., b=...)`.

        Arguments:
            formula, conditional, mode, evidence, query:
                As in `inference`.
            simplify, try_horner, symbolic_backend, incremental_simplify, simplify_threshold:
                Options for building and simplifying the symbolic expression (`fx` backend).
            backend (str):
                `"fx"` (default) compiles the symbolic expression of the inference with `torch.fx`, 
                and supports all modes, evidence, queries and conditionals. `"klay"` compiles the 
                circuit with KLay (requires `pip install "symprog[klay]"`), and supports the 
                `Prob`, `LogProb`, `Viterbi`, `LogViterbi` and `Fuzzy` modes, without evidence, 
                queries or conditionals.

        Raises:
            ValueError: If no underlying solver is available or the backend is unknown.
            NotImplementedError: If the `klay` backend is used with unsupported options.
        """
        if self._solver is None:
            raise ValueError(
                "No underlying solver available for tensorization."
            )
        if backend not in ("fx", "klay"):
            raise ValueError(f"Unknown tensorization backend: {backend} (expected 'fx' or 'klay').")
        warnings.warn(
            "Tensorization is experimental, and its API may change.",
            ExperimentalWarning,
            stacklevel=2,
        )

        if backend == "klay":
            return self._tensorize_klay(
                formula=formula, conditional=conditional, mode=mode, evidence=evidence, query=query, **kwargs
            )

        symbolic_formula = self.symbolic_expr(
            formula=formula,
            conditional=conditional,
            mode=mode,
            evidence=evidence,
            query=query,
            symbolic_backend=symbolic_backend,
            simplify=simplify,
            incremental_simplify=incremental_simplify,
            simplify_threshold=simplify_threshold,
            **kwargs
        )
        module = symbolic_to_fx(
            expressions=[symbolic_formula.inference], 
            try_horner=try_horner,
             **kwargs
        )
        return module

    def _tensorize_klay(
        self,
        formula: T | list[T | str] | None | str = None,
        conditional: T | str | None = None,
        mode: str = "Prob",
        evidence: set[str | int] | str | None = None,
        query: set[str | int] | str | None = None,
        compile: bool = False,
        **kwargs
    ):
        """Tensorization with KLay (see `tensorize`), for the modes in `KLAY_MODES` 
        (`Prob`, `LogProb`, `Viterbi`, `LogViterbi`, `Fuzzy`), without evidence, queries 
        or conditionals."""
        if mode not in KLAY_MODES:
            raise NotImplementedError(
                f"The klay backend supports the modes {', '.join(KLAY_MODES)} (got '{mode}')."
            )
        if evidence or query:
            raise NotImplementedError("The klay backend does not support evidence or queries.")
        if self._stype != "sdd":
            raise NotImplementedError("The klay backend requires the 'sdd' solver.")
        if conditional is not None or self._conditional is not None:
            raise NotImplementedError("The klay backend does not support conditional formulas.")

        self._set_formulas(formula)
        if self._formula_obj is None:
            raise ValueError("No formulas available for tensorization.")

        ## module inputs are all (non-auxiliary) variables of the solver
        hidden = set(self._aux_vars or set()) | set(getattr(self._solver, "_hidden_vars", set()))
        input_symbols = sorted(name for name in self._solver.symbol_map.values() if name not in hidden)
        return klay_module(self._solver, self._formula_obj, input_symbols, mode=mode, compile=compile)

