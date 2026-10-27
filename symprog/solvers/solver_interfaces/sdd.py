import io
from copy import deepcopy
from typing import Optional, Any
from pysdd.sdd import SddManager, SddNode
from pysdd.iterator import SddIterator
from pysdd.util import sdd_to_dot

from contextlib import redirect_stdout

from ..semiring import ProductSemiring
from ..mapper import SymbolicMapper, MapperRegistry, ModelOutput

__all__ = ["SddMapper"]

@MapperRegistry(name = "sdd")
class SddMapper(SymbolicMapper[SddNode]):
    """Symbolic mapper object that maps into sdd circuits using pysdd.

    Attributes: 
        num_variables (int): 
            The number of variables in the program and logic encoding.
        manager (SddManager): 
            The core sdd manager attribute for sdd mappers. 
        false (SddNode): 
            Representation of logical `false` for the mapper. 
        true (SddNode): 
            Representation of logical `true` for this mapping.

    Methods: 
        wmc_interface(formula, semiring, evidence, aux_vars, conditioning, **kwargs): 
            Interface for running weighted model counting with semirings. Takes a given `formula`
            expressed in the SDD format and computes the WMC using the provided `semiring` instance.
            Also allows for optional conditioning on `evidence` (i.e., specifical literals) and more 
            advanced condition on arbitrary boolean formulas via details in`conditioning`. 
        add_symbol(symbol: str | int): 
            Adds a symbol to the the sdd manager and mapper. Allows for on-the-fly new symbolic creation
            using the sdd manager. 
        Or(obj1, obj2): 
            Implementation of logical `Or`.
        And(obj1, obj2): 
            Implementation of logical `And`.
        Implies(obj1, obj2): 
            Implementation of logical material implication.
        Biconditional(obj1, obj2): 
            Implementation of logical biconditional.
        Symbol(name, return_val=False): 
            Representation and creation on primitive symbols.
        Not(obj): 
            Implementation of logical negation. 

    Example usage in the global library: 

    >>> from symprog.solvers.solver_interfaces import SddMapper
    >>> mapper = SddMapper(var_list=["A", "B", "C"])
    >>> A = mapper.Symbol("A", return_val=True)
    >>> B = mapper.Symbol("B", return_val=True)
    >>> C = mapper.Symbol("C", return_val=True)
    >>> formula = mapper.And(A, mapper.Or(B, mapper.Not(C)))
    >>> print(formula)
    >>> # Output: SddNode representation of the formula A ∧ (B ∨ ¬C)

    """
    def _init_symbols(self) -> None:
        """Initializes the `SddManager` and loads variable map."""
        num_vars = len(self.var_list)
        self.manager: SddManager = SddManager(var_count=num_vars)
        self.manager.auto_gc_and_minimize_on()

        for v, var in enumerate(self.var_list):
            var_sdd = self.manager[v + 1]
            self._sym_map[str(var)] = var_sdd

    def add_symbol(self, symbol: str | int) -> None:
        """Adds a symbol to the the sdd manager and mapper.
        
        Arguments: 
            symbol: 
                The symbol to be added to the underlying sdd symbol 
                table. 
        Raises: 
            ValueError: 
                If the symbol is already in the map, flags that a duplicate
                items was attempted to be added to the symbol table.  

        """
        symbol = str(symbol)
        if symbol in self._sym_map:
            raise ValueError(f"Symbol already in map: {symbol}")
        assert symbol not in self.var_list, "Symbol already in var_list"

        self.manager.add_var_after_last()
        var_sdd = self.manager.vars[self.manager.var_count()]
        self._sym_map[str(symbol)] = var_sdd

    def Or(self, obj1: SddNode, obj2: SddNode) -> SddNode:
        """Implementation of logical `Or`.

        Arguments: 
            obj1: The first argument for disjunction. 
            obj2: The second argument for disjunction
        """
        return self.manager.disjoin(obj1, obj2)
    
    def And(self, obj1: SddNode, obj2: SddNode) -> SddNode:
        """Implementation of logical `And`.

        Arguments: 
            obj1: The first argument for conjunction. 
            obj2: The second argument for conjunction. 
        """
        return self.manager.conjoin(obj1, obj2)

    def Implies(self, obj1: SddNode, obj2: SddNode) -> SddNode:
        """Implementation of logical material implication. 

        Arguments: 
            obj1: The first argument for implication. 
            obj2: The second argument for implication. 
        """
        return self.manager.disjoin(~obj1, obj2)
    
    def Biconditional(self, obj1: SddNode, obj2: SddNode) -> SddNode:
        """Implementation of logical material implication. 

        Arguments: 
            obj1: The first argument for implication. 
            obj2: The second argument for implication. 
        """
        return self.And(
            self.Implies(obj1, obj2),
            self.Implies(obj2, obj1)
        )
    
    def Symbol(
        self, name: str | int, return_val: Optional[bool] = False
    ) -> str | SddNode:
        """Representation and creation on primitive symbols.

        Arguments: 
            name: The name of the symbol. 
        """
        name = str(name)
        
        intn = self._sym_map.get(name, None)
        if intn is None:
            raise ValueError(f'Unknown circuit variable encountered: {name}')
        if return_val:
            return self._sym_map[name]
        return name
 
    def Not(self, expr: SddNode) -> SddNode:
        """Implementation of logical negation.

        Arguments: 
            expr: The expression and sdd circuit to negate. 
        """
        return ~expr
    
    @property
    def false(self) -> SddNode:
        """Internal representation of the false primitive."""
        return self.manager.false()
    
    @property
    def true(self) -> SddNode:
        """Internal representation of the true primitive."""
        return self.manager.true
    
    @property  
    def symbol_map(self) -> dict[int, str]:
        """Returns the internal symbol map for the mapper."""
        return {n.literal : vname for vname, n in self._sym_map.items()}


    def _to_dot(self, formula: SddNode) -> str:
        """Method for (optionally) mapping formula to dot visualization.

        Arguments: 
            formula (SddNode): 
                The formula to map to a dot format. Uses the internal pysdd 
                dot function to create.
        Returns: 
            The string encoding the dot graph.  
        """
        lit_map = {}
        for var, identifier in self._sym_map.items():
            lit_map[-identifier.literal] = f"~{var}"
            lit_map[identifier.literal] = f"{var}"
        
        return sdd_to_dot(formula, litnamemap=lit_map)

    def _sat_interface(self, formula: SddNode, **kwargs) -> tuple[str, None]:
        """Interface for running SAT solving. Given that SDDs are `canonical`, we 
        can check for satisfiability be seeing if the formula is equal to `sdd.false` (unsat)
        or not. 

        Arguments: 
            formula (sddNode): The formula to check for satisfiability. 
        """
        sat = "yes" if formula != self.manager.false() else "no"
        details = None 
        return (sat, details)

    def _count_interface(self, formula: SddNode, **kwargs) -> tuple[int, None]:
        """Interface for model counting over all variables in the manager. Uses
        `global_model_count`, which (unlike `formula.wmc`) does not lock the manager
        against later transformations.

        Arguments:
            formula: The formula to count models for.
        """
        count = formula.global_model_count()
        ## variables existentially quantified away (e.g., Tseitin variables) are free
        count //= 2 ** len(self._hidden_vars)
        return (count, None)

    def _model_enumeration(self, formula: SddNode, **kwargs) -> ModelOutput:
        """Interface for enumerating models. Returns a generator
        object. Models range over all variables in the manager (not only those that
        remain in the formula after simplification), excluding hidden auxiliary variables.

        Arguments:
            formula: The formula to check for satisfiability.
        """
        if formula.is_false():
            ## unsatisfiable: no models (pysdd raises an error instead)
            return
        rev_map = {iden.literal : var for var, iden  in self._sym_map.items()}
        l = {0 : False, 1 : True}
        seen = set()
        for model in formula.models(vtree=self.manager.vtree()):
            assignment = {
                rev_map[k] : l[v] for k,v in model.items()
                if rev_map[k] not in self._hidden_vars
            }
            if self._hidden_vars:
                key = frozenset(assignment.items())
                if key in seen:
                    continue
                seen.add(key)
            yield assignment

    @property
    def _hidden_vars(self) -> set[str]:
        """Auxiliary variables that were existentially quantified out of the formula."""
        return getattr(self, "_quantified_vars", set())

    def simplify(self, formula: SddNode, **kwargs) -> SddNode:
        """Simplifies the given formula using the underlying SDD manager's simplification capabilities. 
        Returns the simplified formula. 

        Arguments:
            formula: The formula to simplify.
        """
        self.manager.minimize()
        return formula

    def wmc_interface(
        self, 
        formula: SddNode, 
        semiring: ProductSemiring, 
        evidence: set[tuple[int, str]] | None = None,
        aux_vars: set[str] | None = None,
        conditioning: set[tuple[int, str]] | None = None,
        categoricals: set[str] | None = None,
        query: set[str] | None = None,
        incremental_simplify: bool = False,
        simplify_threshold: int = 4,
        **kwargs
    ):
        """Interface for running weighted model counting with semirings. Internally, 
        builds an `SddIterator` object for traversing the SDD and computing the WMC via
        the semiring instance. 

        Note: borrows heavily from the problog WMC implementation. 
        
        Arguments: 
            formula (SddNode): 
                The formula to perform WMC on. Assumed to already be in the sdd format. 
            semiring (ProductSemiring): 
                The semiring instance to use for WMC computation. Each semiring must implement
                the `assignment`, `sum`, and `product` methods and defines a new computation. 
            evidence (set[tuple[int, str]], optional): 
                A set of evidence literals to condition on during WMC. Defaults to None. When provided,
                contains names of variables along with their polarity (1 for positive, 0 for negative).
            aux_vars (set[str], optional): 
                A set of auxiliary variables to ignore during WMC. Defaults to None.
            conditioning (set[tuple[int, str]], optional): 
                A set of conditioning literals to condition on during WMC. Defaults to None. Like
                with evidence, when provided, contains names of variables along with their polarity
                (1 for positive, 0 for negative).
            incremental_simplify (bool, optional):
                Whether to perform incremental simplification during WMC computation. Defaults to False
                and only applied to semirings that build symbolic formulas.
            **kwargs:
                Additional keyword arguments for WMC computation. These include `trace`, an
                optional dict filled with the semiring value of each node (`trace["nodes"]`, by
                SDD node id), the (smoothed) value of each element of each decision node
                (`trace["elements"]`, by node id, in the order of `node.elements()`), and the
                value at the root (`trace["root"]`), e.g., for visualizing the computation.

        Returns: 
            The semiring value of the WMC of `formula`.
        
        """
        smoothing = semiring.requires_smoothing or kwargs.get("smoothing", False)
        ## internal identifiers in sdd to original variable names, needed for assignment 
        ## and internal representation of `evidence literals`. 
        rev_map: dict[int, str] = {
            iden.literal : str(var) for var, iden  in self._sym_map.items()
        }
        eliterals = set([
            (1 if "~" not in str(var) else 0, str(var).replace("~", ""))
            for var in (evidence or set())
        ])
        cliterals = set([
            (1 if "~" not in str(var) else 0, str(var).replace("~", ""))
            for var in (conditioning or set())
        ])
        queries = set([
            (1 if "~" not in str(var) else 0, str(var).replace("~", ""))
            for var in (query or set())
        ])
        if queries:
            smoothing = True
        trace = kwargs.get("trace")
        aux_vars = aux_vars or set()
        categoricals = categoricals or set()

        ## smooth with respect to all variables (not only those under the formula's vtree),
        ## so that variables simplified out of the formula are still accounted for 
        smooth_to_root = kwargs.get("smooth_to_root", smoothing)
        sdd_iterator = SddIterator(
            self.manager, smooth_to_root=smooth_to_root,
        )

        asdict = {
            "evidence"     : eliterals,
            "conditioning" : cliterals,
            "aux_vars"     : aux_vars,
            "categoricals" : categoricals, 
            "query"        : queries, 
        }

        def smooth(val: Any, missing_literals: set[int]) -> Any:
            """Multiplies `val` by the summed weight of each missing (non-auxiliary) variable."""
            for l in missing_literals:
                wpos = rev_map[abs(l)]
                if wpos in aux_vars or wpos in self._hidden_vars:
                    continue
                missing_combined_weight = semiring.sum(
                    semiring.assignment(wpos, 0, **asdict), 
                    semiring.assignment(wpos, 1, **asdict)
                )
                val = semiring.product(val, missing_combined_weight)
            return val

        def wmc_func(
            node: SddNode, 
            rvalues: list, 
            expected_prime_vars: set[int], 
            expected_sub_vars: set[int]
        ) -> Any: 
            """Implementation of the wmc function for the SDD iterator using `semiring."""
            if rvalues is None:
                if node.is_true():
                    val = semiring.true 

                    ## apply smoothing if needed to true node 
                    if smoothing: 
                        missing_literals = (
                            expected_prime_vars
                            if expected_prime_vars is not None
                            else set()
                        )
                        missing_literals |= (
                            expected_sub_vars
                            if expected_sub_vars is not None
                            else set()
                        )
                        val = smooth(val, missing_literals)

                    return val
                
                elif node.is_false():
                    return semiring.false
                
                elif node.is_literal():
                    wpos = rev_map[abs(node.literal)]
                    pol = 1 if node.literal > 0 else 0 
                    val = semiring.assignment(wpos, pol, **asdict)

                    if smoothing: 
                        lit_scope = {abs(node.literal)}
                        if expected_prime_vars is not None:
                            missing_literals = expected_prime_vars.difference(lit_scope)
                        else:
                            missing_literals = set()
                        if expected_sub_vars is not None:
                            missing_literals |= expected_sub_vars.difference(lit_scope)
                        val = smooth(val, missing_literals)

                    return val 
            else: 
                if node is not None and not node.is_decision():
                    raise ValueError(
                        f"Unknown decision node encountered during WMC: {node}"
                    ) from None 
        
                out = None
                for (prime_weight, sub_weight, prime_vars, sub_vars) in rvalues:
                    branch_weight = semiring.product(prime_weight, sub_weight)

                    if smoothing:
                        missing_literals = expected_prime_vars.difference(
                            prime_vars
                        ) | expected_sub_vars.difference(sub_vars)
                        branch_weight = smooth(branch_weight, missing_literals)
                    if trace is not None and node is not None:
                        trace["elements"].setdefault(node.id, []).append(branch_weight)
                    if out is not None: 
                        out = semiring.sum(out, branch_weight)
                    else:
                        out = branch_weight

                    if incremental_simplify:
                        scope_size = len((expected_prime_vars or set()) | (expected_sub_vars or set()))
                        if scope_size <= simplify_threshold:
                            out = semiring.simplify(out)

                    #out = semiring.sum(out, branch_weight)
                return out 

        if trace is None:
            return sdd_iterator.depth_first(formula, wmc_func)

        def traced_wmc_func(node, rvalues, expected_prime_vars, expected_sub_vars):
            val = wmc_func(node, rvalues, expected_prime_vars, expected_sub_vars)
            if node is not None:
                trace["nodes"][node.id] = val
            return val

        trace.update(nodes={}, elements={})
        output = sdd_iterator.depth_first(formula, traced_wmc_func)
        trace["root"] = output
        return output


    def load_dimacs(self, dimacs: str) -> SddNode:
        """Creates a SymbolicMapper instance from a DIMACS file.
        
        Arguments:
            dimacs: The DIMACS string containing the formula.

        """
        with redirect_stdout(io.StringIO()):
            mgr, root = SddManager.from_cnf_string(dimacs)
        
        self.manager = mgr
        self._sym_map = {}
        extra_vars = set()
        extra_ids = set()
        ## collect variable names from dimacs:
        for l in dimacs.splitlines():
            if l.startswith('c ') and len((p:=l.split()))==3 and p[1].isdigit():
                var_id = int(p[1])
                var_name = p[2].strip()
                self._sym_map[var_name] = mgr[var_id]
                if "k!" in var_name:
                    extra_vars.add(var_name)
                    extra_ids.add(var_id)

        for vid in extra_ids:
            root = self.manager.exists(vid, root)
        self._quantified_vars = extra_vars

        ## variables that do not occur in the CNF (e.g., removed with tautological
        ## clauses) are still (free) variables of the formula
        for var_name in sorted(str(v) for v in self.var_list):
            if var_name not in self._sym_map:
                self.manager.add_var_after_last()
                self._sym_map[var_name] = self.manager[self.manager.var_count()]

        return (root, extra_vars)
