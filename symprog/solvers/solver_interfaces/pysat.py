from pysat.solvers import Solver
from pysat.formula import Formula, Atom, Implies, Or, And, Neg
from itertools import product
from typing import Optional

from ..solver_utils import ModelOutput, SATOutput
from ..mapper import SymbolicMapper, MapperRegistry

__all__ = ["PysatMapper"]

@MapperRegistry(name = "pysat")
class PysatMapper(SymbolicMapper[Formula]):
    """Symbolic mapper object that maps into pysat propositional theories."""

    def _init_symbols(self) -> None:
        """Adds the provided weights to the underlying symbol map"""        
        for var in self.var_list:
            self.add_symbol(var)
            #self._sym_map[var] = Atom(var)

    def add_symbol(self, symbol: str | int) -> None:
        """Adds a symbol to the mapper.

        Arguments: 
            symbol: The symbol to be added. 
        """
        if symbol in self._sym_map:
            raise ValueError(f"Symbol already in map: {symbol}")  
        self._sym_map[str(symbol)] = Atom(symbol)

    def Or(self, obj1: Formula, obj2: Formula) -> Formula:
        """Implementation of logical `Or`.

        Arguments: 
            obj1: The first argument for disjunction. 
            obj2: The second argument for disjunction
        """
        return Or(obj1, obj2)
    
    def And(self, obj1: Formula, obj2: Formula) -> Formula:
        """Implementation of logical `And`.

        Arguments: 
            obj1: The first argument for conjunction. 
            obj2: The second argument for conjunction. 
        """
        return And(obj1, obj2)  
    
    def Implies(self, obj1: Formula, obj2: Formula) -> Formula:
        """Implementation of logical material implication. 

        Arguments: 
            obj1: The first argument for implication. 
            obj2: The second argument for implication. 
        """
        return Implies(obj1, obj2)
    
    def Biconditional(self, obj1: Formula, obj2: Formula) -> Formula:
        """Implementation of logical material implication. 

        Arguments: 
            obj1: The first argument for implication. 
            obj2: The second argument for implication. 
        """
        return self.And(self.Implies(obj1, obj2),self.Implies(obj2, obj1))
    
    def Not(self, obj: Formula) -> Formula:
        """Implementation of logical negation.

        Arguments:
            obj: The argument for negation.
        """
        return Neg(obj)
    
    def Symbol(
        self, name: str, return_val: Optional[bool] = False
    ) -> str | Formula:
        """Representation and creation on primitive symbols.

        Arguments: 
            name: The name of the symbol. 
        """
        intn = self._sym_map.get(name, None)
        if intn is None:
            raise ValueError(f'Unknown symbol variables: {name}')
        if return_val:
            return self._sym_map[name]
        return name
    
    @property 
    def false(self) -> Formula:
        """Internal representation of the false primitive."""
        return Atom(False)
    @property 
    def true(self) -> Formula:
        """Internal representation of the true primitive."""
        return Atom(True)
    
    def _sat_interface(
        self, 
        formula: Formula, 
        solver_name: str = 'm22',
        include_stats: bool = False,
        **kwargs
    ) -> tuple[str, None | str]:
        """Interface for running SAT solving with pysat. 

        Arguments: 
            formula: The formula to check for satisfiability. 
            solver_name: The name of the pysat solver to use (defaults to `m22`, or minisat).
        """
        with Solver(name=solver_name, bootstrap_with=formula, **kwargs) as solver:
            output = solver.solve()
            sat = "yes" if output else "no"
            details = None
            if include_stats:
                details = str(solver.accum_stats())
            
        return (sat, details)
    
    def _model_enumeration(
        self, formula: Formula, solver_name: Optional[str] = 'm22', **kwargs
    ) -> ModelOutput:
        """Interface for model enumeration.

        Arguments: 
            formula: The formula to enumerate models for. 
        """
        ## pysat clausifies formulas using a global variable pool (with auxiliary Tseitin 
        ## variables), so map pool ids back to the mapper's symbols and project out the rest
        with Solver(bootstrap_with=formula, name=solver_name, **kwargs) as solver: # type: ignore 
            ## (the pool is populated when the formula is clausified, i.e., after bootstrapping)
            vpool = Formula.export_vpool()
            var_map = {
                vpool.obj2id[atom]: name for name, atom in self._sym_map.items()
                if atom in vpool.obj2id
            }
            free_vars = [name for name in self._sym_map if name not in var_map.values()]
            seen = set()

            for model in solver.enum_models(): # type: ignore 
                assignment = {var_map[abs(lit)]: (lit > 0) for lit in model if abs(lit) in var_map} # type: ignore 
                for values in product([False, True], repeat=len(free_vars)):
                    full = {**assignment, **dict(zip(free_vars, values))}
                    key = frozenset(full.items())
                    if key not in seen:
                        seen.add(key)
                        yield full

    def _count_interface(self, formula: Formula, **kwargs) -> tuple[int, None]:
        """Interface for model counting. Works by exhaustively enumerating models.

        WARNING: This method is not optimized and should only be used for small formulas.

        Arguments: 
            formula: The formula to count models for. 
        """
        count = 0
        for _ in self._model_enumeration(formula):
            count += 1
        return (count, None)
    
    def wmc_interface(self, formula: Formula, semiring, **kwargs):
        """Weighted model counting is not yet implemented for pysat."""
        raise NotImplementedError(
            "WMC over pysat formulas not yet implemented; use the 'sdd' solver for weighted inference."
        )

    # def wmc_interface(
    #     self,
    #     formula: Formula,
    #     semiring: SemiringSolver,
    #     *,
    #     solver_name: str = "m22",
    #     **kwargs,
    # ) -> any:
    #     """DPLL-style algebraic model counting using PySAT as a SAT oracle.

    #     This explores a full decision tree over the mapper's variables
    #     (`self._sym_map` / `var_list`), pruning branches that are UNSAT and
    #     combining branch values using the semiring.

    #     Semantics:
    #         - Variable set for the semiring is exactly the mapper's variables.
    #         - For each v in that set we branch on v=0 and v=1.
    #         - At each branch we multiply the child's WMC by w(v=b).
    #         - Leaves where the CNF is SAT under a *complete* assignment return
    #           semiring.true (all assignment weights have been accumulated).
    #         - UNSAT leaves return semiring.false.

    #     This implicitly performs smoothing: variables that never occur in the
    #     CNF are still branched on, so you get factors
    #         Σ_b w(v=b)
    #     for each such variable, as desired for general AMC.

    #     WARNING: This is exponential in the number of variables and intended
    #     as a pedagogical / small-instance implementation, not a sharpSAT-level
    #     solver.
    #     """
    #     # Variables that define the semiring domain
    #     var_names = list(self._sym_map.keys())
    #     n_vars = len(var_names)

    #     # Map from user variable name -> PySAT integer id
    #     var2id: dict[str, int] = {
    #         vname: atom.name for vname, atom in self._sym_map.items()
    #     }

    #     # Cache: key = tuple of assumption literals (sorted), value = semiring value
    #     cache: dict[tuple[int, ...], any] = {}

    #     # Use a single incremental solver; each call uses its own assumption set.
    #     with Solver(name=solver_name, bootstrap_with=formula, **kwargs) as solver:

    #         def amc(depth: int, current_lits: Tuple[int, ...]) -> Any:
    #             """Recursive AMC over a fixed variable order.

    #             Args:
    #                 depth: how many variables from var_names have been decided.
    #                 current_lits: tuple of assumption literals (sorted).

    #             Returns:
    #                 Semiring value for all models consistent with current_lits.
    #             """
    #             # Caching on the current partial assignment
    #             if current_lits in cache:
    #                 return cache[current_lits]

    #             # Check satisfiability under the current partial assignment
    #             if not solver.solve(assumptions=list(current_lits)):
    #                 val = semiring.false
    #             else:
    #                 if depth == n_vars:
    #                     # All mapper variables are assigned. The CNF is SAT under
    #                     # this complete assignment (over mapper vars), so the
    #                     # remaining contribution from below is just semiring.true.
    #                     val = semiring.true
    #                 else:
    #                     # Branch on the next variable in a fixed order
    #                     vname = var_names[depth]
    #                     vid = var2id[vname]

    #                     # Branch 0: vname = 0  (False)
    #                     lits0 = tuple(sorted(current_lits + (-vid,)))
    #                     v0 = amc(depth + 1, lits0)
    #                     w0 = semiring.assignment(vname, 0)
    #                     branch0 = semiring.product(w0, v0)

    #                     # Branch 1: vname = 1  (True)
    #                     lits1 = tuple(sorted(current_lits + (vid,)))
    #                     v1 = amc(depth + 1, lits1)
    #                     w1 = semiring.assignment(vname, 1)
    #                     branch1 = semiring.product(w1, v1)

    #                     # Combine branches with semiring sum (logical OR)
    #                     val = semiring.sum(branch0, branch1)

    #             cache[current_lits] = val
    #             return val

    #         result = amc(0, tuple())

    #     return result