import z3
from z3 import BoolRef, Solver
from typing import Optional

from ..solver_utils import ModelOutput, SATOutput
from ..mapper import SATMapper, MapperRegistry

__all__ = ["Z3Mapper"]

@MapperRegistry(name = "z3")
class Z3Mapper(SATMapper[BoolRef]):
    """Symbolic mapper object that maps into z3 propositional theories. 

    Attributes: 
        num_variables: The number of variables in the program and logic 
            encoding.
        manager: Optional manager attribute (for Sdd mappers)
        false: Representation of logical `false` for the mapper. 
        true: Representation of logical `true` for this mapping.
    """
    def _init_symbols(self) -> None:
        """Adds the provided weights to the underlying symbol map"""
        self._solver = Solver()
        for var in self.var_list:
            self.add_symbol(var)
            #self._sym_map[str(var)] = z3.Bool(var)

    def add_symbol(self, symbol: str | int) -> None:
        """Adds a symbol to the mapper.

        Arguments: 
            symbol: The symbol to be added. 
        """
        if symbol in self._sym_map:
            raise ValueError(f"Symbol {symbol} already exists in the mapper.")
        self._sym_map[str(symbol)] = z3.Bool(symbol)

    def Or(self, obj1: BoolRef, obj2: BoolRef) -> BoolRef:
        """Implementation of logical `Or`.

        Arguments: 
            obj1: The first argument for disjunction. 
            obj2: The second argument for disjunction
        """
        return z3.Or(obj1, obj2) # type: ignore

    def And(self, obj1: BoolRef, obj2: BoolRef) -> BoolRef:
        """Implementation of logical `And`.

        Arguments: 
            obj1: The first argument for conjunction. 
            obj2: The second argument for conjunction. 
        """
        return z3.And(obj1, obj2) # type: ignore

    def Implies(self, obj1: BoolRef, obj2: BoolRef) -> BoolRef:
        """Implementation of logical material implication. 

        Arguments: 
            obj1: The first argument for implication. 
            obj2: The second argument for implication. 
        """
        return z3.Implies(obj1, obj2)
    
    def Biconditional(self, obj1: BoolRef, obj2: BoolRef) -> BoolRef:
        """Implementation of logical material implication. 

        Arguments: 
            obj1: The first argument for implication. 
            obj2: The second argument for implication. 
        """
        return self.And(
            self.Implies(obj1, obj2),
            self.Implies(obj2, obj1)
        )

    def Not(self, expr: BoolRef) -> BoolRef:
        """Implementation of logical negation.

        Arguments: 
            expr: The expression to negate. 
        """
        return z3.Not(expr) # type: ignore 

    def Symbol(self, name: str | int, return_val: Optional[bool] = False):
        """Representation and creation on primitive symbols.

        Arguments: 
            name: The name of the symbol. 
        """
        name = str(name)
        if name not in self._sym_map:
            raise ValueError(f"Unknown z3 variable: {name}")
        return self._sym_map[name] if return_val else name
    
    @property
    def false(self) -> bool:
        """Internal representation of the false primitive."""
        return False
    
    @property
    def true(self) -> bool:
        """Internal representation of the true primitive."""
        return True
    

    def _sat_interface(
        self, formula, include_stats: Optional[bool] = False, **kwargs
    ) -> tuple[str, None | str]:
        """Interface for running SAT solving. Given that SDDs are `canonical`, we 
        can check for satisfiability be seeing if the formula is equal to `sdd.false` (unsat)
        or not. 

        Arguments: 
            formula: The formula to check for satisfiability. 
            include_stats: Include statistics in the output (default=False)
        """
        self._solver.add(formula)
        result = str(self._solver.check())
        details = None
        if include_stats: 
            details = str(self._solver.statistics())
        self._solver.reset()
        return (result, details)  
    
    def _model_enumeration(self, formula: BoolRef, **kwargs) -> ModelOutput:
        """Interface for model enumeration.

        Arguments: 
            formula: The formula to enumerate models for. 
        """ 
        s = self._solver
        s.reset()
        s.add(formula)

        vars = self._sym_map.values()
        
        while s.check() == z3.sat:
            m = s.model()

            asg = {str(v) : z3.is_true(m.eval(v, model_completion=True)) for v in vars}
            yield asg # type: ignore

            block = []
            for v in vars:
                val = m.eval(v, model_completion=True)
                block.append(v != val)
                
            s.add(z3.Or(block))

        s.reset()

    def _count_interface(self, formula: BoolRef, **kwargs) -> tuple[int, None]:
        """Interface for model counting. Works by exhaustively enumerating models.

        WARNING: This method is not optimized and should only be used for small formulas.

        Arguments: 
            formula: The formula to count models for. 
        """
        count = 0
        for _ in self._model_enumeration(formula): 
            count += 1
        return (count, None)
    
    @staticmethod 
    def write_cnf(formulas: list[BoolRef], tactic: tuple[str, ...] = ('tseitin-cnf',)) -> str: 
        """Writes the CNF representation of the formula to a dimacs file. Uses also the provided 
        tactic for the conversion and potential simplification of the target formulas.  
        
        Arguments: 
            formulas: 
                A list of Z3 formulas to convert to dimacs/CNF form. 
            tactic:
                A tuple of tactics to use for the conversion. Default is ('tseitin-cnf',). 
        """
        tactic = z3.Then(*tactic) if len(tactic) > 1 else z3.Tactic(tactic[0]) # type: ignore
        goal = z3.Goal()
        goal.add(formulas)
        return tactic(goal)[0].dimacs() # type: ignore
