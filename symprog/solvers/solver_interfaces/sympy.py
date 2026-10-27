import sympy
from sympy.logic.inference import satisfiable
from sympy.logic.boolalg import Boolean as SympyBoolean
from typing import Optional

from ..solver_utils import ModelOutput, SATOutput
from ..mapper import SymbolicMapper, MapperRegistry

__all__ = ["SympyMapper"]

@MapperRegistry(name = "sympy")
class SympyMapper(SymbolicMapper[SympyBoolean]):
    """Symbolic mapper object that maps into sympy. 

    Attributes: 
        num_variables: The number of variables in the program and logic encoding.
        manager: Optional manager attribute (for Sdd mappers)
        false: Representation of logical `false` for the mapper. 
        true: Representation of logical `true` for this mapping.
    
    """
    def add_symbol(self, symbol: str | int) -> None:
        """Adds a symbol to the mapper.

        Arguments: 
            symbol: The symbol to be added. 
        """
        if symbol in self._sym_map:
            raise ValueError(f"Symbol already in map: {symbol}")  
        self._sym_map[str(symbol)] = sympy.Symbol(symbol)          
        #return sympy.Symbol(symbol)
    
    def Or(self, obj1: SympyBoolean, obj2: SympyBoolean) -> SympyBoolean:
        """Implementation of logical `Or`.

        Arguments: 
            obj1: The first argument for disjunction. 
            obj2: The second argument for disjunction
        """
        return sympy.Or(obj1, obj2)
    
    def And(self, obj1: SympyBoolean, obj2: SympyBoolean) -> SympyBoolean:
        """Implementation of logical `And`.

        Arguments: 
            obj1: The first argument for conjunction. 
            obj2: The second argument for conjunction. 
        """
        return sympy.And(obj1, obj2)
    
    def Implies(self, obj1: SympyBoolean, obj2: SympyBoolean) -> SympyBoolean:
        """Implementation of logical material implication. 

        Arguments: 
            obj1: The first argument for implication. 
            obj2: The second argument for implication. 
        """
        return sympy.Implies(obj1, obj2)
    
    def Biconditional(self, obj1: SympyBoolean, obj2: SympyBoolean) -> SympyBoolean:
        """Implementation of logical material implication. 

        Arguments: 
            obj1: The first argument for implication. 
            obj2: The second argument for implication. 
        """
        return self.And(self.Implies(obj1, obj2),self.Implies(obj2, obj1))
    
    def Symbol(
        self, name: str | int, return_val: Optional[bool] = False
    ) -> str | SympyBoolean:
        """Representation and creation on primitive symbols. Returns either 
        internal string symbol or the underlying object (if `return_val=True`).  

        Arguments: 
            name: The name of the symbol. 
            return_val: Switch to determine whether to return symbol object. 
        """
        name = str(name)
        self._sym_map[name] = sympy.Symbol(name)
        if return_val:
            return self._sym_map[name]
        return name
    
    def Not(self, expr: SympyBoolean) -> SympyBoolean:
        """Implementation of logical negation.

        Arguments: 
            expr: The expression to negate. 
        """
        return sympy.Not(expr)
    
    @property
    def false(self) -> SympyBoolean:
        """Internal representation of the false primitive."""
        return sympy.false
    
    @property
    def true(self) -> SympyBoolean:
        """Internal representation of the true primitive."""
        return sympy.true

    ## interfaces 

    def _model_enumeration(self, formula: SympyBoolean, **kwargs) -> ModelOutput:
        """Interface for model enumeration.

        Arguments: 
            formula: The formula to enumerate models for. 
        """
        for model in satisfiable(formula, all_models=True): # type: ignore
            if model is False:
                ## unsatisfiable: sympy yields False instead of no models
                return
            yield {str(k) : v for k, v in model.items()}

    def _count_interface(self, formula: SympyBoolean, **kwargs) -> tuple[int, None]:
        """Interface for model counting. Works by exhaustively enumerating models.

        WARNING: This method is not optimized and should only be used for small formulas.

        Arguments: 
            formula: The formula to count models for. 
        """
        count = 0
        for _ in satisfiable(formula, all_models=True): # type: ignore 
            count += 1
        return (count, None)

    def _sat_interface(self, formula: SympyBoolean, **kwargs) -> tuple[str, None]:
        """Interface for running SAT solving.

        Arguments: 
            formula: The formula to check for satisfiability. 
        """
        solver_out = satisfiable(formula) # type: ignore
        sat = "yes" if solver_out else "no"
        return (sat, None)
    
