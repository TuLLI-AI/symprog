from abc import abstractmethod
from typing import Generic, Any, TypeVar
from dataclasses import dataclass

from .solver_utils import SATOutput, ModelOutput, Formula, SolverOut
from ..base import UtilBase
from ..registry import Registry

T = TypeVar("T") 

class SymbolicMapper(UtilBase, Generic[T]):
    """Base class for symbolic mappings (i.e., clases that build symbolic 
    objects such as formulas, circuits, decision diagrams ...)

    """
    def __init__(self, var_list: list[str | int] | None = None, **kwargs) -> None:
        """Initializes the mapper instance. 

        Arguments: 
            var_list: Optional list of variables to initialize the mapper with.
        """
        self.var_list: list[int | str] = [] if var_list is None else var_list
        self._sym_map: dict[str, T] = {}
        self._init_symbols()

    def _init_symbols(self) -> None:
        """Adds the provided weights to the underlying symbol map"""
        if self._sym_map is None or not self.var_list:
            raise ValueError("Symbol map not initialized correctly")

        for var in self.var_list:
            self.add_symbol(var)
            #self._sym_map[var] = self.add_symbol(var) 

    def __getitem__(self, s) -> T:
        ### returns underlying symbolic encoding for a symbol
        s = str(s)
        if s not in self._sym_map:
            raise ValueError(f"Unknown symbol: {s}")
        return self._sym_map[s]

    def add_symbol(self, symbol: str | int) -> None:
        """Adds a symbol to the mapper.

        Arguments: 
            symbol: The symbol to be added. 
        """
        raise NotImplementedError("Adding symbols not allowed for this class") 
        
    def Variables(self, var_list: str | list[str | int]) -> list[T]:
        """Returns the internal variables provided in the var_list (e.g., to
        use for building formulas and other more complex objects). 

        Arguments: 
            var_list: The list of variables to retrieve. 
        """
        if isinstance(var_list, str):
            var_list = [v.strip() for v in var_list.split(" ")]
        return [self._sym_map[str(v)] for v in var_list]

    @abstractmethod 
    def Or(self, obj1: T, obj2: T) -> T:
        """Implementation of logical `Or`.

        Arguments: 
            obj1: The first argument for disjunction. 
            obj2: The second argument for disjunction
        """
        raise NotImplementedError('Disjunction not implemented')
    
    @abstractmethod
    def And(self, obj1: T, obj2: T) -> T:
        """Implementation of logical `And`.

        Arguments: 
            obj1: The first argument for conjunction. 
            obj2: The second argument for conjunction. 
        """
        raise NotImplementedError("Conjunction not implemented")

    @abstractmethod
    def Implies(self, obj1: T, obj2: T) -> T:
        """Implementation of logical material implication. 

        Arguments: 
            obj1: The first argument for implication. 
            obj2: The second argument for implication. 
        """
        raise NotImplementedError("Implication not implemented")

    @abstractmethod
    def Biconditional(self, obj1: T, obj2: T) -> T:
        """Implementation of the biconditional.

        Arguments: 
            obj1: The first argument for implication. 
            obj2: The second argument for implication. 
        """
        return self.And(self.Implies(obj1, obj2),self.Implies(obj2, obj1))

    @abstractmethod
    def Not(self, expr: T) -> T:
        """Implementation of logical negation.

        Arguments: 
            expr: The expression to negate. 
        """
        raise NotImplementedError("Negation not implemented") 

    @abstractmethod
    def Symbol(self, name: str | int, return_val: bool = False):
        """Representation and creation on primitive symbols.

        Arguments: 
            name: The name of the symbol. 
        """
        raise NotImplementedError("Symbol creation not implemented")
    
    @property 
    def symbol_map(self) -> dict[str, T]:
        """Returns the internal symbol map for the mapper."""
        return self._sym_map

    @property
    @abstractmethod
    def false(self) -> T:
        """Internal representation of the false primitive."""
        raise NotImplementedError

    @property
    @abstractmethod
    def true(self) -> T:
        """Internal representation of the true primitive."""
        raise NotImplementedError

    def _post_process(self, formula: T) -> T:
        return formula

    ## reasoning interfaces

    def SAT(self, formula: T, **kwargs) -> SolverOut:
        """Checks the satisfiability of the given formula. 

        Arguments: 
            formula: The input formula. 
        """
        result, details = self._sat_interface(formula, **kwargs)
        return SolverOut(inference=result, inference_type="SAT", details=details)
    
    def Count(self, formula: T, **kwargs) -> SolverOut:
        """Returns the model count for the input formula. 

        Arguments: 
            formula: The input formula. 
        """
        result, details = self._count_interface(formula, **kwargs)
        return SolverOut(inference=result, inference_type="Count", details=details)

    def Models(self, formula: T, **kwargs) -> ModelOutput:
        """Returns the propositional models of a formula.  

        Arguments: 
            formula: The input formula. 
        """
        return self._model_enumeration(formula, **kwargs)
    
    def BuildFormula(self, formula: T, **kwargs) -> Formula:
        """Builds a given formula built using this solver into a `Formula` object. 

        Arguments: 
            formula: The input formula. 
        """
        return Formula(formula=formula, dot=self._to_dot(formula))

    @abstractmethod
    def _model_enumeration(self, formula: T, **kwargs) -> ModelOutput:
        """Interface for enumerating propositional models of the formula:

        Arguments: 
            formula: The formula to check for satisfiability. 
        """
        raise NotImplementedError("Model enumeration not implemented for this class")
    
    @abstractmethod 
    def _sat_interface(self, formula: T, **kwargs) -> tuple[SATOutput, str | None]:
        """Interface for running SAT solving. 

        Arguments: 
            formula: The formula to check for satisfiability. 
        """
        raise NotImplementedError("SAT interface not implemented for this class")
    
    @abstractmethod 
    def _count_interface(self, formula: T, **kwargs) -> tuple[int, str | None]:
        """Interface for running SAT solving. 

        Arguments: 
            formula: The formula to check for satisfiability. 
        """
        raise NotImplementedError("SAT interface not implemented for this class")

    ## visualization
    def _to_dot(self, formula: T) -> str | None:
        """Method for (optionally) mapping formula to dot visualization.

        Arguments: 
            formula: The formula to map to a dot format. 
        """
        return None

    def _to_rep(self, formula: T) -> str | None:
        """Method for (optionally) mapping formula to dot visualization.

        Arguments: 
            formula: The formula to map to a dot format. 
        """
        return None 
    
    @property 
    def has_wmc(self) -> bool:
        """ Returns whether the mapper has a WMC interface. False by default""" 
        return False 
    
    def wmc_interface(self, formula: T, semiring, **kwargs) -> Any:
        """Interface for performing weighted model counting using a given semiring object.

        Arguments: 
            formula: The internal symbolic object to perform WMC on. 
            semiring: The semiring to use for WMC. 
        """
        raise NotImplementedError("WMC interface not implemented for this class")
    
    ## methods for loading and backing up mappers 
    @classmethod 
    def from_file(cls, filepath: str, **kwargs):
        """Creates a SymbolicMapper instance from a file.

        Arguments:
            filepath: The path to the file containing the formula or model.
        """
        raise NotImplementedError("from_file method not implemented for this class")

    @classmethod 
    def from_dimacs(cls, filepath: str, **kwargs):
        """Creates a SymbolicMapper instance from a DIMACS file.

        Arguments:
            filepath: The path to the DIMACS file containing the formula.
        """
        raise NotImplementedError("from_dimacs method not implemented for this class")
    
    def simplify(self, formula: T, **kwargs) -> T:
        """Simplifies the given formula using the underlying solver's simplification capabilities.
        Returns the formula unchanged by default.

        Arguments:
            formula: The formula to simplify.
        """
        return formula

class SATMapper(SymbolicMapper[T]):
    """Base class for symbolic mappings that support SAT solving. 
    Inherits from SymbolicMapper and adds SAT-specific methods.
    """
    
    @staticmethod 
    def write_cnf(formulas, **kwargs): 
        """Writes the CNF representation of the formula to a file. 
        Passes by default. 
        """
        raise NotImplementedError("write_cnf method not implemented for this class")


class MapperRegistry(Registry[SymbolicMapper]):
    """Registry class for storing solver interface classes.
    
    Example usage of how to use in using factory `BuildMapper` method: 

    >>> from symprog.solvers import BuildMapper 
    >>> s = BuildMapper(solver='z3', var_list=['a', 'b', 'c'])
    >>> a, b, c = s.Variables("a b c") 
    >>> formula = s.And(s.And(a, b), s.Not(c))
    """