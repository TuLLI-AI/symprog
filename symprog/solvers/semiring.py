from abc import abstractmethod
from typing import Generic, Type, TypeVar

from .numerical import WeightManager
from .solver_utils import SolverOut
from ..base import UtilBase
from ..registry import Registry

S1 = TypeVar('S1', bound="SemiringBase")
S2 = TypeVar('S2', bound="SemiringBase")

I_co = TypeVar('I_co', covariant=True)
T_co = TypeVar('T_co', covariant=True)

TupleType = tuple[I_co, T_co]

M = TypeVar('M', bound=WeightManager)

class SemiringBase(UtilBase):
    """Base class for semiring implementations and defining their
    core operations.
    
    Methods: 
        zero(): Returns the additive identity element. 
        one(): Returns the multiplicative identity element. 
        sum(x, y): Performs the addition operation. 
        product(x, y): Performs the multiplication operation.
        negation(x): Performs the negation operation.
        assignment(identifier, polarity, evidence, aux_vars):
            Assigns a value to a particular literal node.
    
    """
    def __init__(self, weight_mgr: WeightManager, **kwargs) -> None:
        pass 

    @property 
    def requires_smoothing(self) -> bool:
        """Indicates whether smoothing is required for this semiring. 
        Returns true if either component semiring requires smoothing.
        """
        return True 
    
    @property 
    def no_weights(self) -> bool:
        """Indicates whether this semiring uses weights. 
        Returns True.
        """
        return False

    def simplify(self, x):
        """Simplifies the semiring value. Default is a no-op."""
        return x

    @abstractmethod
    def zero(self):
        """Returns the additive identity element."""
        raise NotImplementedError
 
    @abstractmethod
    def one(self): 
        """Returns the multiplicative identity element.""" 
        raise NotImplementedError 
 
    @abstractmethod
    def sum(self, x, y): 
        """Performs the sum operation.""" 
        raise NotImplementedError 
 
    @abstractmethod
    def product(self, x, y): 
        """Performs the product operation.""" 
        raise NotImplementedError
    
    @property 
    def true(self):
        """Alias for the multiplicative identity element."""
        return self.one()
    
    @property
    def false(self): 
        """Alias for the additive identity element."""
        return self.zero()

    
    @abstractmethod
    def assignment(
        self, 
        identifier: int | str, 
        polarity: int, 
        evidence: set[tuple[int, str]] | None = None, 
        conditioning: set[tuple[int, str]] | None = None, 
        aux_vars: set[tuple[int, str]] | None = None, 
        **kwargs
    ):
        """Assigns a value to a particular literal node. By default, returns 1/0
        if the literal (or its negation) is in evidence, otherwise returns the weight. 

        Arguments:
            identifier: The identifier associated with the literal.
            polarity: The polarity of the literal (0 for negated, 1 for positive).
            evidence: Optional set of evidence literals.
        """
        raise NotImplementedError
    
    def process_output(self, output: TupleType, **kwargs) -> SolverOut:
        """Processes the raw output from the weighted model counting
        procedure into a more user-friendly format.
        
        By default, returns the output as is.
        """
        marginal_count, normalization = output
        inference_type = type(self).__name__

        return SolverOut(
            inference=marginal_count,
            details={"right_inference": normalization},
            inference_type=inference_type
        )

class Semiring(SemiringBase): 
    """Base class for semiring implementation with weights and 
    with managers. 
 
    Methods: 
        zero(): Returns the additive identity element. 
        one(): Returns the multiplicative identity element. 
        sum(a, b): Performs the addition operation. 
        product(a, b): Performs the multiplication operation. 
        negation(x): Performs the negation operation.
        assignment(identifier, polarity, evidence, aux_vars):
            Assigns a value to a particular literal node.

    """ 
    def __init__(self, weight_mgr: WeightManager, **kwargs) -> None:
        self._mgr = weight_mgr

    @abstractmethod
    def negation(self, x):
        """Performs the negation operation over literal values `x`.
        
        Arguments:
            x: The literal value to negate.
        """
        raise NotImplementedError

    @property
    def log_prob(self) -> bool:
        """Indicates whether this semiring uses log probabilities. 
        Returns False. 
        """
        return False
    
    def assignment(
        self, 
        identifier: int | str, 
        polarity: int, 
        evidence: set[tuple[int, str]] | None = None, 
        aux_vars: set[tuple[int, str]] | None = None, 
        query: set[tuple[int, str]] | None = None,
        categoricals: set[int | str] | None = None,
        **kwargs
    ):
        """Assigns a value to a particular literal node. By default, returns 1/0
        if the literal (or its negation) is in evidence, otherwise returns the weight. 

        Arguments:
            identifier: The identifier associated with the literal.
            polarity: The polarity of the literal (0 for negated, 1 for positive).
            evidence: Optional set of evidence literals.
            query: Optional set of query literals.
            categoricals: Optional set of categorical literals.
        """
        evidence = evidence if evidence is not None else set()
        categoricals = categoricals if categoricals is not None else set()
        aux_vars = aux_vars if aux_vars is not None else set()
        query = query if query is not None else set()
        
        if self.no_weights: 
            atom_weight = self.one()
        elif identifier in categoricals and polarity == 0: 
            atom_weight = self.one()
        elif identifier in aux_vars: 
            atom_weight = self.one()
        else:
            atom_weight = self._mgr.assignment(identifier) 
            if polarity == 0: 
                atom_weight = self.negation(atom_weight)
            if self.log_prob:
                atom_weight = self._mgr.log(atom_weight)

        ##  handle evidence    
        if (polarity, str(identifier)) in evidence:
            return self.one() 
        elif (1 - polarity, str(identifier)) in evidence:  #and (identifier not in categoricals):
            return self.zero()
        
        ## queries 
        if (polarity, str(identifier)) in query:
            return atom_weight 
        elif (1 - polarity, str(identifier)) in query: 
            return self.zero()
        
        return atom_weight 
    
class SymbolicSemiring(SemiringBase):
    """Base class for symbolic semiring implementations. 
    Symbolic semirings are used to build symbolic representations of logical formulas.
    
    Methods:
        zero(): Returns the additive identity element. 
        one(): Returns the multiplicative identity element. 
        sum(x, y): Performs the addition operation. 
        product(x, y): Performs the multiplication operation.
        negation(x): Performs the negation operation.
        assignment(identifier, polarity, evidence, aux_vars):
            Assigns a value to a particular literal node.
    """
    def __init__(self, weight_mgr: WeightManager | None = None, **kwargs) -> None:
        self._mgr = weight_mgr

    def simplify(self, formula):
        """Simplifies the given symbolic formula. By default, returns the formula as is.
        
        Arguments:
            formula: The symbolic formula to simplify.
        """
        return formula
        

class SemiringBuilder(SemiringBase):
    """Base class for semiring builders, or utilties used with other semirings (themsleves semirings)
    that help to build structures computed from other semirings.

    Methods:
        zero(): Returns the additive identity element. 
        one(): Returns the multiplicative identity element. 
        sum(x, y): Performs the addition operation. 
        product(x, y): Performs the multiplication operation.
        negation(x): Performs the negation operation.
        assignment(identifier, polarity, evidence, aux_vars):
            Assigns a value to a particular literal node.
    
    """
    def __init__(self,weight_mgr: WeightManager | None = None, **kwargs): 
        self._mgr = weight_mgr

    @property 
    def requires_smoothing(self) -> bool:
        """Indicates whether smoothing is required for this semiring. 
        By default, returns False.
        """
        return False
    
    @property 
    def no_weights(self) -> bool:
        """Indicates whether this semiring uses weights. 
        Returns True.
        """
        return True
    
    @property
    def log_prob(self) -> bool:
        """Indicates whether this semiring uses log probabilities. 
        Returns False. 
        """
        return False
    

class ProductSemiring(SemiringBase, Generic[S1, S2]):
    """Base class for product semiring implementations. Designed to facilitate arbitrary 
    conditioning, evidence and auxiliary variable handling over two component semirings.

    Formally, takes the form of a structure `(L x R, ⊕, ⊗, ⊕_0, ⊗_1)` (defined over two seperate 
    semirings) with the following definitions:   
 
    - sum: `(a, b) ⊕ (c, d) = (a ⊕_l c, b ⊕_r d)`
    - product: `(a, b) ⊗ (c, d) = (a ⊗_l c, b ⊗_r d)`
    - zero: `(⊕_l, ⊕_r)`
    - one: `(⊗_l, ⊗_r)`
    
    Attributes: 
        Leftsemiring: The left semiring class (i.e., (L, ⊕_l, ⊗_l, ⊕_l, ⊗_l) ).
        Rightsemiring: The right semiring class (i.e., (R, ⊕_r, ⊗_r, ⊕_r, ⊗_r)).

    Methods:
        zero(): Returns the additive identity element. 
        one(): Returns the multiplicative identity element. 
        sum(x, y): Performs the addition operation. 
        product(x, y): Performs the multiplication operation.
        negation(x): Performs the negation operation.
        assignment(identifier, polarity, evidence, aux_vars):
            Assigns a value to a particular literal node.

    """
    Leftsemiring: Type[S1]
    Rightsemiring: Type[S2] | Type[S1] | None = None

    def __init__(self, weight_mgr: WeightManager[I_co, T_co], **kwargs) -> None:
        self._mgr: WeightManager = weight_mgr
        self.left: S1 = self.Leftsemiring(weight_mgr, **kwargs)
        if self.Rightsemiring is not None:
            self.right: S2 | S1 = self.Rightsemiring(weight_mgr, **kwargs)  
        else:
            self.right: S1 | S2 = self.left

    def one(self) -> TupleType:
        """Returns the identity element for product."""
        return (self.left.one(), self.right.one())

    def zero(self) -> TupleType:
        """Returns the identity element for summation."""
        return (self.left.zero(), self.right.zero())

    def sum(self, x: TupleType, y: TupleType) -> TupleType:
        """Sum operation over the product semiring.
        
        Arguments:
            x: The first tuple value containing left and right semiring values.
            y: The second tuple value containing left and right semiring values.
        """
        return (self.left.sum(x[0], y[0]), self.right.sum(x[1], y[1]))

    def product(self, x: TupleType, y: TupleType) -> TupleType:
        """Product operation over the product semiring.
        
        Arguments:
            x: The first tuple value containing left and right semiring values.
            y: The second tuple value containing left and right semiring values.
        """
        return (self.left.product(x[0], y[0]), self.right.product(x[1], y[1]))
    
    def assignment(
        self, 
        identifier: int | str, 
        polarity: int, 
        evidence: set[tuple[int, str]] | None = None,
        conditioning: set[tuple[int, str]] | None = None, 
        query: set[tuple[int, str]] | None = None, 
        aux_vars: set | None = None, 
        categoricals: set | None = None,
        **kwargs 
    ) -> TupleType:
        """Assignment operation over the product semiring. Takes optional `evidence`
        and `conditioning` sets for the left and right semirings respectively.

        The trick here is to past evidence to the left semiring and conditioning to the right semiring, 
        which when used correctly in WMC will allow for proper conditioning.

        Arguments:
            identifier: The identifier associated with the literal.
            polarity: The polarity of the literal (0 for negated, 1 for positive).
            evidence: Optional set of evidence literals for the left semiring.
            conditioning: Optional set of conditioning literals for the right semiring.
            aux_vars: Optional set of auxiliary unweighted Boolean variables.

        Returns:
            A tuple containing the assigned weights from the left and right semirings with 
            type determined by the given manager `self._mgr`.
                
        """
        evidence = evidence if evidence is not None else set()
        conditioning = conditioning if conditioning is not None else set()
        aux_vars = aux_vars if aux_vars is not None else set()
        categoricals = categoricals if categoricals is not None else set()
        query = query if query is not None else set()

        left_weight = self.left.assignment(
            identifier, 
            polarity, 
            evidence, 
            aux_vars=aux_vars, 
            query=query, 
            categoricals=categoricals, 
            **kwargs
        )
        right_weight = self.right.assignment(
            identifier, 
            polarity, 
            conditioning, 
            aux_vars=aux_vars, 
            categoricals=categoricals, 
            **kwargs
        )
        return (left_weight, right_weight)    
    
    def div(self, x: TupleType, y: TupleType) -> TupleType:
        """Division operation over two values x and y. Needed if conditioning
        is implemented.
        """
        raise NotImplementedError('Division operation not implemented for this semiring.')

    def simplify(self, x: TupleType) -> TupleType:
        """Simplify the semiring value by delegating to each component."""
        return (self.left.simplify(x[0]), self.right.simplify(x[1]))
    
    @property 
    def requires_smoothing(self) -> bool:
        """Indicates whether smoothing is required for this semiring. 
        Returns true if either component semiring requires smoothing.
        """
        return self.left.requires_smoothing or self.right.requires_smoothing
    
    @property 
    def no_weights(self) -> bool:
        """Indicates whether this semiring uses weights. 
        Returns true if both component semirings use no weights.
        """
        return self.left.no_weights and self.right.no_weights
    
    def process_output(self, output: TupleType, **kwargs) -> SolverOut:
        """Processes the raw output from the weighted model counting
        procedure into a more user-friendly format.
        
        By default, returns the output as is.
        """
        marginal_count, normalization = output
        inference_type = type(self).__name__

        return SolverOut(
            inference=marginal_count,
            details={"right_inference": normalization},
            inference_type=inference_type
        )
        
class SemiringRegistry(Registry[ProductSemiring | Semiring]):
    """Utility for storing different generalized `SumProduct` and semiring implementations."""
    
    def __call__(self, cls_: Type[ProductSemiring | Semiring]) -> Type[ProductSemiring]:
        """Registers a new semiring implementation. If a `Semiring` is provided, it is 
        automatically wrapped into a `ProductSemiring` with identical left and right semirings.

        Arguments:
            cls_: The semiring class to register.
        Returns:
            The registered semiring class.

        """
        if issubclass(cls_, Semiring):
            left_semiring = cls_

            wrapper_name = f"{left_semiring.__name__}ProductSemiring"

            attrs = {
                "Leftsemiring": left_semiring,
                "Rightsemiring": left_semiring,
            }

            left_po = getattr(left_semiring, "process_output", None)
            if callable(left_po):
                def process_output(self, output, **kwargs) -> SolverOut:
                    value1, value2 = output
                    out = self.left.process_output((value1, value2), **kwargs)
                    return out
                attrs["process_output"] = process_output # type: ignore

            left_div = getattr(left_semiring, "div", None)
            if callable(left_div):
                def div(self, x, y):
                    return self.left.div(x, y)
                attrs["div"] = div # type: ignore

            Wrapped = type(wrapper_name, (ProductSemiring,), attrs)
            try: 
                Wrapped.__orig_bases__ = (ProductSemiring[left_semiring, left_semiring],) ## type: ignore
            except AttributeError:
                pass    

            cls_ = Wrapped 
        
        return super().__call__(cls_) ## type: ignore