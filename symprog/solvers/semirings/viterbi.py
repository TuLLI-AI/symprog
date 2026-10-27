from .common import MaxProductSemiring, TropicalSemiring
from ..semiring import SemiringRegistry, ProductSemiring, SemiringBuilder, TupleType
from ..solver_utils import SolverOut

__all__ = ["ViterbiAssignment", "Viterbi", "LogViterbi"]

class ViterbiPrinting:
    """Provides string representation for Viterbi semiring results."""
    def process_output(self, output, **kwargs) -> SolverOut:
        """Processes the output from the solver to generate a human-readable string.

        Arguments:
            output: The SolverOut object containing the result from the solver.
        Returns:
            A SolverOut object with a string representation of the result.
        """ 
        prob, assignments = output 
        if not isinstance(assignments, list): 
            assignments = None 
        else: 
            assignments = {var: bool(p) for p, var in assignments}

        return SolverOut(
            inference=prob,
            variables=assignments, 
            inference_type='Viterbi',
        )
    
    def div(self, x, y):
        """Division is not defined for the Viterbi semiring."""
        raise NotImplementedError("Division is not defined for the Viterbi semiring.")
    

@SemiringRegistry(name="Viterbi")
class Viterbi(ViterbiPrinting, ProductSemiring[MaxProductSemiring, MaxProductSemiring]):
    """Implements the Viterbi semiring (R+, max, x, 0, 1) as a product semiring. Uses an additional 
    MaxProductSemiring to track the maximum score alongside the assignments. 

    Methods:
        sum_op: maximum operation.
        product_op: ordinary multiplication.
        zero_op: returns 0.
        one_op: returns 1.
        negation_op: returns 1.0 (no effect)
    """
    Leftsemiring = MaxProductSemiring
    Rightsemiring = MaxProductSemiring


@SemiringRegistry(name='LogViterbi')
class LogViterbi(ViterbiPrinting, ProductSemiring[TropicalSemiring, TropicalSemiring]):
    """Implements the Log Viterbi semiring (R+, max, +, -inf, 0) as a product semiring. Uses an additional 
    TropicalSemiring to track the minimum score alongside the assignments. 

    Methods:
        sum: maximum operation.
        product: ordinary addition.
        zero_op: returns -inf.
        one_op: returns 0.
    """
    Leftsemiring = TropicalSemiring
    Rightsemiring = TropicalSemiring


class SetBuilder(SemiringBuilder):
    """Builder class for Viterbi semiring. Helps to track variables assignments during 
    the computation of the Viterbi score.
    
    NOTE: we only track the assignments here, the score is handled by the other semiring. Also, 
    while we used lists, which are ordered, the order does not matter for Viterbi assignments and 
    repeats do not occur since each variable is assigned only once.

    Methods:
        assignment: assigns a value to a variable.
        product: concatenates lists of assignments.
        sum: selects the list with greater length (not used directly in Viterbi).
        zero: returns the additive identity (empty list).
        one: returns the multiplicative identity (empty list).
    """
    
    def assignment(
        self, 
        identifier, 
        polarity,
        evidence, 
        aux_vars, 
        **kwargs
    ) -> list:
        """Assigns a value to a variable in the Viterbi semiring context.

        Arguments:
            identifier: The variable name (string).
            polarity: Boolean indicating whether the literal is positive (True) or negated (False).
            evidence: Optional evidence value (not used in Viterbi).
            aux_vars: Optional auxiliary variables (not used in Viterbi).
        Returns:
            A list containing a single tuple of (polarity, identifier).
        """
        return [(polarity, identifier)]
    
    def product(self, x, y) -> list:
        """Implementation of the product operator which involves concatenating lists."""
        return x + y
    
    def sum(self, x, y) -> list:
        """Implementation of the sum operator which involves selecting the list with greater length."""
        raise NotImplementedError("Viterbi semiring does not support sum operation directly.")

    @property 
    def requires_smoothing(self) -> bool:
        """Indicates whether smoothing is required for this semiring. Set to 
        false for Viterbi."""
        return True

    def zero(self) -> list: 
        """Returns the additive identity element.""" 
        return []
    
    def one(self) -> list: 
        """Returns the multiplicative identity element.""" 
        return []
    

@SemiringRegistry(name="ViterbiAssignment")
class ViterbiAssignment(ViterbiPrinting, ProductSemiring[MaxProductSemiring, SetBuilder]):
    """Implements the Viterbi semiring (R+, max, x, 0, 1) as a product semiring. Uses an additional 
    SetBuilder semiring to build the set of assignments corresponding to the maximum score. 

    Methods:
        sum_op: maximum operation.
        product_op: ordinary multiplication.
        zero_op: returns 0.
        one_op: returns 1.
        negation_op: returns 1.0 (no effect)
    """
    Leftsemiring = MaxProductSemiring
    Rightsemiring = SetBuilder

    def div(self, x, y):
        """Division is not defined for the Viterbi semiring."""
        raise NotImplementedError("Division is not defined for the Viterbi semiring.")
    
    def product(self, x: TupleType, y: TupleType) -> TupleType:
        """Manual implementation of the product operator over two tuple values x and y.
        
        Multiplies the score components (first elements of the tuples) using the left semiring's 
        product method, and concatenates the assignment components (second elements of the tuples) 
        using the right semiring's product method.
        """
        prod_score = self.left.product(x[0], y[0])
        prod_assignments = self.right.product(x[1], y[1])
        return (prod_score, prod_assignments)
    
    def sum(self, x: TupleType, y: TupleType) -> TupleType:
        """Manual implementation of the max sum operator over two tuple values x and y.
        
        Checks the score of the first element in each tuple and returns the tuple with 
        the higher score.

        NOTE: This will break ties by choosing the first argument, so behavior is undefined 
        when using symbolic (non-concrete) weights. TODO: throw a warning about this. 

        """
        max_item = self.left.sum(x[0], y[0])
        if max_item == x[0]:
            return (x[0], x[1])
        return (y[0], y[1])