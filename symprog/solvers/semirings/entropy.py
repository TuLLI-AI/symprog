from ..semiring import SemiringRegistry, ProductSemiring, TupleType
from ..solver_utils import SolverOut
from .prob import SumProductSemiring
from typing import Optional

__all__ = ["Entropy"]

@SemiringRegistry(name="Entropy")
class Entropy(ProductSemiring[SumProductSemiring, SumProductSemiring]):
    """Implements the entropy semiring for computing entropy alongside 
    probabilities.

    The entropy semiring is defined as the tuple semiring over the 
    probability semiring and itself, i.e., (R+, +, x, 0, 1) x (R+, +, x, 0, 1).

    Methods:
        sum_op: tuple-wise addition.
        product_op: tuple-wise multiplication.
        zero_op: returns (0.0, 0.0).
        one_op: returns (1.0, 0.0).
        negation_op: returns (1 - x, y) 
            (negation only applied during weight assignment).
    """
    Leftsemiring = SumProductSemiring
    Rightsemiring = SumProductSemiring

    def product(self, x: TupleType, y: TupleType) -> TupleType:
        """Implementation of the product operator using tuple-wise multiplication.

        The product is defined as:  
            (p1, h1) * (p2, h2) = (p1 * p2, h1 * p2 + h2 * p1)
        
        where p1, p2 are probabilities and h1, h2 are entropies.
        """
        prob_product = self.left.product(x[0], y[0])
        entropy_product = self.right.sum(
            self.right.product(x[1], y[0]), self.right.product(y[1], x[0])
        )
        return (prob_product, entropy_product)
    
    def one(self) -> TupleType:
        """Returns the identity element for product equal to (1.0, 0.0).
        
        NOTE: overrides the right semiring's one to be 0.0 for entropy.
        """
        return (self.left.one(), self.right.zero())
    
    def assignment(
        self,
        identifier: int | str, 
        polarity: int, 
        evidence: Optional[set] = None,
        conditioning: Optional[set] = None, 
        aux_vars: Optional[set] = None, 
        **kwargs,
    ): 
        """Returns the assignment from the left semiring (probability). Overrides
        the default assignment to compute the entropy component as p * log(p),
        where p is the (evidence-adjusted) probability weight of the literal and
        `0 * log(0) = 0`.

        Raises:
            NotImplementedError: If query literals are provided (entropy is computed
                over all models consistent with the formula and evidence).
        """
        if kwargs.get("query"):
            raise NotImplementedError(
                "Queries are not supported for Entropy; use `evidence` or a conditional formula instead."
            )
        evidence = evidence if evidence is not None else set()
        aux_vars = aux_vars if aux_vars is not None else set()

        left_weight = self.left.assignment(
            identifier,
            polarity,
            evidence,
            aux_vars=aux_vars,
            categoricals=kwargs.get("categoricals"),
        )
        entropy = self._mgr.xlogx(left_weight)

        return (left_weight, entropy)
    
    @property 
    def requires_smoothing(self) -> bool:
        """Indicates whether smoothing is required for this semiring. 
        Set to True for entropy.
        """
        return True
    
    @property 
    def log_prob(self) -> bool: 
        """Indicates whether this semiring uses log probabilities. Returns 
        false for this semiring."""
        return False
    
    def process_output(self, output, **kwargs) -> SolverOut:
        """Processes the output from the solver to generate a human-readable string.

        Arguments:
            output: The SolverOut object containing the result from the solver.
        Returns:
            A SolverOut object with a string representation of the result.
        """ 
        Z, S = output 
        cond_entropy = self._mgr.log(Z) - (S / Z)

        return SolverOut(
            inference=cond_entropy,
            variables=None, 
            inference_type='Entropy',
            details={
                'partition_function': self._mgr.detach(Z) \
                    if Z is not None else None, 
                'unnormalized_entropy': self._mgr.detach(S) \
                    if S is not None else None
            }
        )