from .common import SumProductSemiring, LogProbSemiring
from ..semiring import SemiringRegistry, ProductSemiring
from ..solver_utils import SolverOut

__all__ = ["SumProductSemiring", "LogProbSemiring"]

@SemiringRegistry(name="Prob")
class Prob(ProductSemiring[SumProductSemiring, SumProductSemiring]):
    """Implements the probability semiring (R, +, x, 0, 1) as a product semiring, which gives
    extra support for performing on-the-fly conditioning. 
    
    Methods:
        sum_op: ordinary addition.
        product_op: ordinary multiplication.
        zero_op: returns 0.0.
        one_op: returns 1.0.
        negation_op: returns 1 - x 
            (negation only applied during weight assignment).
    """
    Leftsemiring = SumProductSemiring

    def div(self, x, y):    
        """Division operation over two values x and y. Needed for conditioning."""
        return x / y # type: ignore 
    
    def process_output(self, output, **kwargs) -> SolverOut:
        """Processes the output from the solver to generate a human-readable string.

        Arguments:
            output: The SolverOut object containing the result from the solver.
        Returns:
            A SolverOut object with a string representation of the result.
        """ 
        conditional = kwargs.get("conditional", False)
        unprob, normalization = output 
        marginal = kwargs.get("marginal", False)

        if conditional: 
            prob = unprob / normalization #(unprob + normalization)
        else: 
            prob = unprob 

        return SolverOut(
            inference=prob,
            variables=None, 
            inference_type='Prob',
            details={
                'conditional': conditional, 
                'unnormalized_prob': self._mgr.detach(unprob) \
                    if unprob is not None else None, 
                'normalization': self._mgr.detach(normalization) \
                    if normalization is not None else None, 
                "log" : False, 
                "marginal": marginal
            }
        )
    
@SemiringRegistry(name="LogProb")
class LogProb(ProductSemiring[LogProbSemiring, LogProbSemiring]):
    """Implements the log probability semiring: (R, logaddexp, +, -inf, 1) as a 
    product semiring.

    Note: assumes negation only over atoms, so inherits negation from probability 
    and doesn't require special log negation. 

    Methods:
        sum_op: logaddexp addition.
        product_op: ordinary addition.
        zero_op: returns -inf.
        one_op: returns 0.0.
        negation_op: returns 1 - x 
            (negation only applied during weight assignment).
    """
    Leftsemiring = LogProbSemiring

    def div(self, x, y):    
        """Division operation over two values x and y in log space. 
        Needed for conditioning."""
        return x - y # type: ignore 
    
    def process_output(self, output, **kwargs) -> SolverOut:
        """Processes the output from the solver to generate a human-readable string.

        Arguments:
            output: The SolverOut object containing the result from the solver.
        Returns:
            A SolverOut object with a string representation of the result.
        """ 
        conditional = kwargs.get("conditional", False)
        marginal = kwargs.get("marginal", False)
        
        unprob, normalization = output 
        if conditional: 
            ## TODO: check that normalization is not nan or 0
            #prob = self._mgr.log_sigmoid(unprob - normalization)
            prob = unprob - normalization 
        else: 
            prob = unprob 

        return SolverOut(
            inference=prob,
            variables=None, 
            inference_type='Prob',
            details={
                'conditional': conditional,
                'unnormalized_prob': self._mgr.detach(unprob) \
                    if unprob is not None else None, 
                'normalization': self._mgr.detach(normalization) \
                    if normalization is not None else None, 
                "log" : False, 
                "marginal": marginal
            }
        )
    