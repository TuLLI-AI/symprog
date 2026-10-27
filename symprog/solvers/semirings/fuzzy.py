import warnings

from .common import MaxMinSemiring, FuzzyProduct
from ..semiring import SemiringRegistry, ProductSemiring
from ...base import ExperimentalWarning

__all__ = ["Fuzzy", "Product"]


@SemiringRegistry(name="Fuzzy")
class Fuzzy(ProductSemiring[MaxMinSemiring, MaxMinSemiring]):
    """Implements the fuzzy semiring: ([0,1], max, min, 0, 1) as a product semiring,
    which gives extra support for performing on-the-fly conditioning. 

    Properties:
        no_weights: Boolean indicating whether this semiring uses no weights
            (all weights are 1).
        log_prob: Boolean indicating whether this semiring uses log probabilities.
            (default set to False). 
    Methods:
        sum_op: maximum operator.
        product_op: minimum operator.
        zero_op: returns 0.
        one_op: returns 1.
        negation_op: returns 1 - x.
    """
    Leftsemiring = MaxMinSemiring

    def div(self, x, y):
        raise NotImplementedError("Division is not supported for the Fuzzy semiring.")
    
@SemiringRegistry(name="Product")
class Product(ProductSemiring[FuzzyProduct, FuzzyProduct]):
    """EXPERIMENTAL: Implements product fuzzy logic: ([0,1], x + y - xy, x * y, 0, 1), i.e.,
    the product t-norm with the probabilistic sum t-conorm, as a product semiring.

    WARNING: the probabilistic sum does not distribute over multiplication, so this
    structure is not a true semiring. Results therefore depend on the structure of the
    compiled circuit (e.g., the SDD and its variable ordering) and are not equal to the
    probabilistic sum over models in general.

    Properties:
        no_weights: Boolean indicating whether this semiring uses no weights
            (all weights are 1).
        log_prob: Boolean indicating whether this semiring uses log probabilities.
            (default set to False).
    Methods:
        sum_op: probabilistic sum (x + y - xy).
        product_op: ordinary multiplication.
        zero_op: returns 0.
        one_op: returns 1.
        negation_op: returns 1 - x.
    """
    Leftsemiring = FuzzyProduct

    def __init__(self, weight_mgr, **kwargs) -> None:
        warnings.warn(
            "The `Product` mode is experimental: it is not a true semiring, so results "
            "depend on the compiled circuit structure.",
            ExperimentalWarning,
            stacklevel=2,
        )
        super().__init__(weight_mgr, **kwargs)