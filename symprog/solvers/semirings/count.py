from ..semiring import Semiring, SemiringRegistry, ProductSemiring

__all__ = ["Count"]

class CountSemiring(Semiring):
    """Implements the counting semiring (N, +, x, 0, 1)
    
    Methods:
        sum_op: ordinary addition.
        product_op: ordinary multiplication.
        zero_op: returns 0.
        one_op: returns 1.
        negation_op: returns 1.0 (no effect)
    """
    
    def one(self):
        """Returns the identity element for product equal to 1."""
        return self._mgr.one()
    def zero(self):
        """Returns the identity element for summation equal to 0."""
        return self._mgr.zero() 
    def sum(self,x, y):
        """Implementation of the summation operator using ordinary addition."""
        return self._mgr.add(x, y)
    def product(self,x, y):
        """Implementation of the product operator using ordinary multiplication."""
        return self._mgr.mult(x, y)
    def negation(self,x):
        """Implementation of negation operator as identity (no change)."""
        return self._mgr.one()

    @property 
    def no_weights(self) -> bool: 
        """Indicates whether this semiring uses no weights (all weights are 1). Set to 
        true for this semiring."""
        return True 
    
    @property 
    def requires_smoothing(self) -> bool: 
        """Indicates whether smoothing is required for this semiring. Set to true 
        for this semiring to avoid zero counts."""
        return True
    
@SemiringRegistry(name="Count")
class Count(ProductSemiring[CountSemiring, CountSemiring]):
    """Implements the counting semiring (N, +, x, 0, 1) as a product semiring, which gives
    extra support for performing on-the-fly conditioning. 

    NOTE: When using pysdd as the weight manager, it is much faster to use
    the built-in counting functionality of pysdd rather than this semiring.
    This implementation is primarily for educational purposes.

    Methods:
        sum_op: ordinary addition.
        product_op: ordinary multiplication.
        zero_op: returns 0.
        one_op: returns 1.
        negation_op: returns 1.0 (no effect)
    """
    Leftsemiring = CountSemiring

    def div(self, x, y):
        """Implementation of the division operator using ordinary division."""
        return x / y

    @property 
    def no_weights(self) -> bool: 
        """Indicates whether this semiring uses no weights (all weights are 1). Set to 
        true for this semiring."""
        return True 
    
    @property 
    def requires_smoothing(self) -> bool: 
        """Indicates whether smoothing is required for this semiring. Set to true 
        for this semiring to avoid zero counts."""
        return True
