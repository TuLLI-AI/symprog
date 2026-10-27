from ..semiring import Semiring, SymbolicSemiring

__all__ = [
    "MaxProductSemiring", 
    "TropicalSemiring", 
    "SumProductSemiring", 
    "LogProbSemiring",
    "MaxMinSemiring", 
    "FormulaSemiring", 
    "FuzzyProduct",
]

class MaxProductSemiring(Semiring):
    """Implements the Viterbi semiring (R+, max, x, 0, 1)
    
    Methods:
        sum_op: maximum operation.
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
        """Implementation of the summation operator using maximum."""
        return self._mgr.max(x, y)
    def product(self,x, y):
        """Implementation of the product operator using ordinary multiplication."""
        return self._mgr.mult(x, y)
    def negation(self, x):
        """Implementation of negation operator as identity (no change)."""
        return self._mgr.neg(x)
    
    @property 
    def log_prob(self) -> bool: 
        """Indicates whether this semiring uses log probabilities. Returns 
        false for this semiring."""
        return False
    
    @property 
    def requires_smoothing(self) -> bool:
        """Indicates whether smoothing is required for this semiring. Set to 
        false for Viterbi."""
        return True
    
    @property 
    def no_weights(self) -> bool: 
        """Indicates whether this semiring uses no weights (all weights are 1). Set to 
        false for this semiring."""
        return False
    
class TropicalSemiring(Semiring):
    """Implements the Tropical semiring (R+, max, +, -inf, 0)
    
    Methods:
        sum_op: maximum operation.
        product_op: ordinary addition.
        zero_op: returns -inf.
        one_op: returns 0.
    """
    
    def one(self):
        """Returns the identity element for product equal to 0."""
        return self._mgr.zero()
    def zero(self):
        """Returns the identity element for summation equal to inf."""
        return self._mgr.neg_inf() 
    def sum(self,x, y):
        """Implementation of the summation operator using minimum."""
        return self._mgr.max(x, y)
    def product(self,x, y):
        """Implementation of the product operator using ordinary addition."""
        return self._mgr.add(x, y)
    def negation(self, x):
        """Implementation of negation operator as identity (no change)."""
        return self._mgr.neg(x)
    
    @property 
    def log_prob(self) -> bool: 
        """Indicates whether this semiring uses log probabilities. Returns 
        true for this semiring."""
        return True
    
    @property 
    def requires_smoothing(self) -> bool:
        """Indicates whether smoothing is required for this semiring. Set to 
        false for Tropical."""
        return True
    
    @property 
    def no_weights(self) -> bool: 
        """Indicates whether this semiring uses no weights (all weights are 1). Set to 
        false for this semiring."""
        return False


class SumProductSemiring(Semiring):
    """Implements the probability semiring (R, +, x, 0, 1)
    
    Methods:
        sum_op: ordinary addition.
        product_op: ordinary multiplication.
        zero_op: returns 0.0.
        one_op: returns 1.0.
        negation_op: returns 1 - x 
            (negation only applied during weight assignment).
    """
    
    def one(self):
        """Returns the identity element for product equal to 1.0."""
        return self._mgr.one()
    def zero(self):
        """Returns the identity element for summation equal to 0.0."""
        return self._mgr.zero() 
    def sum(self,x, y):
        """Implementation of the summation operator using ordinary addition."""
        return self._mgr.add(x, y)
    def product(self,x, y):
        """Implementation of the product operator using ordinary multiplication."""
        return self._mgr.mult(x, y)
    def negation(self,x):
        """Implementation of negation operator using 1 - x."""
        return self._mgr.neg(x)
    def simplify(self, expr):
        """Delegates simplification to the underlying manager."""
        return self._mgr.simplify(expr)
    @property
    def requires_smoothing(self) -> bool:
        """Indicates whether smoothing is required for this semiring.
        Set to False for probability.
        """
        return False
    @property
    def log_prob(self) -> bool:
        """Indicates whether this semiring uses log probabilities. Returns
        false for this semiring."""
        return False


class LogProbSemiring(Semiring):
    """Implements the log probability semiring: (R, logaddexp, +, -inf, 1). 

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
    def one(self):
        """Returns the identity element for product, equal to 0 in log space."""
        return self._mgr.zero()
    def zero(self):
        """Returns the identity element for summation, equal to -inf in log space."""
        return self._mgr.neg_inf()
    def sum(self,x, y):
        """Implementation of the summation operator using logaddexp."""
        return self._mgr.logaddexp(x, y)
    def product(self, x, y):
        """Implementation of the product operator using ordinary addition in log space."""
        return self._mgr.add(x, y)
    def negation(self,x):
        """Implementation of negation operator using 1 - x."""
        return self._mgr.neg(x)
    
    @property 
    def log_prob(self) -> bool: 
        """Indicates whether this semiring uses log probabilities. Returns 
        true for this semiring."""
        return True
    @property
    def requires_smoothing(self) -> bool:
        """Indicates whether smoothing is required for this semiring. 
        Set to False for probability.
        """
        return False

class MaxMinSemiring(Semiring):
    """Implements the fuzzy semiring: ([0,1], max, min, 0, 1)
    
    Methods:
        sum_op: maximum operator.
        product_op: minimum operator.
        zero_op: returns 0.
        one_op: returns 1.
        negation_op: returns 1 - x.
    """
    
    def one(self):
        """Returns the identity element for product equal to 1."""
        return self._mgr.one()
    def zero(self):
        """Returns the identity element for summation equal to 0."""
        return self._mgr.zero() 
    def sum(self,x, y):
        """Implementation of the summation operator using maximum."""
        return self._mgr.max(x, y)
    def product(self,x, y):
        """Implementation of the product operator using minimum."""
        return self._mgr.min(x, y)
    def negation(self,x):
        """Implementation of negation operator using 1 - x."""
        return self._mgr.neg(x)
    
    @property 
    def requires_smoothing(self) -> bool: 
        """Indicates whether smoothing is required for this semiring. Set to true 
        for this semiring to avoid zero counts."""
        return True
    @property 
    def no_weights(self) -> bool: 
        """Indicates whether this semiring uses no weights (all weights are 1). Set to 
        false for this semiring."""
        return False
    @property
    def log_prob(self) -> bool: 
        """Indicates whether this semiring uses log probabilities. Returns 
        false for this semiring."""
        return False
    
class FuzzyProduct(Semiring):
    def one(self):
        """Returns the identity element for product equal to 1."""
        return self._mgr.one()
    def zero(self):
        """Returns the identity element for summation equal to 0."""
        return self._mgr.zero() 
    def sum(self,x, y):
        """Implementation of the summation operator using maximum."""
        return self._mgr.add(x, y) - self._mgr.mult(x, y)
    def product(self,x, y):
        """Implementation of the product operator using minimum."""
        return self._mgr.mult(x, y)
    def negation(self,x):
        """Implementation of negation operator using 1 - x."""
        return self._mgr.neg(x)
    
    @property 
    def requires_smoothing(self) -> bool: 
        """Indicates whether smoothing is required for this semiring. Set to true 
        for this semiring to avoid zero counts."""
        return True
    @property 
    def no_weights(self) -> bool: 
        """Indicates whether this semiring uses no weights (all weights are 1). Set to 
        false for this semiring."""
        return False
    @property
    def log_prob(self) -> bool: 
        """Indicates whether this semiring uses log probabilities. Returns 
        false for this semiring."""
        return False
    
class FormulaSemiring(Semiring):
    """Semiring for building symbolic Boolean formulas. 

    Methods:
        sum_op: logical OR operation.
        product_op: logical AND operation.
        zero_op: returns False.
        one_op: returns True.
        negation_op: returns logical NOT.
    """
    def one(self):
        """Returns the identity element for product equal to True."""
        return self._mgr.logic_true()
    def zero(self):
        """Returns the identity element for summation equal to False."""
        return self._mgr.logic_false() 
    def sum(self,x, y):
        """Implementation of the summation operator using logical OR."""
        return self._mgr.logical_or(x, y)
    def product(self,x, y):
        """Implementation of the product operator using logical AND."""
        return self._mgr.logical_and(x, y)
    def negation(self,x):
        """Implementation of negation operator using logical NOT."""
        return self._mgr.logical_not(x)
    def simplify(self, expr):
        """Delegates simplification to the underlying manager."""
        return self._mgr.simplify(expr)
    @property
    def requires_smoothing(self) -> bool:
        """Indicates whether smoothing is required for this semiring. Set to true 
        for this semiring to avoid zero counts."""
        return False
    @property 
    def no_weights(self) -> bool: 
        """Indicates whether this semiring uses no weights (all weights are 1). Set to 
        false for this semiring."""
        return False
    @property
    def log_prob(self) -> bool: 
        """Indicates whether this semiring uses log probabilities. Returns 
        false for this semiring."""
        return False