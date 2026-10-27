import torch
import numpy as np
from collections.abc import Sequence
from numbers import Real
from ..numerical import SparseBeta, WeightRegistry

__all__ = ["PythonWeights"]

@WeightRegistry(name = "python")
class PythonWeights(SparseBeta[Real, Real]):
    """Class for managing weights via native python and numpy. 

    """
    @property 
    def weights(self):
        return self._params["weights"]
    
    @weights.setter
    def weights(self, new_weights: Sequence[Real]) -> None:
        """Sets new weights for the weight manager.
        
        Raises: 
            ValueError whenever the weights don't match. 
        """
        if len(new_weights) != len(self._params["weights"]):
            raise ValueError(
                "New weights must match shape of existing weights."
            ) from None
        self._params["weights"] = new_weights

    def _config_weights(self) -> None:
        """Called after initialization for configuring weight inputs. Maps 
        tensorized objects into lists.  

        Raises: 
            ValueError on unknown weight object or type. 
        """
        self._requires_grad = False
        lit_params = self._params["weights"]
        if isinstance(lit_params, torch.Tensor):
            self._params["weights"] = lit_params.detach().tolist()
        elif not (isinstance(lit_params, (tuple, list, np.ndarray))):
            raise ValueError(f"Unknown weight type: {type(lit_params)}")
    
    def identity(self, val: Real) -> Real:
        """The identity function. Returns the value back without modification.

        Arguments: 
            val: The input value. 
        """
        return val 
    
    def add(self, val1: Real, val2: Real) -> Real:
        """Method for computing addition.  

        Arguments: 
            val1: The first value in addition. 
            val2: The second value in addition.
        """
        return val1 + val2 # type: ignore
    
    def log_add(self, val1: Real, val2: Real) -> Real:
        """Method for computing addition in log probability space, defines
        `logaddexp` using `numpy.logaddexp`.

        Arguments: 
            val1: The first value in addition. 
            val2: The second value in addition.
        """
        return np.logaddexp(val1, val2) # type: ignore
    
    logaddexp = log_add  # alias

    def mult(self, val1: Real, val2: Real) -> Real:
        """Method for computing multiplication, uses standard Python
        multiplication. 

        Arguments: 
            val1: The first value in multiplication.
            val2: The second value in multiplication. 
        """
        return val1 * val2 # type: ignore

    def log_mult(self, val1: Real, val2: Real) -> Real:
        """Method for computing multiplication in log probability space using
        standard Python addition. 

        Arguments: 
            val1: The first value in multiplication.
            val2: The second value in multiplication. 
        """
        return val1 + val2  # type: ignore
    
    def div(self, val1: Real, val2: Real) -> Real:
        """Method for computing division. 

        Arguments: 
            val1: The first value in division.
            val2: The second value in division. 
        """
        return val1 / val2  # type: ignore

    def log_div(self, val1: Real, val2: Real) -> Real:
        """Method for computing division. 

        Arguments: 
            val1: The first value in division.
            val2: The second value in division. 
        """
        return val1 - val2  # type: ignore
    
    def neg(self, val: Real) -> Real:
        """Implementation of logical negation over literal probability. 

        Arguments: 
            val: Literal probability to negate. 
        """
        return 1.0 - val   # type: ignore
    
    def log_neg(self, val: Real) -> Real:
        """Implementation of logical negation over literal log probability, 
        using `math.log1p` and `math.exp`.  

        Arguments: 
            val: Literal log probability to negate. 
        """
        return np.log1p(-np.exp(val)) # type: ignore

    def log(self, val) -> Real:
        return np.log(val) # type: ignore

    def xlogx(self, val) -> Real:
        """Method for computing `val * log(val)` with `0 * log(0) = 0`."""
        return 0.0 if val == 0 else val * np.log(val) # type: ignore
    
    def clone(self, val: Real) -> Real:
        """Method for cloning weight values."""
        n = val
        return n
    
    def zero(self) -> Real:
        """Method for returning zero value."""
        return 0.0 # type: ignore
    
    def one(self) -> Real:
        """Method for returning one value."""
        return 1.0 # type: ignore   
        
    def inf(self) -> Real:
        """Method for returning infinity value."""
        return float('inf') # type: ignore
        
    def neg_inf(self) -> Real:
        """Method for returning negative infinity value."""
        return float('-inf') # type: ignore

    def max(self, val1: Real, val2: Real) -> Real:
        return max(val1, val2)
    
    def min(self, val1: Real, val2: Real):
        return min(val1, val2)
    
    def sigmoid(self, val: Real) -> Real:
        """Method for cloning weight values."""
        return 1 / (1 + np.exp(-val))  # type: ignore
        
    def log_sigmoid(self, val: Real) -> Real:
        """Method for cloning weight values."""
        return np.log(1 / (1 + np.exp(-val))) # type: ignore
    
    def exp(self, val: Real) -> Real:
        """Method for cloning weight values."""
        return np.exp(val) # type: ignore
    
    