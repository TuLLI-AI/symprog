import torch
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from collections.abc import Sequence
from numbers import Real
from typing import TypeAlias
from ..numerical import SparseBeta, WeightRegistry

__all__ = ["PytorchWeights"]

PytorchValue: TypeAlias = (
    Tensor
    | Real
    | np.ndarray
    | Sequence[Real]
)

def _safe_logaddexp(val1: Tensor, val2: Tensor) -> Tensor:
    """`torch.logaddexp` whose gradient is 0 (rather than nan) when both inputs
    are -inf (i.e., log of zero, which arises from evidence in log space)."""
    both_neg_inf = torch.isneginf(val1) & torch.isneginf(val2)
    if not both_neg_inf.any():
        return torch.logaddexp(val1, val2)
    ## mask inputs so that no nan gradients flow through the unused branch
    safe1 = torch.where(both_neg_inf, torch.zeros_like(val1), val1)
    safe2 = torch.where(both_neg_inf, torch.zeros_like(val2), val2)
    return torch.where(both_neg_inf, val1, torch.logaddexp(safe1, safe2))

@WeightRegistry(name = "pytorch")
class PytorchWeights(SparseBeta[PytorchValue, Tensor]):
    """Class for managing weights via pytorch."""
    _device: torch.device

    def _config_weights(self) -> None:
        """Called after initialization for configuring weight inputs. Maps 
        non-tensor weights (e.g., Python lists, numpy arrays) into tensors. 

        Raises: 
            ValueError on unknown weight object or type. 
        """
        ### set device correctly
        if self._device is None:
            self._device = torch.device("cpu")
        else:
            self._device = torch.device(self._device)

        lit_params = self._params["weights"]
        ## convert other weight types into tensor
        if isinstance(lit_params, (list, tuple, np.ndarray)):
            self._params["weights"] = torch.tensor(
                lit_params, device=self._device, requires_grad=self._requires_grad
            )
        elif isinstance(lit_params, Tensor):
            if self._requires_grad:
                lit_params.requires_grad = True
            if lit_params.device != self._device:
                self._params["weights"] = lit_params.to(self._device)
        else:
            raise ValueError(f"Unknown weight object: {type(lit_params)}") from None
        
    def gradient(self, f: Tensor, key: str | None = None, **kwargs) -> dict[str, Tensor] | Tensor:
        """Method for computing weight gradients. 

        Arguments: 
            f: The particular functions to compute gradients for.
        Raises: 
            NotImplementedErrror, is not implemented by default. 
        """
        if self._requires_grad is False:
            raise ValueError(
                "Cannot compute gradients when requires_grad is False."
            ) from None

        weights = self._params["weights"]
        if not (isinstance(f, Tensor) and f.requires_grad):
            ## output does not depend on the weights (e.g., fully determined by evidence)
            gradients = torch.zeros_like(weights)
        else:
            gradients = torch.autograd.grad(
                outputs = f,
                inputs = weights,
                allow_unused=True,
                retain_graph=False,
                create_graph=False
            )[0]
            if gradients is None:
                gradients = torch.zeros_like(weights)
        grad_map = { "weights" : gradients }
        if key is not None:
            return grad_map[key]
        return grad_map

    @property 
    def weights(self):
        return self._params["weights"]
    
    @weights.setter
    def weights(self, new_weights: Tensor) -> None:
        """Sets new weights for the weight manager.
        
        Raises: 
            ValueError whenever the new weights do not match the shape of existing weights.
        """
        if new_weights.shape != self._params["weights"].shape:
            raise ValueError(
                "New weights must match shape of existing weights."
            ) from None
        self._params["weights"] = new_weights.to(self._device)

    @property 
    def requires_grad(self) -> bool:
        """Indicates whether weights require gradient computation."""
        return self._requires_grad
    
    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        """Turns on gradient tracking for weights if not already done 
        and requested.
        
        Arguments: 
            value (bool): 
                Whether weights should require gradients.
        """
        for p in self._params.values():
            if isinstance(p, Tensor):
                p.requires_grad = value
        self._requires_grad = value

    def identity(self, val: Real) -> Tensor:
        """Method for mapping an input value into the right weight type
        
        Arguments: 
             val: The input value to map. 
        """
        return torch.as_tensor(val, device=self._device)
    
    def detach(self, val: Tensor) -> Tensor:
        """Method for mapping an input value into the right weight type
        
        Arguments: 
             val: The input value to map. 
        """
        return val.detach().cpu() 

    def _map_value(self, val: PytorchValue) -> Tensor:
        """Internal method for mapping tensor or non-tensor values to torch tensors, 
        and aligning the tensor devices with the specified device.

        Returns: 
            A tensorized version of `val`.
        """
        if not isinstance(val, Tensor):
            val = torch.as_tensor(val, device=self._device)
        elif val.device != self._device:
            val = val.to(self._device)
        return val

    def add(self, val1: PytorchValue, val2: PytorchValue) -> Tensor:
        """Method for computing addition, uses `pytorch.add` magic method. 

        Arguments: 
            val1: The first value in addition. 
            val2: The second value in addition.
        """
        return self._map_value(val1) + self._map_value(val2)
    
    def log_add(self, val1: PytorchValue, val2: PytorchValue) -> Tensor:
        """Method for computing addition, uses `pytorch.add` magic method. 

        Arguments: 
            val1: The first value in addition. 
            val2: The second value in addition.
        """
        return _safe_logaddexp(
            self._map_value(val1), self._map_value(val2)
        )

    logaddexp = log_add  # alias

    def mult(self, val1: PytorchValue, val2: PytorchValue) -> Tensor:
        """Method for computing multiplication, using `torch.prod`. 

        Arguments: 
            val1: The first value in multiplication.
            val2: The second value in multiplication. 
        """
        return self._map_value(val1) * self._map_value(val2)

    def log_mult(self, val1: PytorchValue, val2: PytorchValue) -> Tensor:
        """Method for computing multiplication in log probability space, using `torch.prod`. 

        Arguments: 
            val1: The first value in multiplication.
            val2: The second value in multiplication. 
        """
        return self._map_value(val1) + self._map_value(val2)
    
    def div(self, val1: PytorchValue, val2: PytorchValue) -> Tensor:
        """Method for computing multiplication in log probability space, using `torch.div`. 

        Arguments: 
            val1: The first value in multiplication.
            val2: The second value in multiplication. 
        """
        return self._map_value(val1) / self._map_value(val2)

    def log_div(self, val1: PytorchValue, val2: PytorchValue) -> Tensor:
        """Method for computing multiplication in log probability space, using torch. 

        Arguments: 
            val1: The first value in multiplication.
            val2: The second value in multiplication. 
        """
        return self._map_value(val1) - self._map_value(val2)
    
    def neg(self, val: PytorchValue) -> Tensor:
        """Method for the negationn of a literal and its probability.  

        Arguments: 
            val: The literal probability 
        """
        return 1.0 - self._map_value(val)

    def log_neg(self, val: PytorchValue) -> Tensor:
        """Method for the negationn of a literal and its probability given 
        in log probability space. Uses `torch.log1p` and `torch.exp`. 

        Arguments: 
            val: The literal probability 
        """
        return torch.log1p(-torch.exp(self._map_value(val)))

    def log(self, val: PytorchValue) -> Tensor:
        """Method for computing natural logarithm.
        """
        return torch.log(self._map_value(val))

    def xlogx(self, val: PytorchValue) -> Tensor:
        """Method for computing `val * log(val)` with `0 * log(0) = 0`."""
        val = self._map_value(val)
        return torch.special.xlogy(val, val)

    def inf(self) -> Tensor:
        """Method for returning infinity value."""
        return torch.as_tensor(torch.inf, device=self._device)
    
    def zero(self) -> Tensor:
        """Method for returning zero value."""
        return torch.as_tensor(0.0, device=self._device)

    def one(self) -> Tensor:
        """Method for returning one value."""
        return torch.as_tensor(1.0, device=self._device)

    def neg_inf(self) -> Tensor:
        """Method for returning negative infinity value."""
        return torch.as_tensor(-torch.inf, device=self._device)

    def clone(self, val: Tensor) -> Tensor:
        """Method for cloning weight values."""
        n = val.clone()
        return n
    
    def sigmoid(self, val: Tensor) -> Tensor:
        """Method for cloning weight values.
        
        Arguments:
            val: The input value.
        """
        return F.sigmoid(self._map_value(val))
    
    def log_sigmoid(self, val: Tensor) -> Tensor:
        """Method for cloning weight values.
        
        Arguments:
            val: The input value.
        """
        return F.logsigmoid(self._map_value(val))
    
    def exp(self, val: Tensor) -> Tensor:
        """Method for cloning weight values.
        
        Arguments:
            val: The input value.
        """
        return torch.exp(self._map_value(val))
    
    def max(self, val1: PytorchValue, val2: PytorchValue) -> Tensor:
        """Method for cloning weight values.
        
        Arguments: 
            val1: The first value.
            val2: The second value.
        """
        return torch.max(self._map_value(val1), self._map_value(val2))
    
    def min(self, val1: PytorchValue, val2: PytorchValue) -> Tensor:
        """Method for cloning weight values.
        
        Arguments: 
            val1: The first value.
            val2: The second value.
        """
        return torch.min(self._map_value(val1), self._map_value(val2))

@WeightRegistry(name = "fast_pytorch")
class PytorchWeightsFast(PytorchWeights):
    """Faster version of PytorchWeights that assumes all inputs are tensors already."""

    def _map_value(self, val: Tensor) -> Tensor:
        """Internal method for mapping tensor or non-tensor values to torch tensors, 
        and aligning the tensor devices with the specified device.

        In this fast version, it assumes a tensor is already provided, so it is 
        equal to the identity function. 

        Returns: 
            A tensorized version of `val`.
        """
        return val

    def _check_maps(self):
        pass 
    
    ## manually removed `_map_value` to avoid any potential bottlenecks
    
    def add(self, val1: Tensor, val2: Tensor) -> Tensor:
        return val1 + val2

    def mult(self, val1: Tensor, val2: Tensor) -> Tensor:
        return val1 * val2

    def log_add(self, val1: Tensor, val2: Tensor) -> Tensor:
        return _safe_logaddexp(val1, val2)

    logaddexp = log_add  # alias

    def log_mult(self, val1: Tensor, val2: Tensor) -> Tensor:
        return val1 + val2

    def div(self, val1: Tensor, val2: Tensor) -> Tensor:
        return val1 / val2

    def log_div(self, val1: Tensor, val2: Tensor) -> Tensor:
        return val1 - val2

    def neg(self, val: Tensor) -> Tensor:
        return 1.0 - val

    def log_neg(self, val: Tensor) -> Tensor:
        return torch.log1p(-torch.exp(val))

    def log(self, val):
        """Method for computing natural logarithm.
        """
        return torch.log(val)
    
    def max(self, val1: Tensor, val2: Tensor) -> Tensor:
        """Method for max."""
        return torch.max(val1, val2)
    
    def min(self, val1: Tensor, val2: Tensor) -> Tensor:
        """Method for min."""
        return torch.min(val1, val2)
    
    def sigmoid(self, val: Tensor) -> Tensor:
        """Method for cloning weight values."""
        return F.sigmoid(val)
    
    def log_sigmoid(self, val: Tensor) -> Tensor:
        """Method for cloning weight values."""
        return F.logsigmoid(val)
    
    def exp(self, val: Tensor) -> Tensor:
        """Method for cloning weight values."""
        return torch.exp(val)