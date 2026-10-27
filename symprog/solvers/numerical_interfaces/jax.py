"""EXPERIMENTAL JAX weight backend.

Supports inference and gradients (`Solver.gradient`), and exposes inference as a 
pure function of the weights via `Solver.as_function`, which can be transformed with 
`jax.grad`, `jax.jit` and `jax.vmap`. Not imported by default: it is registered lazily
when `container_type="jax"` is requested (requires `pip install 'symprog[jax]'`).

Known limitation: `ViterbiAssignment` compares values in Python, so it cannot be 
used under `jax.jit`.
"""
import warnings
import numpy as np
from collections.abc import Sequence
from numbers import Real
from typing import TypeAlias, Callable, Any

import jax
import jax.numpy as jnp
import jax.scipy.special

from ..numerical import SparseBeta
from ..numerical import WeightRegistry
from ...base import ExperimentalWarning

__all__ = ["JaxWeights"]

JaxArray: TypeAlias = Any
JaxValue: TypeAlias = (
    JaxArray
    | Real
    | np.ndarray
    | Sequence[Real]
)

@WeightRegistry(name="jax")
class JaxWeights(SparseBeta[JaxValue, JaxArray]):
    """EXPERIMENTAL: Class for managing weights via JAX. Since JAX differentiates 
    functions rather than computed values, gradients are computed from a function 
    of the weights (see `Solver.as_function`).
    """
    functional_gradients = True

    @property
    def weights(self) -> JaxArray:
        return self._params["weights"]

    @weights.setter
    def weights(self, new_weights: JaxValue) -> None:
        self._params["weights"] = jnp.asarray(new_weights)

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        self._requires_grad = bool(value)

    def _config_weights(self) -> None:
        """Called after initialization for configuring weight inputs. Maps
        non-array weights (e.g., Python lists, numpy arrays) into JAX arrays.
        """
        warnings.warn(
            "The JAX backend is experimental.",
            ExperimentalWarning,
            stacklevel=2,
        )
        lit_params = self._params["weights"]
        if isinstance(lit_params, (list, tuple, np.ndarray)):
            self._params["weights"] = jnp.asarray(lit_params)
        elif hasattr(lit_params, "dtype"):  # jax.Array-like
            # Already a JAX array; nothing to do
            pass
        else:
            # Last resort, try coercion
            try:
                self._params["weights"] = jnp.asarray(lit_params)
            except Exception as e:
                raise ValueError(f"Unknown weight object: {type(lit_params)}") from e

    def gradient(
        self, f: Callable[[JaxArray], JaxArray], key: str | None = None, **kwargs
    ) -> dict[str, JaxArray] | JaxArray:
        """Compute gradients of a callable `f(weights)` with respect to the current weights.
        Non-scalar outputs are summed.

        Raises:
            TypeError: If `f` is not callable (e.g., an already computed value).
        """
        if not callable(f):
            raise TypeError(
                "JAX gradients require a function of the weights, e.g., from `Solver.as_function`."
            )

        def scalar_f(w: JaxArray) -> JaxArray:
            out = f(w)
            return out if getattr(out, "ndim", 0) == 0 else jnp.sum(out)

        grads = jax.grad(scalar_f)(self._params["weights"])
        grad_map = {"weights": grads}
        if key is not None:
            return grad_map[key]
        return grad_map

    def identity(self, val: JaxValue) -> JaxArray:
        """Map value to JAX array."""
        return jnp.asarray(val)

    def detach(self, val: JaxValue) -> JaxArray:
        """Return the value without gradient tracking (traceable under `jax.grad`/`jax.jit`)."""
        return jax.lax.stop_gradient(self._map_value(val))

    def _map_value(self, val: JaxValue) -> JaxArray:
        """Coerce val to a JAX array."""
        if hasattr(val, "dtype"):  # jax.Array-like
            return val
        return jnp.asarray(val)

    def add(self, val1: JaxValue, val2: JaxValue) -> JaxArray:
        return self._map_value(val1) + self._map_value(val2)

    def log_add(self, val1: JaxValue, val2: JaxValue) -> JaxArray:
        v1 = self._map_value(val1)
        v2 = self._map_value(val2)
        ## mask -inf + -inf (log of zero), whose logaddexp gradient is otherwise nan
        both_neg_inf = jnp.isneginf(v1) & jnp.isneginf(v2)
        safe1 = jnp.where(both_neg_inf, 0.0, v1)
        safe2 = jnp.where(both_neg_inf, 0.0, v2)
        return jnp.where(both_neg_inf, v1, jnp.logaddexp(safe1, safe2))

    logaddexp = log_add  # alias

    def mult(self, val1: JaxValue, val2: JaxValue) -> JaxArray:
        return self._map_value(val1) * self._map_value(val2)

    def log_mult(self, val1: JaxValue, val2: JaxValue) -> JaxArray:
        return self._map_value(val1) + self._map_value(val2)

    def div(self, val1: JaxValue, val2: JaxValue) -> JaxArray:
        return self._map_value(val1) / self._map_value(val2)

    def log_div(self, val1: JaxValue, val2: JaxValue) -> JaxArray:
        return self._map_value(val1) - self._map_value(val2)

    def neg(self, val: JaxValue) -> JaxArray:
        return 1.0 - self._map_value(val)

    def log_neg(self, val: JaxValue) -> JaxArray:
        return jnp.log1p(-jnp.exp(self._map_value(val)))

    def log(self, val: JaxValue) -> JaxArray:
        return jnp.log(self._map_value(val))

    def xlogx(self, val: JaxValue) -> JaxArray:
        val = self._map_value(val)
        return jax.scipy.special.xlogy(val, val)

    def inf(self) -> JaxArray:
        return jnp.asarray(jnp.inf)

    def neg_inf(self) -> JaxArray:
        return jnp.asarray(-jnp.inf)

    def one(self) -> JaxArray:
        return jnp.asarray(1.0)

    def zero(self) -> JaxArray:
        return jnp.asarray(0.0)

    def clone(self, val: JaxValue) -> JaxArray:
        return jnp.copy(self._map_value(val))

    def max(self, val1: JaxValue, val2: JaxValue) -> JaxArray:
        return jnp.maximum(self._map_value(val1), self._map_value(val2))

    def min(self, val1: JaxValue, val2: JaxValue) -> JaxArray:
        return jnp.minimum(self._map_value(val1), self._map_value(val2))

    def sigmoid(self, val: JaxValue) -> JaxArray:
        return jax.nn.sigmoid(self._map_value(val))

    def log_sigmoid(self, val: JaxValue) -> JaxArray:
        return jax.nn.log_sigmoid(self._map_value(val))
    
    def exp(self, val: JaxValue) -> JaxArray:
        return jnp.exp(self._map_value(val))