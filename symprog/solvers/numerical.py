from abc import abstractmethod
from typing import Generic, TypeVar

from ..base import UtilBase
from ..registry import Registry

I = TypeVar("I")
T = TypeVar("T")

class WeightManager(UtilBase, Generic[I, T]):
    """Base class defining and managing weights in probabilistic 
    models and model parameters. Defines arithemtic operations over weights 
    (or mappings from `I -> T`). 

    Attributes: 
        device: The particular device to host weights on.
        weights: The underlying weights of the model.
        functional_gradients: Whether `gradient` takes a function of the weights
            (e.g., JAX) rather than a computed value (e.g., PyTorch, sympy).
    """
    functional_gradients: bool = False

    @abstractmethod
    def _config_weights(self) -> None:
        """Called after initialization for configuring weight inputs"""
        raise NotImplementedError
    
    def identity(self, val):
        """Method for mapping an input value into the right weight type
        
        Arguments: 
             val: The input value to map. 
        """
        return val
    
    def detach(self, val): 
        """Returns a detached version of the value. By value, simply returns 
        the value (for interfaces like Pytorch, detach is used here to detach
        values from computation graphs). 
        """
        return val 
    
    ### base methods to implement
    def gradient(self, f: T, key: str | None = None, **kwargs) -> dict[str, T] | T:
        """Method for computing weight gradients

        By default is unimplemented. 

        Arguments: 
            f: The particular functions to compute gradients for.
        Raises: 
            NotImplementedErrror, is not implemented by default. 
        """
        raise NotImplementedError('Gradients not supported by this class')
    
    @abstractmethod
    def add(self, val1: I, val2: I) -> T:
        """Method for computing addition.  

        Arguments: 
            val1: The first value in addition. 
            val2: The second value in addition.
        """
        raise NotImplementedError
    
    @abstractmethod
    def log_add(self, val1: I, val2: I) -> T:
        """Method for computing addition in log probability space.  

        Arguments: 
            val1: The first value in addition. 
            val2: The second value in addition.
        """
        raise NotImplementedError
    
    logaddexp = log_add  # alias

    @abstractmethod
    def mult(self, val1: I, val2: I) -> T:
        """Method for computing multiplication.    

        Arguments: 
            val1: The first value in multiplication.
            val2: The second value in multiplication. 
        """
        raise NotImplementedError
    
    @abstractmethod
    def log(self, val: I) -> T:
        """Method for computing natural logarithm.
        """
        raise NotImplementedError

    def xlogx(self, val: I) -> T:
        """Method for computing `val * log(val)`, with the convention that
        `0 * log(0) = 0` (e.g., for entropy computations). Numerical managers
        should override this to handle zero values.
        """
        return self.mult(val, self.log(val))

    @abstractmethod
    def log_mult(self, val1: I, val2: I) -> T:
        """Method for computing multiplication in log probability space.    

        Arguments: 
            val1: The first value in multiplication.
            val2: The second value in multiplication. 
        """
        raise NotImplementedError

    @abstractmethod
    def div(self, val1: I, val2: I) -> T:
        """Method for computing division. 

        Arguments: 
            val1: The first value in division.
            val2: The second value in division. 
        """
        raise NotImplementedError
    
    @abstractmethod
    def log_div(self, val1: I, val2: I) -> T:
        """Method for computing division in log probability space. 

        Arguments: 
            val1: The first value in division.
            val2: The second value in division. 
        """
        raise NotImplementedError
    
    @abstractmethod
    def neg(self, val: I) -> T:
        """Implementation of logical negation over literal probability. 

        Arguments: 
            val: Literal probability to negate. 
        """
        raise NotImplementedError

    @abstractmethod
    def log_neg(self, val: I) -> T:
        """Implementation of logical negation over literal log probability. 

        Arguments: 
            val: Literal log probability to negate. 
        """
        raise NotImplementedError
    
    @abstractmethod
    def inf(self) -> T:
        """Returns infinity value for the weight type."""
        raise NotImplementedError
    @abstractmethod
    def neg_inf(self) -> T:
        """Returns negative infinity value for the weight type."""
        raise NotImplementedError
    @abstractmethod
    def one(self) -> T:
        """Method for returning one constant"""
        raise NotImplementedError
    @abstractmethod
    def zero(self) -> T:
        """Method for returning zero constant"""
        raise NotImplementedError
    
    @abstractmethod
    def clone(self, val: I) -> T:
        """Method for cloning weight values.
        
        Arguments: 
            val: The value to clone.
        """
        raise NotImplementedError
    
    @abstractmethod
    def max(self, val1: I, val2: I) -> T:
        """Method for returning maximum of two values
        
        Arguments:
            val1: The first value to compare.
            val2: The second value to compare.
        """
        raise NotImplementedError
    
    @abstractmethod
    def min(self, val1: I, val2: I) -> T:
        """Method for returning minimum of two values
        
        Arguments: 
            val1: The first value to compare.
            val2: The second value to compare.
        """
        raise NotImplementedError
    
    @abstractmethod
    def exp(self, val: I) -> T:
        """Method for computing the exponential of a value.
        
        Arguments: 
            val: The value to exponentiate.
        """
        raise NotImplementedError
    
    @abstractmethod
    def sigmoid(self, val: I) -> T:
        """Method for computing the sigmoid of a value.
        
        Arguments: 
            val: The value to compute sigmoid for.
        """
        raise NotImplementedError
    
    @abstractmethod
    def log_sigmoid(self, val: I) -> T:
        """Method for computed the log sigmoid of a value.
        
        Arguments: 
            val: The value to compute sigmoid for.
        """
        raise NotImplementedError
    
    def assignment(self, identifier: int | str) -> T:
        raise NotImplementedError("assignment method not implemented for base WeightManager")
    
    @property 
    def symbolic(self) -> bool:
        """Indicates whether this weight manager uses symbolic weights. Defaults to False."""
        return False
    
    @property 
    def requires_grad(self) -> bool:
        """Indicates whether the weight manager requires gradient computation."""
        raise NotImplementedError("requires_grad property not implemented for WeightRegistry")
    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        raise NotImplementedError("requires_grad setter not implemented for WeightRegistry")
    
    ## logical operations 
    def logical_and(self, val1: I, val2: I) -> T:
        """Method for computing logical AND operation.
        """
        raise NotImplementedError("Not Implemented for general numerical classes")
    
    def logical_or(self, val1: I, val2: I) -> T:
        """Method for computing logical OR operation.
        """
        raise NotImplementedError("Not Implemented for general numerical classes")
    
    def logical_not(self, val: I) -> T:
        """Method for computing logical NOT operation.
        """
        raise NotImplementedError("Not Implemented for general numerical classes")
    
    def logical_imply(self, val1: I, val2: I) -> T:
        """Method for computing logical IMPLICATION operation.
        """
        raise NotImplementedError("Not Implemented for general numerical classes")
    
    def logical_bicond(self, val1: I, val2: I) -> T:
        """Method for computing logical BICONDITIONAL operation.
        """
        raise NotImplementedError("Not Implemented for general numerical classes")
    
    def logical_xor(self, val1: I, val2: I) -> T:
        """Method for computing logical XOR operation.
        """
        raise NotImplementedError("Not Implemented for general numerical classes")
    
    def logic_true(self) -> T:
        """Returns the logical True value for the weight type."""
        raise NotImplementedError("Not Implemented for general numerical classes")
    def logic_false(self) -> T:
        """Returns the logical False value for the weight type."""
        raise NotImplementedError("Not Implemented for general numerical classes")

    def simplify(self, expr: T) -> T:
        """Simplifies a symbolic expression. Default is a no-op."""
        return expr


class WeightRegistry(Registry[WeightManager]):
    """Resitry for storing different classes the manage weights."""
    
    @classmethod
    def build_instance(
        cls, 
        name: str, 
        symbolic: bool = False,
        **kwargs
    ) -> WeightManager:
        """Builds an instance of a weight manager from the registry.

        Arguments:
            name: The name of the weight manager to build.
            symbolic: Whether to build a symbolic weight manager.
            **kwargs: Keyword arguments to pass to the weight manager constructor.
        Returns:
            An instance of the requested weight manager.

        Raises:
            ValueError: If the requested weight manager is not registered or symbolic
                version is requested but wrong name is provided.
        """
        if name == "jax" and name not in cls._IMPLEMENTATIONS:
            ## experimental backend, registered lazily to avoid importing jax by default
            try:
                from .numerical_interfaces import jax as _jax_backend  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    "The experimental JAX backend requires jax: pip install 'symprog[jax]'"
                ) from e

        implementation_tuple = cls._IMPLEMENTATIONS.get(name)
        if implementation_tuple is None:
            impl = list(cls._IMPLEMENTATIONS.keys())
            raise ValueError(
                f"Resource not found registered, value={name}, available classes={', '.join(impl)}"
            )
                
        impl_cls, config_cls = implementation_tuple
        if symbolic and not issubclass(impl_cls, SparseSymbolic):
            raise ValueError(
                f"Requested symbolic weight manager, but {name} is not symbolic."
            )
        
        if config_cls is None: 
            return impl_cls.from_config(**kwargs)
        
        config = config_cls.build_config(**kwargs)
        return impl_cls.from_config(config) # type: ignore 
    



class SparseBeta(WeightManager[I, T], Generic[I, T]):
    """Base class for defining weight managers with sparse literal parameters 
    and (optional) beta parameters. Currently this covers our current use 
    cases. 

    Attributes: 
        parameters: The underlying sparse weight parameters. 
        parameter_map: The map from symbols to those parameter values. 
        beta_parameters: Underlying beta parameters (optional).
        beta_map: Mapping from symbols to those parameter values.

    """
    def __init__(
        self,
        weights: I,
        wmap: dict[str, int] | None = None,
        beta_weights: I | None = None,
        beta_map: dict[str, int] | None = None,
        device: str | None = None,
        requires_grad: bool = False,
        **kwargs
    ) -> None:
        """Initializes weight instance. 

        Arguments: 
            literal weights: The weights of literals in logic. 
            beta_weights: (optional) beta weights (TODO: implement)
            device: The particular device to host weights on. (Note: 
                only needed for pytorch). 
        """
        self._params = {
            "weights" : weights, "beta": beta_weights,
        }
        self._wmap = wmap
        self._beta_map = beta_map
        self._device = device
        self._requires_grad = requires_grad
        
        ### set up individual weighs
        self._check_maps() 
        self._config_weights()

    def _check_maps(self):
        """Ensures that maps are correct, builds identity maps if neded."""
        lit_params = self._params["weights"]
        if self._wmap is None:
            self._wmap = {str(i) : i for i in range(len(lit_params))}
        elif self._wmap is not None: 
            self._wmap = {str(k) : v for k, v in self._wmap.items()}
        if len(self._wmap) != len(lit_params):
            raise ValueError('Dimension of literal map does not match params')

    def __repr__(self) -> str:
        values = []
        for name in sorted(self._wmap, key=self._wmap.get):
            value = self._params["weights"][self._wmap[name]]
            if hasattr(value, "detach"):
                value = value.detach()
            try:
                values.append(f"{name}={float(value):.4g}")
            except (TypeError, ValueError):
                values.append(f"{name}={value}")
        return f"{type(self).__name__}({', '.join(values)})"

    @property
    def requires_grad(self) -> bool:
        """Indicates whether weights require gradient computation."""
        return self._requires_grad
    
    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        self._requires_grad = value
    

    ## weight attributes 
    
    @property 
    def weights(self):
        return self._params["weights"]

    # @weights.setter
    # @abstractmethod
    # def weights(self, new_weights: Sequence[Real]) -> None:
    #     """Sets new weights for the weight manager."""
    #     raise NotImplementedError 

    @property
    def device(self):
        """Returns a description of the host device."""
        return self._device
    
    @property
    def parameters(self):
        return self._params
    @property
    def wmap(self):
        return self._wmap
    @property 
    def beta_map(self):
        return self._beta_map

    def assignment(self, identifier: int | str) -> T:
        """Returns a weight for a given literal based on its id. 

        Arguments: 
            identifier: The unique id of the target literal. 
        Raises: 
            KeyError when unknown identifier is encountered
        """   
        if self._wmap is None or not isinstance(self._wmap, dict):
            raise KeyError(
                f"Bad initialization, weight weight not specified, "
                f"unknown identifier: {identifier}"
            ) from None
        try:
            idx = self._wmap[str(identifier)]
        except (AttributeError, KeyError, TypeError):
            raise KeyError(
                f"Unknown literal identifier encountered: {identifier}"
            ) from None
        return self._params["weights"][idx]

class SparseSymbolic(SparseBeta[I, T], Generic[I, T]):
    """Base class for defining weight managers with symbolic 
    parameters and operations. Can be used, for example, for building 
    symbolic equations for circuit or BDD computations.  
    
    Attributes: 
        parameters: The underlying weight parameters. 
        parameter_map: The map from symbols to those parameter values. 
    Raises: 
        ValueError whenever variable names are not provided or the names don't 
            match the number of provided parameters. 
    """
    def __init__(
        self, 
        real: bool = True, 
        pos: bool = True, 
        nonzero: bool = True,
        boolean: bool = False,
        **kwargs
    ) -> None: 
        """Initializes a symbolic sparse weight manager. 
        
        Arguments: 
            real: Boolean indicating whether weights are `real`. (default=True)
            pos: Boolean indicating whether weights are `positive`. (default=True)
        """
        self._real = real if not boolean else None
        self._pos = pos if not boolean else None
        self._nonzero = nonzero if not boolean else None
        self._boolean = boolean
        super().__init__(**kwargs)

    def _check_maps(self) -> None:
        """Ensures that maps are correct, builds identity maps if neded.
        
        Raises: 
            ValueError whenever variable names are not provided or the names don't 
                match the number of provided parameters.
        """
        lit_params = self._params["weights"]
        if self._wmap is None:
            raise ValueError("For symbolic weights, weight names must be proivided")
        if len(self._wmap) != len(lit_params):
            raise ValueError('Dimension of literal map does not match params')
        
        if not max(lit_params) > 0.0:
            self._pos = False
            self._nonzero = False 
                
    def _config_weights(self) -> None: 
        """Creates symbolic weight variables. Maps each weight (by symbol) to a symbolic variable."""
        self._params["sym_weights"] = {}
        if self._wmap is not None: 
            for symbol, _ in self._wmap.items():
                self._params["sym_weights"][str(symbol)] = self.identity(symbol)

    def assignment(self, identifier: int | str) -> T:
        """Returns a weight for a given literal based on its id. 

        Arguments: 
            identifier: The unique id of the target literal. 
        Raises: 
            KeyError when unknown identifier is encountered
        """
        try:
            sym_var = self._params["sym_weights"][identifier]
        except KeyError:
            raise KeyError(
                f"Unknown symbolic variables: {identifier}"
            ) from None
        return sym_var

    @abstractmethod
    def simplify(self, expr: T) -> T:
        """Function for simplifying symbolic expresisons
        
        Arguments: 
            expr: The input expression to simplify. 
        """
        raise NotImplementedError  
    
    @property 
    def symbolic(self) -> bool:
        """Indicates whether this weight manager uses symbolic weights. 
        Defaults to True."""
        return True
    
    ### added logical methods for symbolic weights

    @abstractmethod
    def logical_and(self, val1: I, val2: I) -> T:
        """Method for computing logical AND operation.

        Arguments:
            val1: The first value in AND operation.
            val2: The second value in AND operation.
        """
        raise NotImplementedError
    
    @abstractmethod
    def logical_or(self, val1: I, val2: I) -> T:
        """Method for computing logical OR operation.

        Arguments:
            val1: The first value in OR operation.
            val2: The second value in OR operation.
        """
        raise NotImplementedError       
    
    @abstractmethod
    def logical_not(self, val: I) -> T:
        """Method for computing logical NOT operation.

        Arguments:
            val: The value to negate.
        """
        raise NotImplementedError
    
    @abstractmethod
    def logical_imply(self, val1: I, val2: I) -> T:
        """Method for computing logical IMPLICATION operation.

        Arguments:
            val1: The antecedent value.
            val2: The consequent value.
        """
        raise NotImplementedError
    
    @abstractmethod
    def logical_bicond(self, val1: I, val2: I) -> T:
        """Method for computing logical BICONDITIONAL operation.

        Arguments:
            val1: The antecedent value.
            val2: The consequent value.
        """
        raise NotImplementedError
    
    @abstractmethod
    def logical_xor(self, val1: I, val2: I) -> T:
        """Method for computing logical XOR operation.

        Arguments:
            val1: The first value.
            val2: The second value.
        """
        raise NotImplementedError
    
    @abstractmethod
    def logic_true(self, val1: I, val2: I) -> T:
        """Returns the logical True value for the weight type."""
        raise NotImplementedError
    @abstractmethod
    def logic_false(self, val1: I, val2: I) -> T:
        """Returns the logical False value for the weight type."""
        raise NotImplementedError