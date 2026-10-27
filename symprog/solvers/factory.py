from .solver_interfaces import *
from .numerical_interfaces import *
from .semirings import *

from .mapper import MapperRegistry
from .numerical import WeightRegistry

__all__ = ["BuildMapper", "BuildWeights"]

def BuildMapper(solver: str, var_list, **kwargs):
    """Factory method for building a mapper instance.

    Arguments: 
        mapper_name: The name of the target mapper. 
        var_list: A list of input variables. 

    Example usage below: 

    >>> from symprog.solvers import BuildMapper 
    >>> s = BuildMapper(solver='z3', var_list=['a', 'b', 'c'])
    >>> a, b, c = s.Variables("a b c") 
    >>> formula = s.And(s.And(a, b), s.Not(c))
    """    
    mapper = MapperRegistry.build_instance(
        name=solver, var_list=var_list, **kwargs
    ) 
    return mapper 

def BuildWeights(manager: str, **kwargs):
    """Factory method for building a `WeightManager` object.

    Arguments: 
        manager: The name of the target manager to use. 
    
    Example usage below: 

    >>> from symprog.solvers import BuildWeights 
    >>> weights = [0.5, 0.6]; m = {"A" : 0, "B" : 1}
    >>> w = BuildWeights(manager='pytorch', weights=weights, wmap=m)
    >>> w.assignment("A")
    tensor(0.5000)
    """
    weights = WeightRegistry.build_instance(
        name=manager, **kwargs
    )
    return weights