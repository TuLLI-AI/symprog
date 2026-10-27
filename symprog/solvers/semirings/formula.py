from .common import FormulaSemiring
from ..semiring import SemiringRegistry, ProductSemiring
from ..solver_utils import SolverOut

__all__ = ["SymbolicFormula"]

@SemiringRegistry(name="Formula")
class SymbolicFormula(ProductSemiring[FormulaSemiring, FormulaSemiring]):
    """Implements the formula semiring for building symbolic Boolean formulas.
    
    Methods:
        sum_op: logical OR operation.
        product_op: logical AND operation.
        zero_op: returns False.
        one_op: returns True.
        negation_op: returns NOT x.
    """

    Leftsemiring = FormulaSemiring
    
    def process_output(self, output, simplify: bool = False,**kwargs) -> SolverOut:
        """Processes the output from the solver to generate a human-readable string.

        Arguments:
            output: The SolverOut object containing the result from the solver.
        Returns:
            A SolverOut object with a string representation of the result.
        """ 
        formula, cond_formula = output 
        details = None
        if cond_formula is not None:
            details = {'conditional_formula': cond_formula}
        if simplify is True and formula is not None:
            formula = formula.simplify()  
            
        return SolverOut(
            inference=formula,
            variables=None, 
            inference_type='Formula',
            details=details,
        )