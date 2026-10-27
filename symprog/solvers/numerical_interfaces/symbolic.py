from typing import Optional, TypeAlias
import sympy
import symengine
from sympy import Expr as SympyExpr, Function
from symengine import Expr as SymengineExpr
from ..numerical import SparseSymbolic, WeightRegistry

__all__ = ["SympySymbolic", "SymengineSymbolic"]

SympyValue: TypeAlias = (
    SympyExpr 
    | str
    | int
)

class LogAddExp(Function): 
    nargs = (2,)
    is_real = True 

    @classmethod 
    def eval(cls, a, b):
        if a is sympy.S.NegativeInfinity:
            return b
        if b is sympy.S.NegativeInfinity:
            return a
        if a == b:
            return a + sympy.log(2)
        return None 
    
    def _eval_simplify(self, **kwargs):
        return self 
    
    def _eval_evalf(self):
        a, b = self.args
        return sympy.log(sympy.exp(a) + sympy.exp(b)) # type: ignore 
    

@WeightRegistry(name="sympy")
class SympySymbolic(SparseSymbolic[SympyValue, SympyExpr]): 
    """Symbolic weights via sympy symbolic equations."""

    def identity(self, val: SympyValue) -> SympyExpr:
        """Method for mapping an input value into the right weight type
        
        Arguments: 
             val: The input value to map. 
        """
        if isinstance(val, str):
            return sympy.Symbol(
                val, 
                real=self._real, 
                nonzero=self._nonzero, 
                positive=self._pos
            )
        elif isinstance(val, SympyExpr):
            return val
        else:
            raise ValueError(f"Unknown symbolic weight type: {type(val)}")
        
    ### base methods to implement
    def add(self, val1: SympyValue, val2: SympyValue) -> SympyExpr:
        """Method for computing addition.  

        Arguments: 
            val1: The first value in addition. 
            val2: The second value in addition.
        """
        val1, val2 = self.identity(val1), self.identity(val2)
        ## -oo (log of zero) is absorbing; sympy otherwise leaves e.g. `log(1 - a) - oo` unevaluated
        if val1 == -sympy.oo or val2 == -sympy.oo:
            return -sympy.oo
        return val1 + val2  # type: ignore 
    
    def log_add(self, val1: SympyValue, val2: SympyValue) -> SympyExpr:
        """Method for computing addition in log probability space.  

        Arguments: 
            val1: The first value in addition. 
            val2: The second value in addition.
        """
        val1, val2 = self.identity(val1), self.identity(val2)
        ## -oo (log of zero) is the identity 
        if val1 == -sympy.oo:
            return val2
        if val2 == -sympy.oo:
            return val1
        return sympy.log(
            sympy.exp(val1) + sympy.exp(val2) # type: ignore 
        ) # type: ignore 
    
    logaddexp = log_add  # alias

    def mult(self, val1: SympyValue, val2: SympyValue) -> SympyExpr:
        """Method for computing multiplication.    

        Arguments: 
            val1: The first value in multiplication.
            val2: The second value in multiplication. 
        """
        return self.identity(val1) * self.identity(val2) # type: ignore 
    
    def log_mult(self, val1: SympyValue, val2: SympyValue) -> SympyExpr:
        """Method for computing multiplication in log probability space.    

        Arguments: 
            val1: The first value in multiplication.
            val2: The second value in multiplication. 
        """
        return self.add(val1, val2)
    
    def log(self, val: SympyValue) -> SympyExpr:
        """Method for computing natural logarithm.

        Arguments: 
            val: The input value. 
        """
        return sympy.log(self.identity(val)) # type: ignore 

    def xlogx(self, val: SympyValue) -> SympyExpr:
        """Method for computing `val * log(val)` with `0 * log(0) = 0`."""
        val = self.identity(val)
        if val == 0:
            return sympy.S.Zero
        return val * sympy.log(val) # type: ignore 
    
    def div(self, val1: SympyValue, val2: SympyValue) -> SympyExpr:
        """Method for computing division. 

        Arguments: 
            val1: The first value in division.
            val2: The second value in division. 
        """
        return self.identity(val1) / self.identity(val2) # type: ignore 
    
    def log_div(self, val1: SympyValue, val2: SympyValue) -> SympyExpr:
        """Method for computing division in log probability space. 

        Arguments: 
            val1: The first value in division.
            val2: The second value in division. 
        """
        return self.identity(val1) - self.identity(val2) # type: ignore 
    
    def min(self, val1: SympyValue, val2: SympyValue) -> SympyExpr:
        """Method for computing minimum of two values.

        Arguments:
            val1: The first value.
            val2: The second value.
        """
        return sympy.Min(self.identity(val1), self.identity(val2))
    
    def max(self, val1: SympyValue, val2: SympyValue) -> SympyExpr:
        """Method for computing maximum of two values.

        Arguments:
            val1: The first value.
            val2: The second value.
        """
        return sympy.Max(self.identity(val1), self.identity(val2))
        
    def exp(self, val: SympyValue) -> SympyExpr:
        """Method for computing the exponential function.

        Arguments: 
            val: The input value. 
        """
        return sympy.exp(self.identity(val)) #type: ignore 
    
    def sigmoid(self, val: SympyValue) -> SympyExpr:
        """Method for computing the sigmoid function.

        Arguments: 
            val: The input value. 
        """
        x = self.identity(val)
        return 1 / (1 + sympy.exp(-x)) # type: ignore 
    
    def log_sigmoid(self, val: SympyValue) -> SympyExpr:
        """Method for computing the log sigmoid function.

        Arguments: 
            val: The input value. 
        """
        x = self.identity(val)
        return -sympy.log(1 + sympy.exp(-x)) # type: ignore 
    
    
    def neg(self, val: SympyValue) -> SympyExpr:
        """Method for the negationn of a literal and its probability.  

        Arguments: 
            val: The literal probability 
        """
        return 1 - self.identity(val) # type: ignore 
    
    def log_neg(self, val: SympyValue) -> SympyExpr:
        """Method for the negationn of a literal and its probability given 
        in log probability space. Uses `log1p` and `exp`. 

        Arguments: 
            val: The literal probability 
        """
        return sympy.log(1 - sympy.exp(self.identity(val))) # type: ignore 
    
    def clone(self, val: SympyValue) -> SympyExpr:
        """Method for cloning weight values.

        Arguments: 
            val: The value to clone. 
        """
        return self.identity(val)
    
    def inf(self) -> SympyExpr:
        """Returns infinity value for the weight type."""
        return sympy.oo
    
    def neg_inf(self) -> SympyExpr:
        """Returns negative infinity value for the weight type."""
        return -sympy.oo
    
    def one(self) -> SympyExpr:
        """Method for returning one constant"""
        return sympy.S.One
    
    def zero(self) -> SympyExpr:
        """Method for returning zero constant"""
        return sympy.S.Zero
    
    def gradient(
        self, f: SympyExpr, key: Optional[str] = None, boolean: bool = False, simplify: bool = False, **kwargs
    ) -> dict[str, SympyExpr| list[SympyExpr]] | SympyExpr:
        """Method for computing weight gradients. Returns back gradient 
        vector for weight parameters. 
    
        Note: assumes maps is contiguous. TODO: need to check 
        when loading the weight map for all classes. 

        Arguments: 
            f: The particular functions to compute gradients for.
        """
        if self._wmap is None or not isinstance(self._wmap, dict):
            raise ValueError(
                "Weight map not initialized for gradient computation."
            )
        
        pderivs = []

        if boolean is False:
            for var in sorted(self._wmap.keys()):
                w = self._params["sym_weights"][var]
                df_dw = sympy.diff(f, w)
                pderivs.append(df_dw)
            grad_map = { "weights": sympy.Matrix(pderivs) }
        else: 
            ## boolean partial derivatives via XOR
            xor = lambda a, b: sympy.Or(
                sympy.And(a, sympy.Not(b)),
                sympy.And(sympy.Not(a), b)
            )
            for var in sorted(self._wmap.keys()):
                w = self._params["sym_weights"][var]
                bpartial = xor(f.subs(w, sympy.S.true), f.subs(w, sympy.S.false))
                if simplify is True:
                    bpartial = bpartial.simplify()
                pderivs.append(bpartial)

            grad_map = { "weights": pderivs }
            
        if key is not None:
            return grad_map[key] # type: ignore 
        return grad_map # type: ignore 
    
    def simplify(self, expr: SympyExpr) -> SympyExpr:
        """Function for simplifying symbolic expresisons
        
        Arguments: 
            expr: The input expression to simplify. 
        """
        #e = sympy.simplify(expr)
        return sympy.factor(expr)
    
    ### logical operations 
    
    def logical_and(self, val1: SympyExpr, val2: SympyExpr):
        """Method for computing logical AND operation.

        Arguments:
            val1: The first value in AND operation.
            val2: The second value in AND operation.
        """
        return sympy.And(val1, val2)
    
    def logical_or(self, val1: SympyExpr, val2: SympyExpr):
        """Method for computing logical OR operation.

        Arguments:
            val1: The first value in OR operation.
            val2: The second value in OR operation.
        """
        return sympy.Or(val1, val2)
    
    def logical_not(self, val: SympyExpr):
        """Method for computing logical NOT operation.

        Arguments:
            val: The value to negate.
        """
        return sympy.Not(val)
    
    def logical_imply(self, val1: SympyExpr, val2: SympyExpr):
        """Method for computing logical IMPLICATION operation.

        Arguments:
            val1: The antecedent value.
            val2: The consequent value.
        """
        return sympy.Implies(val1, val2)
    
    def logical_bicond(self, val1: SympyExpr, val2: SympyExpr):
        """Method for computing logical BICONDITIONAL operation.

        Arguments:
            val1: The antecedent value.
            val2: The consequent value.
        """
        return sympy.And(
            sympy.Implies(val1, val2),
            sympy.Implies(val2, val1)
        )
    
    def logical_xor(self, val1: SympyExpr, val2: SympyExpr):
        """Method for computing logical XOR operation.

        Arguments:
            val1: The first value.
            val2: The second value.
        """
        return sympy.Or(
            sympy.And(val1, sympy.Not(val2)),
            sympy.And(sympy.Not(val1), val2)
        )
    
    def logic_true(self):
        """Returns the logical True value for the weight type."""
        return sympy.S.true
    def logic_false(self):
        """Returns the logical False value for the weight type.""" 
        return sympy.S.false
    
SymengineValue: TypeAlias = SymengineExpr | str | int


@WeightRegistry(name="symengine")
class SymengineSymbolic(SparseSymbolic[SymengineValue, SymengineExpr]):
    """Symbolic weights via symengine — same interface as SympySymbolic but
    backed by symengine's C++ expression tree, which is significantly faster
    for expression construction during SDD traversal.

    Logical operations are not supported (symengine has no Boolean logic);
    use SympySymbolic for boolean gradient computation.

    The expression produced is compatible with sympy.sympify(), so the
    downstream CSE + FX compilation pipeline in tensorization.py is unchanged.
    """

    def identity(self, val: SymengineValue) -> SymengineExpr:
        if isinstance(val, str):
            return symengine.Symbol(val)
        elif isinstance(val, SymengineExpr):
            return val
        else:
            raise ValueError(f"Unknown symbolic weight type: {type(val)}")

    def add(self, val1: SymengineValue, val2: SymengineValue) -> SymengineExpr:
        val1, val2 = self.identity(val1), self.identity(val2)
        ## -oo (log of zero) is absorbing
        if val1 == -symengine.oo or val2 == -symengine.oo:
            return -symengine.oo
        return val1 + val2

    def log_add(self, val1: SymengineValue, val2: SymengineValue) -> SymengineExpr:
        val1, val2 = self.identity(val1), self.identity(val2)
        ## -oo (log of zero) is the identity
        if val1 == -symengine.oo:
            return val2
        if val2 == -symengine.oo:
            return val1
        return symengine.log(symengine.exp(val1) + symengine.exp(val2))

    logaddexp = log_add

    def mult(self, val1: SymengineValue, val2: SymengineValue) -> SymengineExpr:
        return self.identity(val1) * self.identity(val2)

    def log_mult(self, val1: SymengineValue, val2: SymengineValue) -> SymengineExpr:
        return self.add(val1, val2)

    def log(self, val: SymengineValue) -> SymengineExpr:
        return symengine.log(self.identity(val))

    def xlogx(self, val: SymengineValue) -> SymengineExpr:
        val = self.identity(val)
        if val == 0:
            return symengine.S.Zero
        return val * symengine.log(val)

    def div(self, val1: SymengineValue, val2: SymengineValue) -> SymengineExpr:
        return self.identity(val1) / self.identity(val2)

    def log_div(self, val1: SymengineValue, val2: SymengineValue) -> SymengineExpr:
        return self.identity(val1) - self.identity(val2)

    def exp(self, val: SymengineValue) -> SymengineExpr:
        return symengine.exp(self.identity(val))

    def sigmoid(self, val: SymengineValue) -> SymengineExpr:
        x = self.identity(val)
        return symengine.S.One / (symengine.S.One + symengine.exp(-x))

    def log_sigmoid(self, val: SymengineValue) -> SymengineExpr:
        x = self.identity(val)
        return -symengine.log(symengine.S.One + symengine.exp(-x))

    def neg(self, val: SymengineValue) -> SymengineExpr:
        return symengine.S.One - self.identity(val)

    def log_neg(self, val: SymengineValue) -> SymengineExpr:
        return symengine.log(symengine.S.One - symengine.exp(self.identity(val)))

    def clone(self, val: SymengineValue) -> SymengineExpr:
        return self.identity(val)

    def min(self, val1: SymengineValue, val2: SymengineValue) -> SymengineExpr:
        return symengine.Min(self.identity(val1), self.identity(val2))

    def max(self, val1: SymengineValue, val2: SymengineValue) -> SymengineExpr:
        return symengine.Max(self.identity(val1), self.identity(val2))

    def inf(self) -> SymengineExpr:
        return symengine.oo

    def neg_inf(self) -> SymengineExpr:
        return -symengine.oo

    def one(self) -> SymengineExpr:
        return symengine.S.One

    def zero(self) -> SymengineExpr:
        return symengine.S.Zero

    def gradient(
        self, f: SymengineExpr, key: Optional[str] = None, boolean: bool = False, simplify: bool = False, **kwargs
    ) -> dict[str, list[SymengineExpr]] | list[SymengineExpr]:
        if boolean:
            raise NotImplementedError("Boolean gradients require SympySymbolic")
        if self._wmap is None or not isinstance(self._wmap, dict):
            raise ValueError("Weight map not initialized for gradient computation.")
        pderivs = []
        for var in sorted(self._wmap.keys()):
            w = self._params["sym_weights"][var]
            df_dw = symengine.diff(f, w)
            if df_dw.atoms(symengine.Derivative):
                raise NotImplementedError(
                    "symengine cannot differentiate this expression (e.g., `max`/`min` in "
                    "Viterbi or Fuzzy inference); use symbolic_backend='sympy' instead."
                )
            pderivs.append(df_dw)
        grad_map = {"weights": pderivs}
        if key is not None:
            return grad_map[key]
        return grad_map

    def simplify(self, expr: SymengineExpr) -> SymengineExpr:
        return expr.expand(expr)

    ### logical operations — not supported in symengine
    def logical_and(self, val1, val2):
        raise NotImplementedError("Use SympySymbolic for logical operations")
    def logical_or(self, val1, val2):
        raise NotImplementedError("Use SympySymbolic for logical operations")
    def logical_not(self, val):
        raise NotImplementedError("Use SympySymbolic for logical operations")
    def logical_imply(self, val1, val2):
        raise NotImplementedError("Use SympySymbolic for logical operations")
    def logical_bicond(self, val1, val2):
        raise NotImplementedError("Use SympySymbolic for logical operations")
    def logical_xor(self, val1, val2):
        raise NotImplementedError("Use SympySymbolic for logical operations")
    def logic_true(self):
        raise NotImplementedError("Use SympySymbolic for logical operations")
    def logic_false(self):
        raise NotImplementedError("Use SympySymbolic for logical operations")
    