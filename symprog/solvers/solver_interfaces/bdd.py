import os
import tempfile

try: 
    from dd.autoref import BDD, Function
    import dd.bdd as _bdd
except:
    BDD = None 

from typing import FrozenSet
from ..mapper import SymbolicMapper, MapperRegistry, SATOutput, ModelOutput

__all__ = ["BddMapper"]

@MapperRegistry(name = "bdd")
class BddMapper(SymbolicMapper[Function]):
    """Binary decision diagram (BDD) mapper class for 
    translating formulas into BDDs."""
    
    def _init_symbols(self) -> None:
        """Adds the provided weights to the underlying symbol map"""
        self.bdd = BDD() # type: ignore 
        if self.var_list is not None:
            self.bdd.declare(*self.var_list) # type: ignore 

        for v in self.var_list:
            self.add_symbol(v)
            
    def add_symbol(self, symbol: str | int) -> None:
        """Optionally support adding new variables dynamically.

        Arguments: 
            symbol: The symbol to be added to the BDD. 
        """
        name = str(symbol)
        if name in self._sym_map:
            raise ValueError(f"Symbol {name} already exists in the mapper.")
        self.bdd.declare(name)
        node = self.bdd.var(name)
        self._sym_map[name] = node
        #return node

    def Or(self, obj1: Function, obj2: Function) -> Function:
        """Implementation of logical `Or`.

        Arguments: 
            obj1: The first argument for disjunction. 
            obj2: The second argument for disjunction
        """
        return self.bdd.apply('or', obj1, obj2)

    def And(self, obj1: Function, obj2: Function) -> Function:
        """Implementation of logical `And`.

        Arguments: 
            obj1: The first argument for conjunction. 
            obj2: The second argument for conjunction. 
        """
        return self.bdd.apply('and', obj1, obj2)

    def Implies(self, obj1: Function, obj2: Function) -> Function:
        """Implementation of logical material implication. Uses 
        equivalence between `or(not(obj1), obj2)`. 

        Arguments: 
            obj1: The first argument for implication. 
            obj2: The second argument for implication. 
        """
        not_a = self.bdd.apply('not', obj1)
        return self.bdd.apply('or', not_a, obj2)
    
    def Biconditional(self, obj1: Function, obj2: Function) -> Function:
        """Implementation of the biconditional.
        
        Arguments: 
            obj1: The first argument for implication. 
            obj2: The second argument for implication.
        """
        return self.And(self.Implies(obj1, obj2),self.Implies(obj2, obj1))
    

    def Not(self, expr: Function) -> Function:
        """Implementation of logical negation.

        Arguments: 
            expr: The expression to negate. 
        """
        return self.bdd.apply('not', expr)

    def Symbol(self, name: str, return_val: bool = False):
        """Representation and creation on primitive symbols.

        Arguments: 
            name: The name of the symbol. 
        """
        if name not in self._sym_map:
            raise ValueError(f"Unknown BDD variable: {name}")
        return self._sym_map[name] if return_val else name

    @property
    def false(self) -> Function:
        """Internal representation of the false primitive."""
        return self.bdd.false

    @property
    def true(self) -> Function:
        """Internal representation of the true primitive."""
        return self.bdd.true

    def _to_dot(self, formula) -> str:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'formula.dot')
            self.bdd.dump(path, roots=[formula])
            with open(path, 'r') as f:
                dot_str = f.read()
        return dot_str

    def _model_enumeration(self, formula, **kwargs) -> ModelOutput:
        for model in self.bdd.pick_iter(formula):
            yield model # type: ignore 

    def _sat_interface(self, formula: Function, **kwargs) -> tuple[str, None]:
        """Interface for running SAT solving. Given that SDDs are `canonical`, we 
        can check for satisfiability be seeing if the formula is equal to `sdd.false` (unsat)
        or not. 

        Arguments: 
            formula: The formula to check for satisfiability. 
        """
        sat = "yes" if formula != self.bdd.false else "no"
        details = None 
        return (sat, details)

    def _count_interface(self, formula: Function, **kwargs) -> tuple[int, None]:
        """Interface for running SAT solving. Uses `bdd.count`.  

        Arguments: 
            formula: The formula to check for satisfiability. 
        """
        count = self.bdd.count(formula)
        return (count, None)
    
    def wmc_interface(self, formula: Function, semiring, **kwargs):
        """Algebraic WMC over a BDD using a generic semiring.

        Each BDD node u with variable x is treated as:
            f_u = (x ∧ f_high) ∨ (~x ∧ f_low)

        and evaluated as:
            Val(u) = ⊕_{b ∈ {0,1}} [ w(x=b) ⊗ Val(child_b) ⊗ Smoothing(child_b) ]

        where Smoothing(child_b) multiplies in ∑_b w(v=b) for variables v that are
        in the scope below u but absent from that child.

        Args:
            formula: Root BDD function (dd.autoref.Function).
            semiring: Semiring with methods:
                - sum(a, b)
                - product(a, b)
                - assignment(var_name, 0|1)
                - true, false, zero
                - requires_smoothing: bool
            smoothing (kwarg): Force smoothing even if semiring.requires_smoothing=False.
            smooth_to_root (kwarg): Additionally smooth to a global variable set.
            all_vars (kwarg): Iterable of variable names for global smoothing
                              when smooth_to_root=True. Defaults to mapper.var_list
                              or BDD vars.

        Returns:
            The semiring value of the WMC of `formula`.
        """
        raise NotImplementedError("WMC over BDDs not yet implemented.")
    

        # bdd = self.bdd
        # root = formula.node
        # succ = bdd.succ            # node -> (var, low, high)
        # true_node = bdd.true.node
        # false_node = bdd.false.node

        # smoothing = semiring.requires_smoothing or kwargs.get("smoothing", False)
        # smooth_to_root = kwargs.get("smooth_to_root", False)

        # # --------------------------------------------------------------
        # # 1. Compute support (variable set) for every BDD node.
        # # --------------------------------------------------------------
        # support_cache: dict[int, FrozenSet[str]] = {}

        # def compute_support(u: int) -> FrozenSet[str]:
        #     if u in support_cache:
        #         return support_cache[u]

        #     if u == true_node or u == false_node:
        #         s: FrozenSet[str] = frozenset()
        #     else:
        #         var, low, high = succ[u]
        #         vname = str(var)
        #         s_low = compute_support(low)
        #         s_high = compute_support(high)
        #         s = frozenset({vname}) | s_low | s_high

        #     support_cache[u] = s
        #     return s

        # compute_support(root)

        # # --------------------------------------------------------------
        # # 2. Bottom‑up semiring evaluation with local smoothing.
        # # --------------------------------------------------------------
        # memo: dict[int, any] = {}
        # def wmc_node(u: int) -> any:
        #     if u in memo:
        #         return memo[u]

        #     if u == false_node:
        #         val = semiring.false

        #     elif u == true_node:
        #         # Constant-true function over the empty support here; any
        #         # remaining variables below are handled by ancestors.
        #         val = semiring.true

        #     else:
        #         var, low, high = succ[u]
        #         vname = str(var)

        #         supp_u = support_cache[u]
        #         supp_low = support_cache[low]
        #         supp_high = support_cache[high]

        #         # Variables that can still appear below u, excluding vname.
        #         expected_below = supp_u - {vname}

        #         # Weights for current decision variable
        #         w0 = semiring.assignment(vname, 0)
        #         w1 = semiring.assignment(vname, 1)

        #         # Recursively compute children
        #         val_low = wmc_node(low)
        #         val_high = wmc_node(high)

        #         # Base branch weights
        #         branch_low = semiring.product(w0, val_low)
        #         branch_high = semiring.product(w1, val_high)

        #         if smoothing:
        #             # Variables that are in the scope below u but missing in each child
        #             missing_low = expected_below - supp_low
        #             missing_high = expected_below - supp_high

        #             for v in missing_low:
        #                 wv0 = semiring.assignment(v, 0)
        #                 wv1 = semiring.assignment(v, 1)
        #                 combined = semiring.sum(wv0, wv1)
        #                 branch_low = semiring.product(branch_low, combined)

        #             for v in missing_high:
        #                 wv0 = semiring.assignment(v, 0)
        #                 wv1 = semiring.assignment(v, 1)
        #                 combined = semiring.sum(wv0, wv1)
        #                 branch_high = semiring.product(branch_high, combined)

        #         # OR (sum) of the two branches
        #         val = semiring.sum(branch_low, branch_high)

        #     memo[u] = val
        #     return val
        
        # results = wmc_node(root)

        # # --------------------------------------------------------------
        # # 3. Optional global smoothing to a larger variable set.
        # # --------------------------------------------------------------
        # if smooth_to_root:
        #     all_vars = kwargs.get("all_vars", None)
        #     if all_vars is None:
        #         # Prefer mapper.var_list if you have it; otherwise BDD's vars
        #         if getattr(self, "var_list", None):
        #             all_vars = [str(v) for v in self.var_list]
        #         else:
        #             # bdd.vars is {var_name: level}
        #             all_vars = list(str(v) for v in getattr(bdd, "vars", {}).keys())

        #     root_support = support_cache[root]
        #     extra_vars = {str(v) for v in all_vars} - root_support

        #     for v in extra_vars:
        #         w0 = semiring.assignment(v, 0)
        #         w1 = semiring.assignment(v, 1)
        #         combined = semiring.sum(w0, w1)
        #         result = semiring.product(result, combined)

        # return result


        
        
    
## works for probability weights only!
# def wmc_bdd(bdd_mgr, root, var_probs):
#     """
#     Weighted model counting on a BDD.
#     bdd_mgr: dd.autoref.BDD instance
#     root: node id (BDD root)
#     var_probs: dict mapping variable name -> P(var=True)
#     Assumes variables are independent; literals get:
#       w(x) = P(x=True), w(~x) = 1 - P(x=True)
#     """
#     cache = {}
#     def eval_node(u):
#         if u in cache:
#             return cache[u]
#         if u == bdd_mgr.false:
#             val = 0.0
#         elif u == bdd_mgr.true:
#             val = 1.0
#         else:
#             vname = bdd_mgr.var(u)         # variable name (str)
#             p = var_probs[vname]          # P(vname = True)
#             low = bdd_mgr.low(u)
#             high = bdd_mgr.high(u)
#             val_low = eval_node(low)
#             val_high = eval_node(high)
#             val = (1 - p) * val_low + p * val_high
#         cache[u] = val
#         return val
#     return eval_node(root)    
