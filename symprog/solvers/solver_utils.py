### utilities for helping with solver implementations.
import ast
import torch.nn as nn
import torch.fx as fx
from typing import Any, Callable, Mapping, Sequence, Literal, Generator, Any
from dataclasses import dataclass

## helpers for solver outputs

SATOutput = Literal["sat", "unsat", "unknown"]
ModelOutput = Generator[dict[str | int, bool], None, None]

@dataclass
class Formula:
    """Helper class for representing formulas

    Attributes: 
        formula: The underlying formula is whatever format is given by 
            the solver class. 
        dot: Graphical representaiton of the formula (if provided in solver class)
            using dot and graphviz. 
    """
    formula: Any
    dot: str | None = None
    rep: str | None = None

@dataclass
class SolverOut:
    """Helper class for representing solver output. 
    
    Attributes:
        inference: The main inference result from the solver.
        inference_type: A string describing the type of inference performed.
        normalization: Optional normalization factor (if applicable).
        details: Optional additional details about the solver run.
    """
    inference: Any
    inference_type: str
    variables: dict[str| int, bool] | None = None
    details: dict[Any, Any] | str | None = None

    def __repr__(self) -> str:
        if self.variables is not None: 
            return f"SolverOut(inference_type={self.inference_type}, inference={self.inference}, variables={self.variables})"
        return f"SolverOut(inference_type={self.inference_type}, inference={self.inference})"

### formula parsing utilities 

ALLOWED_FUNCS = {"And", "Or", "Not", "Implies", "Biconditional", "Xor"}

class _FormulaVarCollector(ast.NodeVisitor):
    """AST visitor to collect variable names from a formula."""

    def __init__(self) -> None:
        self.vars: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in ALLOWED_FUNCS or node.id.lower() in {"true", "false"}:
            return
        self.vars.add(node.id)

def extract_vars(formulas) -> set[str]:
    """Extract variable names from a formula string.
    
    Arguments: 
        formulas: An input Boolean formula as a string or as a list of strings.
    Returns:
        A set of variable names used in the formula.
    """
    formula_list = [formulas] if isinstance(formulas, str) else formulas
    var_set = set()
    for formula in formula_list:
        tree = ast.parse(formula, mode="eval")
        collector = _FormulaVarCollector()
        collector.visit(tree)
        var_set.update(collector.vars)

    return var_set

def _build_from_ast(node: ast.AST, mapper: Any) -> object:
    """Recursively build a formula object from an AST node.
    
    Arguments: 
        node: An AST node representing part of the formula.
        mapper: A SymbolicMapper to convert variable names to solver-specific symbols.
    """
    if isinstance(node, ast.Expression):
        return _build_from_ast(node.body, mapper)

    if isinstance(node, ast.Name):
        if node.id == "True":
            return mapper.true
        if node.id == "False":
            return mapper.false
        # Logical constructors are handled as Call.func, not here.
        return mapper.Symbol(node.id, return_val=True)
    
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fname = node.func.id
        if fname not in ALLOWED_FUNCS:
            raise ValueError(f"Unknown logical operator: {fname}")

        args = [_build_from_ast(arg, mapper) for arg in node.args]

        if fname == "Not":
            if len(args) != 1:
                raise ValueError("Not() takes exactly one argument")
            return mapper.Not(args[0])
        if fname in {"And", "Or"}:
            if len(args) < 2:
                raise ValueError(f"{fname}() takes at least two arguments")
            out = args[0]
            for arg in args[1:]:
                out = getattr(mapper, fname)(out, arg)
            return out
        if fname == "Xor":
            ## exactly one argument is true (as in the constraint language)
            if len(args) < 2:
                raise ValueError("Xor() takes at least two arguments")
            if len(args) == 2:
                return mapper.Not(mapper.Biconditional(args[0], args[1]))
            out = None
            for i, arg in enumerate(args):
                term = arg
                for j, other in enumerate(args):
                    if j != i:
                        term = mapper.And(term, mapper.Not(other))
                out = term if out is None else mapper.Or(out, term)
            return out
        if fname in {"Implies", "Biconditional"}:
            if len(args) != 2:
                raise ValueError(f"{fname}() takes exactly two arguments")
            return getattr(mapper, fname)(args[0], args[1])

    ## operator syntax: `~a`, `a & b`, `a | b`
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
        return mapper.Not(_build_from_ast(node.operand, mapper))
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitAnd, ast.BitOr)):
        left = _build_from_ast(node.left, mapper)
        right = _build_from_ast(node.right, mapper)
        if isinstance(node.op, ast.BitAnd):
            return mapper.And(left, right)
        return mapper.Or(left, right)

    raise ValueError(f"Unsupported AST node: {ast.dump(node)}")

def build_formula(formulas, mapper):
    """Build a Formula object from a formula string.
    
    Arguments: 
        formulas: An input Boolean formula as a string or as a list of strings.
    """
    formula_list = [formulas] if isinstance(formulas, str) else formulas
    out_formulas = []
    for formula in formula_list:
        tree = ast.parse(formula, mode="eval")
        new_formula = _build_from_ast(tree, mapper)
        out_formulas.append(new_formula)

    return out_formulas

def build_from_cnf(
        stype: str,
        formulas: list[str] | str, 
        variables: list[str | int], 
        tactic: tuple[str, ...] = ('tseitin-cnf',),  
    ):
    """Build a CNF representation of the formula using the provided mapper. Uses 
    z3/SAT to convert the formula to CNF.
    
    Arguments: 
        stype: 
            The type of solver to build. 
        formulas: 
             An input Boolean formula as a string or as a list of strings.
        variables: 
            A SymbolicMapper to convert variable names to solver-specific symbols.
    """
    from .mapper import MapperRegistry

    ### SAT preprocessor
    sat_mapper = MapperRegistry.build_instance("z3", var_list=variables)
    ## get formulas
    if isinstance(formulas, str) or (isinstance(formulas, list) and all(isinstance(f, str) for f in formulas)): 
        formulas = build_formula(formulas, sat_mapper)

    ###  get the dimacs representation 
    dimacs = sat_mapper.write_cnf(formulas, tactic=tactic)  # type: ignore 

    ### create target solver 
    solver = MapperRegistry.build_instance(stype, var_list=variables)
    formula, aux_vars = solver.load_dimacs(dimacs)  # type: ignore
    return (solver, formula, aux_vars)