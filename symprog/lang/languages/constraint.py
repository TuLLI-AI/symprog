import ast
import itertools
import copy
from platform import node
from unittest.mock import call
import numpy as np
from typing import Any, Callable
from types import ModuleType

from .utils import (
    ProgramValidator, 
    extract_cond, 
    make_sampler,
    build_nary_xor,
) 
from ..language import ASTModelingLanguage
from ..language import LanguageRegistry
from ...solvers import Solver

__all__ = ["ConstraintLanguage"]

class GroundingTransformer(ast.NodeTransformer): 
    """A Node transformer for handling the grounding of the constraint program 
    (in the case of first-order constraints). 
    
    For example, it can transform the following input program:
    
    ```
    a: X; b: Y
    Implies(X, Y)
    ```
    to the following output program:
    
    ```
    Implies(a__a, b__a)
    Implies(a__b, b__b)
    ```
    
    where the type variables X and Y are grounded to their respective variable instances.

    Attributes: 
        domains: 
            A dictionary mapping type names to lists of variable names in that type. 
        var_types: 
            A dictionary mapping variable names to their type names. 
        predicate_weights: 
            A dictionary mapping predicate names to their weight tables, where 
            each weight table is a dictionary mapping variable names to weights. 
        function_weights: 
            A dictionary mapping function names to their weight tables, where 
            each weight table is a dictionary mapping variable names to weights. 
        all_predicates: 
            A set of all predicate names encountered in the program, used for 
            merging predicate and function weight tables.
    
    """
    def __init__(self) -> None:
        self.domains = {}
        self.var_types = {}
        self.predicate_weights = {}
        self.function_weights = {}
        self.all_predicates = set()
        self._emitted_samplers = set()

    def _merged_weights(self) -> dict[str, dict[str, float]]:
        """Merges the predicate and function weight tables, with function weights taking 
        precedence over predicate weights.
        
        Returns: 
            A dictionary mapping predicate names to their merged weight tables.
        """
        merged = {}
        for pred in self.all_predicates:
            table = {}
            table.update(self.predicate_weights.get(pred, {}))
            table.update(self.function_weights.get(pred, {}))
            merged[pred] = table
            
        return merged
    
    def ground_variable_weights(self) -> dict[str, float]:
        """Produces a flat dict of ground variable weights. Defaults to 0.5 
        (uniform model) for any variable without a specified weight. 
        
        Returns: 
            A dictionary mapping ground variable names to their weights.
        """
        merged = self._merged_weights()
        weights = {}
        for pred in self.all_predicates:
            table = merged.get(pred, {})
            for type_name in self.domains:
                for var in self.domains[type_name]:
                    weights[f"{pred}__{var}"] = table.get(var, 0.5)
        return weights
    
    def visit_AnnAssign(self, node):
        assert isinstance(node.target, ast.Name)
        name = node.target.id

        # handle condition: before anything else
        if name == "condition":
            ann = node.annotation
            type_vars = self._find_type_vars(ann)
            if not type_vars: 
                node.annotation = _GroundSubstituter({}, self).visit(
                    copy.deepcopy(ann)
                )
                return node
            
            ## first-order case 
            nodes = []
            for substitution in self._enumerate_substitutions(type_vars):
                grounded = _GroundSubstituter(substitution, self).visit(
                    copy.deepcopy(ann)
                )
                nodes.append(ast.AnnAssign(
                    target=ast.Name(id="condition", ctx=ast.Store()),
                    annotation=grounded,
                    value=None,
                    simple=1,
                ))
            return nodes
            
        ann = node.annotation

        # a: X  →  register variable in domain
        if isinstance(ann, ast.Name) and ann.id[0].isupper():
            type_name = ann.id
            self.var_types[name] = type_name
            self.domains.setdefault(type_name, []).append(name)
            return None

        # s: {a: 0.9, b: 0.8}  →  predicate weight table
        if isinstance(ann, ast.Dict):
            weights = {}
            for key, val in zip(ann.keys, ann.values):
                assert isinstance(key, ast.Name)
                assert isinstance(val, ast.Constant)
                weights[key.id] = val.value
            self.predicate_weights[name] = weights
            self._emitted_samplers.update(f"{name}__{var}" for var in weights)
            return [
               make_sampler(f"{name}__{var}", w)
                for var, w in weights.items()
            ]
        
        return node
    
    def visit_Assign(self, node):
        target = node.targets[0]
        if (isinstance(target, ast.Call)
                and isinstance(target.func, ast.Name)
                and target.func.id == "w"):
            args = target.args
            assert len(args) == 2
            assert isinstance(args[0], ast.Name)
            assert isinstance(args[1], ast.Name)
            assert isinstance(node.value, ast.Constant)

            pred = args[0].id
            var = args[1].id
            weight = node.value.value

            self.function_weights.setdefault(pred, {})[var] = weight
            self._emitted_samplers.add(f"{pred}__{var}")

            return make_sampler(f"{pred}__{var}", weight)

        return node

    def _is_aggregate(self, node: ast.AST) -> bool:
        """Single-arg call to Xor/And/Or where the arg contains type variables.
        
        This rule allows one to write `And(s(X))` instead of having to manually 
        ground to And(s__a, s__b, ...) for all variables in the domain of X. This 
        is particular useful in the context of exclusive-or constraints `Xor(s(X))`, 
        which are common in constraint problems but can be tedious to ground manually.
        
        """
        if not isinstance(node, ast.Call):
            return False
        if not isinstance(node.func, ast.Name):
            return False
        if node.func.id not in ("Xor", "And", "Or"):
            return False
        if len(node.args) != 1:
            return False
        return bool(self._find_type_vars(node.args[0]))
    
    def _ground_aggregate(self, node):
        """Xor(s(X)) → Xor(s__a, s__b, ...) over domain of X."""
        call = node.value
        inner = call.args[0]
        type_vars = self._find_type_vars(inner)

        ground_args = []
        for substitution in self._enumerate_substitutions(type_vars):
            grounded = _GroundSubstituter(substitution, self).visit(
                copy.deepcopy(inner)
            )
            ground_args.append(grounded)

        # emit samplers for any new ground variables
        stmts = []
        for arg in ground_args:
            for child in ast.walk(arg):
                if isinstance(child, ast.Name) and "__" in child.id:
                    if child.id not in self._emitted_samplers:
                        weight = self._get_weight(child.id)
                        stmts.append(make_sampler(child.id, weight))
                        self._emitted_samplers.add(child.id)

        # rebuild the call with all ground args
        new_call = ast.Call(
            func=call.func,
            args=ground_args,
            keywords=[],
        )
        stmts.append(ast.Expr(value=new_call))
        return stmts


    def _get_weight(self, ground_name: str) -> float:
        """Look up weight for a ground variable like s__a, defaults to 0.5 when
        the variable is not found. 
        
        Arguments: 
            ground_name: The name of the ground variable, in the format "predicate__variable".
        """
        parts = ground_name.split("__", 1)
        if len(parts) == 2:
            pred, var = parts
            merged = {**self.predicate_weights.get(pred, {}),
                      **self.function_weights.get(pred, {})}
            return merged.get(var, 0.5)
        return 0.5
    
    def visit_Expr(self, node):
        if not isinstance(node.value, (ast.BoolOp, ast.UnaryOp, ast.Call)):
            return node
        
        ## detect aggregates like And(X) and ground them
        if self._is_aggregate(node.value):
            return self._ground_aggregate(node)

        type_vars = self._find_type_vars(node.value)
        if not type_vars:
            return node

        ground_stmts = []
        for substitution in self._enumerate_substitutions(type_vars):
            grounded = _GroundSubstituter(substitution, self).visit(
                copy.deepcopy(node.value)
            )

            ## find varibles without samplers 
            for child in ast.walk(grounded):
                if isinstance(child, ast.Name) and "__" in child.id:
                    if child.id not in self._emitted_samplers:
                        weight = self._get_weight(child.id)
                        ground_stmts.append(make_sampler(child.id, weight))
                        self._emitted_samplers.add(child.id)
            ground_stmts.append(ast.Expr(value=grounded))

        return ground_stmts
    
    def _find_type_vars(self, node):
        found = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                for arg in child.args:
                    if isinstance(arg, ast.Name) and arg.id in self.domains:
                        found.add(arg.id)
        return found

    def _enumerate_substitutions(self, type_vars):
        vars_list = sorted(type_vars)
        domain_lists = [self.domains[v] for v in vars_list]
        for combo in itertools.product(*domain_lists):
            yield dict(zip(vars_list, combo))

    def visit_Name(self, node):
        if "__" in node.id:
            raise ValueError(
                f"Variable name '{node.id}' contains '__', which is reserved "
                "for ground predicate notation"
            )
        return node
    
class _GroundSubstituter(ast.NodeTransformer):
    """Replaces predicate calls like s(X) with ground variable names like s__a."""

    def __init__(self, substitution, grounding):
        self.substitution = substitution
        self.grounding = grounding

    def visit_Call(self, node):
        self.generic_visit(node)

        if not isinstance(node.func, ast.Name):
            return node

        pred = node.func.id
        grounded_args = []
        has_substitution = False

        for arg in node.args:
            if not isinstance(arg, ast.Name):
                return node
            if arg.id in self.substitution:
                grounded_args.append(self.substitution[arg.id])
                has_substitution = True
            elif arg.id in self.grounding.var_types:
                grounded_args.append(arg.id)
                has_substitution = True
            else:
                grounded_args.append(arg.id)

        if not has_substitution:
            return node

        self.grounding.all_predicates.add(pred)
        ground_name = f"{pred}__{'__'.join(grounded_args)}"
        return ast.Name(id=ground_name, ctx=ast.Load())


class ConstraintTransformer(ast.NodeTransformer):
    """A transformer that transforms the AST of a constraint program to a form that 
    can be executed as native Python. 

    For example, it can transform the following input program: 

        ```
        a: 0.9; b: 0.7
        (not a) or b
        a or b 
        ```

    to the following output program: 

        ```python 
            a = np.random.choice([True, False], p=[0.9, 0.1])
            b = np.random.choice([True, False], p=[0.7, 0.3])
            r1 = (not a) or b
            r2 = a or b
    
            if r1 and r2:
                return "True"
            else:
                return "False"
        ```
 
    """
    def __init__(self, variable_weights: dict[str, float] | None = None): 
        self.variable_weights = variable_weights if variable_weights is not None else {}
        self.constraint_counter = 0
        self.constraint_names = []

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        """Transforms variable weight annotations to sampling statements. 

        For example, it can transform the following input program: 
            ```
            a: 0.9; b: 0.7
            ```
        to the following output program: 
            ```
            a = np.random.choice([True, False], p=[0.9, 0.1])
            b = np.random.choice([True, False], p=[0.7, 0.3])
            ```
        """
        if isinstance(node.target, ast.Name) and node.target.id == "condition":
            return ast.Assert(test=node.annotation)
    
        ### store the weight 
        assert isinstance(node.target, ast.Name)
        assert isinstance(node.annotation, ast.Constant)
        
        name = node.target.id
        weight = node.annotation.value          # raw float
        if not isinstance(weight, float):
            raise ValueError(f"Variable weight must be a float, got {type(weight)}")
        self.variable_weights[name] = weight
        

        assert isinstance(weight, float)
        return make_sampler(name, weight)
    
    def visit_Call(self, node: ast.Call):
        """Transforms boolean formulas in operator notation to native Python boolean expressions.
        
        e.g., it can transform the following input program:

            ```
            Implies(a, b)
            And(a, b)
            Or(a, b)
            Not(a)
            Biconditional(a, b)
            Xor(a, b)
            ```
        
        to the following output program:
        
            ```
            (not a) or b
            a and b
            a or b
            not a
            (a and b) or (not a and not b)
            (a and not b) or (not a and b)
            ```

        """
        self.generic_visit(node) 
        if not isinstance(node.func, ast.Name):
            return node
        
        args = node.args 
        match node.func.id: 
            case "Implies": 
                return ast.BoolOp(
                    op=ast.Or(),
                    values=[
                        ast.UnaryOp(op=ast.Not(), operand=args[0]), 
                        args[1]
                    ],
                )
            case "And":
                if len(args) == 1:
                    return args[0]
                return ast.BoolOp(op=ast.And(), values=args)
            case "Or":
                if len(args) == 1:
                    return args[0]
                return ast.BoolOp(op=ast.Or(), values=args)
            case "Not":
                assert len(args) == 1, "Wrong number of negation input"
                return ast.UnaryOp(
                    op=ast.Not(), operand=args[0],
                )
            case "Biconditional":
                assert len(args) == 2
                return ast.BoolOp(
                    op=ast.Or(),
                    values=[
                        ast.BoolOp(
                            op=ast.And(),
                            values=[args[0], args[1]],
                        ),
                        ast.BoolOp(
                            op=ast.And(),
                            values=[
                                ast.UnaryOp(op=ast.Not(), operand=args[0]), 
                                ast.UnaryOp(op=ast.Not(), operand=args[1])
                            ],
                        ),
                    ],
                )
            case "Xor":
                if len(args) == 1:
                    return args[0]
                return build_nary_xor(args)
            case _:
                return node

    def visit_Expr(self, node: ast.Expr):
        """Transforms boolean formulas to statements that compute the value of the formula. 

        For example, it can transform the following input program: 

            ```
            (not a) or b
            a or b 
            condition: a or b
            ```

        to the following output program: 

            ```
            r1 = (not a) or b
            r2 = a or b
            assert a or b
            ```

        """
        node.value = self.visit(node.value)

        ## ignore non boolean things
        if not isinstance(node.value, (ast.BoolOp, ast.UnaryOp, ast.Name)):
            return node
        
        self.constraint_counter += 1
        name = f"r{self.constraint_counter}"
        self.constraint_names.append(name)
        return ast.Assign(
            targets=[ast.Name(id=name, ctx=ast.Store())],
            value=node.value,
        )


class ConstraintSemantics(ast.NodeVisitor):
    """A semantics that executes the transformed AST of a constraint program. 
    """
    def __init__(self, mapper): 
        self.mapper = mapper
        self._formulas = []
        self._conditionals = [] 

    @property 
    def formulas(self) -> list[Any]:
        """Returns the list of formulas extracted from the program. 
        """
        return self._formulas
    
    @property 
    def conditionals(self) -> list[Any]:
        """Returns the list of conditionals extracted from the program. 
        """
        return self._conditionals

    def visit_Assign(self, node):
        # r1 = not a or b  →  extract the BoolOp
        # skip sampling assignments (np.random.choice calls)
        if isinstance(node.value, (ast.BoolOp, ast.UnaryOp, ast.Name)):
            formula = extract_cond(
                self.mapper, test=node.value,
            )
            self._formulas.append(formula)

    def visit_Assert(self, node):
        # assert not a or not b  →  conditioning
        formula = extract_cond(
            self.mapper, test=node.test,
        )
        self._conditionals.append(formula)

    def visit_FunctionDef(self, node):
        # descend into the function body
        self.generic_visit(node)

@LanguageRegistry(name="constraint")
class ConstraintLanguage(ASTModelingLanguage):
    """A modeling language for specifying Boolean constraints.

    Below is an example inline program. 

    ```python
        @Parser(language="constraint")
        def my_program():
            a: 0.9; b: 0.7
            (not a) or b
            a or b 
            condition: a or b
    ```
    where: 
        - variable weights are specified on the top as type annotations
        - constraints are specified as boolean formulas in the middle
        - conditioning statements are specified at the end with the "condition:" 
        prefix.
    
    """
    def __init__(self, **solver_kwargs) -> None:
        super().__init__(**solver_kwargs) 
        self._variable_weights = {}

    @property 
    def glbs(self) -> dict[str, ModuleType | Callable]:
        """Global variables to be injected into the program during execution
        
        Returns: 
            A dictionary of string to module or callable mappings to be injected 
            into the program during execution. For example, in the constraint language, 
            we inject the numpy module as "np" to allow users to sample from distributions.
        """
        return {"np" : np}

    @property 
    def variable_weights(self) -> dict[str, float]:
        """Return dictionary containing variables and their weights. This is used by the solver 
        to compute the probability of each variable assignment.
        
        Returns:
            A dictionary mapping variable names to their weights.  
        """
        return self._variable_weights
    
    @property 
    def variables(self) -> set[str]:
        """Returns the set of variable names in the program. This is used by the solver to 
        compute the probability of each variable assignment.
        
        Returns:
            A set of variable names in the program.  
        """
        return set(self._variable_weights.keys())

    def parse(self, tree: ast.AST) -> ast.AST:
        """Transforms the raw AST of the input program to a Python AST that can be executed. 
        Serves as the foundation of the transformation pipeline for the constraint language.
        
        Arguments: 
            tree: The raw AST of the input program.
        Returns:
            The transformed AST of the input program.
        """
        ## check for unsafe or unsupported syntax 
        ProgramValidator().visit(tree)

        ## ground the program 
        grounder = GroundingTransformer()
        tree = grounder.visit(tree)
        for name, weight in grounder.ground_variable_weights().items():
            self._variable_weights[name] = weight

        ## transform the program to executable Python
        transformer = ConstraintTransformer(self._variable_weights)
        tree = transformer.visit(tree)

        # find the function body
        func_def = tree.body[0]  # type: ignore

        # append if-then to the function body, not the module
        if len(transformer.constraint_names) == 0:
            func_def.body.append(ast.Return(value=ast.Constant("True")))
            ast.fix_missing_locations(tree)
            return tree
        elif len(transformer.constraint_names) == 1:
            test = ast.Name(id=transformer.constraint_names[0], ctx=ast.Load())
        else:
            test = ast.BoolOp(
                op=ast.And(),
                values=[ast.Name(id=n, ctx=ast.Load()) for n in transformer.constraint_names],
            )

        func_def.body.append(ast.If(
            test=test,
            body=[ast.Return(value=ast.Constant("True"))],
            orelse=[ast.Return(value=ast.Constant("False"))],
        ))

        ast.fix_missing_locations(tree)
        return tree
    
    def semantics(self, program) -> Solver:
        """Executes the transformed AST of the input program and returns the result. 
        
        Arguments: 
            program: 
                The raw python native program to extra semantics from and execute. 
                For the constraint language, this will be a string containing the raw program.
        Returns:
            The result of executing the input program. For the constraint language, 
            this will be a boolean value indicating whether the constraints are satisfied.
        
        """
        var_weights = self.variable_weights
        vars = [str(v) for v in var_weights.keys()]

        ## build the solver and mapper 
        solver = Solver.from_kwargs(
            var_list=vars, # type: ignore 
            var_weights=var_weights, # type: ignore 
            **self.solver_kwargs, # type: ignore
        )

        ## build the core formula using solver mapper 
        mapper = solver.solver 
        semantics = ConstraintSemantics(mapper)

        tree = ast.parse(program)
        semantics.visit(tree)
        solver.add(semantics.formulas)

        if semantics.conditionals:
            if len(semantics.conditionals) > 1:
                raise ValueError(
                    "Multiple conditionals not supported yet, manually conjoin."
                )
            solver.add_conditional(semantics.conditionals[0])

        return solver 
    
        