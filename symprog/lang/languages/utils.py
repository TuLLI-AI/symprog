import ast
from platform import node 

__all__ = [
    "ProgramValidator", 
    "extract_cond", 
    "KEYWORDS", 
    "make_sampler",
    "build_nary_xor"
]

# SAFETY VALIDATOR
###
BLOCKED_BUILTINS = {
    "exec", 
    "eval", 
    "compile", 
    "__import__", 
    "open", 
    "getattr", 
    "setattr", 
    "delattr"
}
ALLOWED_CALLS = {
    "Implies", 
    "And", 
    "Or", 
    "Biconditional", 
    "Xor", 
    "Not", 
    "observe", 
    "evidence",
    "query",
    "Query",
    "Evidence",
    "Observe",
}

class ProgramValidator(ast.NodeVisitor):
    """Validates that the user's raw AST contains only safe constructs.
    
    Makes the following exclusions: 
    - No import statements (Import, ImportFrom)
    - No function definitions (FunctionDef, AsyncFunctionDef)
    - No class definitions (ClassDef)
    - No lambda expressions (Lambda)
    - No attribute access (Attribute)
    - No function calls except for a specific allowlist of known safe functions (Call)
    
    Raises:
        ValueError if any of the above constructs are found in the AST.
    
    Can be customized and subclassed for different language implementations. 
    """

    def visit_Import(self, node):
        raise ValueError("Import statements are not allowed")

    def visit_ImportFrom(self, node):
        raise ValueError("Import statements are not allowed")

    def visit_FunctionDef(self, node):
        if not hasattr(self, '_seen_top_level'):
            self._seen_top_level = True
            self.generic_visit(node)
        else:
            raise ValueError("Nested function definitions are not allowed")

    def visit_AsyncFunctionDef(self, node):
        raise ValueError("Async function definitions are not allowed")

    def visit_ClassDef(self, node):
        raise ValueError("Class definitions are not allowed")

    def visit_Lambda(self, node):
        raise ValueError("Lambda expressions are not allowed")

    def visit_Attribute(self, node):
        raise ValueError("Attribute access is not allowed")
    
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            if node.func.id in BLOCKED_BUILTINS:
                raise ValueError(f"Call to '{node.func.id}' is not allowed")
            # allow known operators and predicate-like calls: s(X), f(a)
            if node.func.id in ALLOWED_CALLS:
                self.generic_visit(node)
                return
            # predicate calls: all args must be plain names
            if all(isinstance(arg, ast.Name) for arg in node.args):
                self.generic_visit(node)
                return
            raise ValueError(f"Unknown function '{node.func.id}'")
        else:
            raise ValueError("Only named function calls are allowed")


######
######
## utilities for parsing boolean expressions

KEYWORDS = {
    "ValueError" : "Exc",
}
def extract_cond(
    mapper,
    *,
    test: ast.BoolOp | ast.Name | ast.UnaryOp | ast.Constant | ast.expr,
    bools: set[str] | None = None,
    relax: bool = False,
    ):
    """Extracts the `if` conditions in an if-then construction (ast) and translates to mapper object.

    Arguments:
        mapper: The particular symbolic mapper being used. 
        test: The node containing the if test code.
        bools: The program boolean variables. 
        relax: Parse formuals/conditions without checking if all variables 
            are defined (used for parsing global `observe` statements). 
    Raises:
        SyntaxError when encodig an unsopported test case.
        ValueError when condition is not a Bool or an unknown variable.

    """
    if isinstance(test, ast.BoolOp):
        cond_formulas = [
            extract_cond(mapper, test=i, bools=bools, relax=relax) for i in test.values
        ]
        if isinstance(test.op, ast.And):
            op = mapper.And
        elif isinstance(test.op, ast.Or):
            op = mapper.Or
        else:
            raise ValueError(f"Unknown Boolean op: {test.op}")
            
        formula = None
        for f in cond_formulas:
            formula = f if formula is None else op(formula, f)
        return formula
    elif isinstance(test, ast.Name):
        if (not relax) and bools and (test.id not in bools):
            raise ValueError(
                f"Unknown variable used in bool condition: {test.id}"
            )
        return mapper.Symbol(KEYWORDS.get(test.id, test.id), return_val=True)
    elif isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return mapper.Not(extract_cond(mapper, test=test.operand, bools=bools, relax=relax)) 
    ## primitive `True` and `False`
    elif isinstance(test, ast.Constant) and (test.value is True):
        return mapper.true
    elif isinstance(test, ast.Constant) and (test.value is False):
        return mapper.false
    else:
        raise SyntaxError(f"Unsupported if-test: {ast.dump(test)}")
    
#######################
#######################
#######################
## ast helpers 

def make_sampler(name, weight, evidence=None):
    sampler = ast.Compare(
        left=ast.Call(
            func=ast.Attribute(
                value=ast.Attribute(
                    value=ast.Name(id='np', ctx=ast.Load()),
                    attr='random', ctx=ast.Load()),
                attr='random', ctx=ast.Load()),
            args=[], keywords=[]),
        ops=[ast.Lt()],
        comparators=[ast.Constant(weight)],
    )
    
    if evidence is not None:
        sampler = ast.IfExp(
            test=ast.Constant(False),
            body=sampler,
            orelse=ast.Constant(evidence),
        )
    
    return ast.Assign(
        targets=[ast.Name(id=name, ctx=ast.Store())],
        value=sampler,
    )

def build_nary_xor(args) -> ast.BoolOp:
    """Builds an ast representation of the exclusive or (XOR) of the given arguments.
    
    Arguments:
        args: A list of ast expressions representing the variables to 
            be XORed.
    Returns: 
        An ast.BoolOp representing the XOR of the given arguments.
    """
    at_least_one = ast.BoolOp(op=ast.Or(), values=list(args))

    at_most_one = []
    for i in range(len(args)):
        for j in range(i + 1, len(args)):
            at_most_one.append(
                ast.BoolOp(
                    op=ast.Or(),
                    values=[
                        ast.UnaryOp(op=ast.Not(), operand=args[i]),
                        ast.UnaryOp(op=ast.Not(), operand=args[j]),
                    ],
                )
            )
    return ast.BoolOp(
        op=ast.And(),
        values=[at_least_one] + at_most_one, # type: ignore 
    )