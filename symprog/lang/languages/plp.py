import ast
import copy
import itertools
import numpy as np
from typing import Any, Callable
from types import ModuleType
from platform import node
from symprog.lang.language import ASTModelingLanguage, LanguageRegistry
from symprog.lang.languages.utils import ProgramValidator
from symprog.solvers.solver import Solver
from .utils import make_sampler, extract_cond


def _eval_weight(node):
    """Like _eval_numeric but also maps True → 1.0 and False → 0.0.

    Used when parsing fact-annotation weights where boolean literals are
    valid shorthand for certainty/impossibility.  Not used for head-subscript
    parsing where ``False`` signals an inhibitory rule rather than a weight.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return float(node.value)
    return _eval_numeric(node)


def _eval_numeric(node):
    """Evaluate a constant arithmetic expression to a float, or return None.

    Handles plain numeric literals and compound expressions like ``1/9``, ``3*0.1``.
    Returns None for non-numeric nodes (e.g. Name, BoolOp) and for boolean
    constants (False / True) so they are not mistaken for probability weights.
    """
    if isinstance(node, ast.Constant):
        v = node.value
        if isinstance(v, bool):
            return None  # False/True are not numeric weights
        if isinstance(v, (int, float)):
            return float(v) # type: ignore 
        return None
    if isinstance(node, ast.BinOp):
        left = _eval_numeric(node.left)
        right = _eval_numeric(node.right)
        if left is None or right is None:
            return None
        try:
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.FloorDiv):
                return float(int(left // right))
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Pow):
                return float(left ** right)
        except (ZeroDivisionError, ArithmeticError):
            return None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        val = _eval_numeric(node.operand)
        return -val if val is not None else None
    return None


class PLPGroundingTransformer(ast.NodeTransformer):
    def __init__(self):
        self.predicate_domains = {}
        self.constants = set()
        self._base_predicates = set()
        self._ground_atoms = set()
        self._derivable_names: set | None = None  # set once forward-chaining runs

    def _is_variable(self, name):
        return isinstance(name, str) and name[0].isupper()

    def _parse_subscript(self, node):
        if not isinstance(node, ast.Subscript):
            return None, []
        if not isinstance(node.value, ast.Name):
            return None, []

        pred = node.value.id
        slc = node.slice

        if isinstance(slc, ast.Name):
            return pred, [slc.id]
        if isinstance(slc, ast.Tuple):
            args = []
            for e in slc.elts:
                if isinstance(e, ast.Name):
                    args.append(e.id)
                elif isinstance(e, ast.Constant):
                    args.append(e.value)
                else:
                    v = _eval_numeric(e)
                    args.append(v if v is not None else e)
            return pred, args
        if isinstance(slc, ast.Constant):
            return pred, [slc.value]
        # Numeric expression used as a sole weight (e.g. pred[1/3])
        v = _eval_numeric(slc)
        if v is not None:
            return pred, [v]
        return pred, []

    def _is_first_order(self, args):
        return any(self._is_variable(a) for a in args)

    def _register_ground_fact(self, pred, args):
        if pred not in self.predicate_domains:
            self.predicate_domains[pred] = {}
        self._base_predicates.add(pred)
        self._ground_atoms.add((pred, tuple(str(a) for a in args)))
        for i, arg in enumerate(args):
            if not self._is_variable(arg):
                self.predicate_domains[pred].setdefault(i, set()).add(arg)
                self.constants.add(arg)

    def _get_variable_domains(self, body_node):
        domains = {}
        for child in ast.walk(body_node):
            if isinstance(child, ast.Subscript) and isinstance(child.value, ast.Name):
                pred, args = self._parse_subscript(child)
                if pred is None:
                    continue
                for i, arg in enumerate(args):
                    if self._is_variable(arg):
                        pred_domain = self.predicate_domains.get(pred, {}).get(i, set())
                        if arg in domains:
                            domains[arg] = domains[arg] & pred_domain
                        else:
                            domains[arg] = set(pred_domain)
        return domains

    def _ground_name(self, pred, args):
        return f"{pred}__{'__'.join(str(a) for a in args)}"

    def _substitute_node(self, node, substitution):
        return _PLPSubstituter(substitution, self).visit(copy.deepcopy(node))

    def _enumerate_substitutions(self, var_domains):
        vars_list = sorted(var_domains.keys())
        domain_lists = [sorted(var_domains[v]) for v in vars_list]
        for combo in itertools.product(*domain_lists):
            yield dict(zip(vars_list, combo))

    # ------------------------------------------------------------------
    # Inequality constraints (grounding-time filter, not Boolean vars)
    # ------------------------------------------------------------------

    def _extract_inequalities(self, node):
        """Strip `X != Y` comparisons from body; return (stripped_body, inequalities).

        inequalities: list of (lhs_str, rhs_str) pairs.  Each element is a
        variable name or a constant string (resolved via the substitution dict).

        Python operator precedence means `A & B & Y != Z` parses as
        `(A & B & Y) != Z` (a top-level Compare with a BinOp on the left).
        Explicit parentheses `A & B & (Y != Z)` produce a BinOp chain whose
        rightmost leaf is the Compare.  Both forms are handled here.
        """
        def get_name(n):
            if isinstance(n, ast.Name):
                return n.id
            if isinstance(n, ast.Constant):
                return str(n.value)
            return None

        def is_notEq(n):
            return (isinstance(n, ast.Compare)
                    and len(n.ops) == 1
                    and isinstance(n.ops[0], ast.NotEq))

        def try_extract(n):
            """Try to read n as an inequality; return (remaining_body, (lhs, rhs)) or None.

            Handles two forms:
              - `Y != Z`          → remaining = True
              - `(A & ... & Y) != Z`  → remaining = A & ...  (precedence artifact)
            """
            if not is_notEq(n):
                return None
            rhs = get_name(n.comparators[0])
            if rhs is None:
                return None
            lhs_node = n.left
            lhs = get_name(lhs_node)
            if lhs is not None:
                return ast.Constant(True), (lhs, rhs)
            # Python precedence: left side is a BinOp chain ending in a Name
            if isinstance(lhs_node, ast.BinOp) and isinstance(lhs_node.op, ast.BitAnd):
                trailing = get_name(lhs_node.right)
                if trailing is not None:
                    return lhs_node.left, (trailing, rhs)
            return None

        # ── case 1: top-level is a Compare (no-parens precedence artifact) ──────
        result = try_extract(node)
        if result is not None:
            body, ineq = result
            return body, [ineq]

        # ── case 2: BinOp(&) chain — flatten, partition, rebuild ─────────────────
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd):
            # Flatten left-associative BinOp(&) chain into an ordered list
            elems: list = []
            cur = node
            while isinstance(cur, ast.BinOp) and isinstance(cur.op, ast.BitAnd):
                elems.insert(0, cur.right)
                cur = cur.left
            elems.insert(0, cur)

            keep, ineqs = [], []
            for elem in elems:
                r = try_extract(elem)
                if r is not None:
                    remaining, ineq = r
                    if not (isinstance(remaining, ast.Constant) and remaining.value is True):
                        keep.append(remaining)  # residual body after precedence extraction
                    ineqs.append(ineq)
                else:
                    keep.append(elem)

            if not ineqs:
                return node, []

            if len(keep) == 0:
                return ast.Constant(True), ineqs
            if len(keep) == 1:
                return keep[0], ineqs
            rebuilt = keep[0]
            for k in keep[1:]:
                rebuilt = ast.BinOp(left=rebuilt, op=ast.BitAnd(), right=k)
            return rebuilt, ineqs

        return node, []

    def _satisfies_inequalities(self, substitution, inequalities):
        """Return True iff all inequality constraints pass for this substitution."""
        for lhs, rhs in inequalities:
            lv = substitution.get(lhs, lhs)
            rv = substitution.get(rhs, rhs)
            if str(lv) == str(rv):
                return False
        return True

    # ------------------------------------------------------------------
    # Relevance grounding: forward-chaining from base facts
    # ------------------------------------------------------------------

    def _domains_from_derivable(self, body_node, derivable):
        """Compute variable domains by matching body subscripts against derivable atoms.

        For each subscript `pred[A, B, ...]` in the body, find all (pred, args) tuples
        in derivable that match the pattern (constants must equal, variables are free).
        Returns a dict mapping variable name → set of possible ground values.
        """
        domains: dict[str, set] = {}
        for child in ast.walk(body_node):
            if not isinstance(child, ast.Subscript) or not isinstance(child.value, ast.Name):
                continue
            pred, args = self._parse_subscript(child)
            if pred is None or not args:
                continue
            for (dpred, dargs) in derivable:
                if dpred != pred or len(dargs) != len(args):
                    continue
                # Check constant positions match
                ok = True
                for parg, darg in zip(args, dargs):
                    if not self._is_variable(parg) and str(parg) != str(darg):
                        ok = False
                        break
                if ok:
                    for parg, darg in zip(args, dargs):
                        if self._is_variable(parg):
                            domains.setdefault(str(parg), set()).add(str(darg))
        return domains

    def _body_is_derivable(self, node, derivable_names: set) -> bool:
        """True iff every ground predicate atom referenced in node is in derivable_names."""
        known_preds = set(self.predicate_domains.keys())
        for child in ast.walk(node):
            if not isinstance(child, ast.Name) or not isinstance(child.ctx, ast.Load):
                continue
            name = child.id
            for pred in known_preds:
                if name == pred or name.startswith(pred + '__'):
                    if name not in derivable_names:
                        return False
                    break
        return True

    def _head_args_from_fo_stmt(self, stmt):
        """Return (pred, head_args) for a first-order AnnAssign, stripping any trailing weight."""
        if not isinstance(stmt, ast.AnnAssign) or not isinstance(stmt.target, ast.Subscript):
            return None, None
        pred, args = self._parse_subscript(stmt.target)
        if pred is None or not self._is_first_order(args):
            return None, None
        head_args = args
        if args and isinstance(stmt.target.slice, ast.Tuple):
            last_elt = stmt.target.slice.elts[-1]
            w = _eval_numeric(last_elt) if not isinstance(last_elt, ast.Name) else None
            if w is not None:
                head_args = args[:-1]
        return pred, head_args

    def _compute_derivable(self, fo_stmts: list) -> set:
        """Forward-chain from explicitly declared facts to find all derivable ground atoms.

        Returns a set of ground name strings (e.g. 'edge__n1__n2', 'path__n1__n5').
        Also updates self.predicate_domains with domains of derived predicates.
        """
        # Seed with explicitly declared base facts (weight > 0 or any explicit declaration)
        derivable: set[tuple] = set(self._ground_atoms)
        derivable_names: set[str] = {
            self._ground_name(p, list(a)) for p, a in derivable
        }

        changed = True
        while changed:
            changed = False
            for stmt in fo_stmts:
                pred, head_args = self._head_args_from_fo_stmt(stmt)
                if pred is None or head_args is None:
                    continue
                body_raw = stmt.annotation
                body, inequalities = self._extract_inequalities(body_raw)

                # Variable domains restricted to currently derivable atoms
                var_domains = self._domains_from_derivable(body, derivable)
                for v in head_args:
                    if self._is_variable(v) and v not in var_domains:
                        var_domains[v] = set(self.constants) # type: ignore
                if not var_domains and any(self._is_variable(v) for v in head_args):
                    continue  # no domain info yet

                for subst in self._enumerate_substitutions(var_domains):
                    if not self._satisfies_inequalities(subst, inequalities):
                        continue
                    ground_head_args = tuple(str(subst.get(str(a), a)) for a in head_args)
                    atom = (pred, ground_head_args)
                    if atom in derivable:
                        continue
                    ground_body = self._substitute_node(body, subst)
                    if self._body_is_derivable(ground_body, derivable_names):
                        derivable.add(atom)
                        gname = self._ground_name(pred, list(ground_head_args))
                        derivable_names.add(gname)
                        for i, arg in enumerate(ground_head_args):
                            self.predicate_domains.setdefault(pred, {}).setdefault(i, set()).add(arg)
                        changed = True

        return derivable_names

    def _body_has_subscripts(self, node):
        """Return True if the node contains any Subscript with a Name value (a predicate call)."""
        for child in ast.walk(node):
            if isinstance(child, ast.Subscript) and isinstance(child.value, ast.Name):
                return True
        return False

    def _has_subscript_body_plain_head(self, stmt):
        """True if stmt is a plain-Name-head rule whose body contains subscript atoms."""
        if not isinstance(stmt, ast.AnnAssign) or not isinstance(stmt.target, ast.Name):
            return False
        return self._body_has_subscripts(stmt.annotation)

    def visit_AnnAssign(self, node):
        target = node.target

        if not isinstance(target, ast.Subscript):
            # Plain-Name head: rewrite body subscripts when they reference known predicates.
            # This handles `someHead :- pred[c1]` (ground body atom) and
            # `someHead :- pred[_]` (wildcard, already renamed to ANON* before this runs).
            if isinstance(target, ast.Name) and self._body_has_subscripts(node.annotation):
                body = node.annotation
                var_domains = self._get_variable_domains(body)
                if not var_domains:
                    # All-ground body — rewrite subscripts with an empty substitution.
                    node.annotation = self._substitute_node(body, {})
                    return node
                # Body has variables (including renamed wildcards) — one rule per substitution.
                rules = []
                for substitution in self._enumerate_substitutions(var_domains):
                    ground_body = self._substitute_node(body, substitution)
                    rules.append(ast.AnnAssign(
                        target=copy.deepcopy(target),
                        annotation=ground_body,
                        value=None,
                        simple=1,
                    ))
                return rules
            return node

        # pass through weighted/inhibitory heads: b[0.4] :- c, b[False] :- c, b[0.2, False] :- c
        # Also pass through fraction weights: b[1/3] :- c
        if isinstance(target.slice, ast.Constant):
            return node
        if isinstance(target.slice, ast.Tuple) and all(isinstance(e, ast.Constant) for e in target.slice.elts):
            return node
        if _eval_numeric(target.slice) is not None:
            return node

        pred, args = self._parse_subscript(target)

        ann_weight = _eval_weight(node.annotation)
        if ann_weight is not None:
            if not self._is_first_order(args):
                self._register_ground_fact(pred, args)
                ground = self._ground_name(pred, args)
                return ast.AnnAssign(
                    target=ast.Name(id=ground, ctx=ast.Store()),
                    annotation=ast.Constant(ann_weight),
                    value=None,
                    simple=1,
                )
            raise ValueError(f"Cannot have variables in weight declaration: {pred}[{args}]")

        if self._is_first_order(args):
            body_raw = node.annotation
            body, inequalities = self._extract_inequalities(body_raw)
            weight = None
            head_args = args
            if args and isinstance(target.slice, ast.Tuple):
                last_elt = target.slice.elts[-1]
                w = _eval_numeric(last_elt) if not isinstance(last_elt, ast.Name) else None
                if w is not None:
                    weight = w
                    head_args = args[:-1]

            all_vars = set()
            for a in head_args:
                if self._is_variable(a):
                    all_vars.add(a)
            var_domains = self._get_variable_domains(body)
            for v in all_vars:
                if v not in var_domains:
                    var_domains[v] = set(self.constants)

            ground_rules = []
            for substitution in self._enumerate_substitutions(var_domains):
                if not self._satisfies_inequalities(substitution, inequalities):
                    continue
                ground_head_args = [substitution.get(a, a) for a in head_args]
                # Relevance filter: skip atoms not reachable by forward-chaining
                if self._derivable_names is not None:
                    gname = self._ground_name(pred, ground_head_args)
                    if gname not in self._derivable_names:
                        continue
                for i, arg in enumerate(ground_head_args):
                    self.predicate_domains.setdefault(pred, {}).setdefault(i, set()).add(arg)
                ground_head = self._ground_name(pred, ground_head_args)

                if weight is not None:
                    head_node = ast.Subscript(
                        value=ast.Name(id=ground_head, ctx=ast.Load()),
                        slice=ast.Constant(weight),
                        ctx=ast.Store(),
                    )
                else:
                    head_node = ast.Name(id=ground_head, ctx=ast.Store())

                ground_body = self._substitute_node(body, substitution)

                ground_rules.append(ast.AnnAssign(
                    target=head_node,
                    annotation=ground_body,
                    value=None,
                    simple=1,
                ))
            return ground_rules

        ground = self._ground_name(pred, args)
        ground_body = self._substitute_node(node.annotation, {})
        return ast.AnnAssign(
            target=ast.Name(id=ground, ctx=ast.Store()),
            annotation=ground_body,
            value=None,
            simple=1,
        )

    def _is_first_order_stmt(self, stmt):
        if not isinstance(stmt, ast.AnnAssign) or not isinstance(stmt.target, ast.Subscript):
            return False
        if isinstance(stmt.target.slice, ast.Constant):
            return False
        if isinstance(stmt.target.slice, ast.Tuple) and all(isinstance(e, ast.Constant) for e in stmt.target.slice.elts):
            return False
        if _eval_numeric(stmt.target.slice) is not None:
            return False
        pred, args = self._parse_subscript(stmt.target)
        return pred is not None and self._is_first_order(args)

    def _visit_stmt(self, stmt):
        result = self.visit(stmt)
        if isinstance(result, list):
            return result
        if result is not None:
            return [result]
        return []

    def visit_FunctionDef(self, node):
        # Rename every `_` to a unique anonymous variable (ANON0, ANON1, …) so
        # wildcards are treated as independent existential variables during grounding.
        renamer = _WildcardRenamer()
        node.body = [renamer.visit(stmt) for stmt in node.body]

        new_body = []
        deferred_fo = []   # first-order subscript-head rules (may register new predicate domains)
        deferred_wc = []   # plain-Name-head rules with subscript bodies (depend on predicate domains)
        deferred_q  = []   # query/evidence calls with wildcard subscripts (need all domains known)

        for stmt in node.body:
            if self._is_first_order_stmt(stmt):
                deferred_fo.append(stmt)
            elif self._has_subscript_body_plain_head(stmt):
                deferred_wc.append(stmt)
            elif self._is_wildcard_query_stmt(stmt):
                deferred_q.append(stmt)
            else:
                new_body.extend(self._visit_stmt(stmt))

        # Relevance grounding: forward-chain from base facts to find only derivable atoms.
        # This also populates predicate_domains for derived predicates.
        self._derivable_names = self._compute_derivable(deferred_fo)

        # First deferred pass: first-order subscript-head rules (only emit derivable groundings)
        for stmt in deferred_fo:
            new_body.extend(self._visit_stmt(stmt))

        # Second deferred pass: plain-Name-head rules with subscript bodies (uses those domains)
        for stmt in deferred_wc:
            new_body.extend(self._visit_stmt(stmt))

        # Third deferred pass: wildcard query/evidence calls (all predicate domains now known)
        for stmt in deferred_q:
            new_body.extend(self._visit_stmt(stmt))

        node.body = new_body
        return node

    def _is_wildcard_query_stmt(self, stmt):
        """True for query(pred[ANON*]) / evidence(pred[ANON*], …) calls with variable subscripts."""
        if not isinstance(stmt, ast.Expr):
            return False
        val = stmt.value
        if not isinstance(val, ast.Call) or not isinstance(val.func, ast.Name):
            return False
        if val.func.id not in ('query', 'evidence'):
            return False
        inner = val.args[0] if val.args else None
        if not isinstance(inner, ast.Subscript):
            return False
        pred, args = self._parse_subscript(inner)
        return pred is not None and self._is_first_order(args)

    def visit_Expr(self, node):
        val = node.value

        if isinstance(val, ast.Subscript) and isinstance(val.value, ast.Name):
            pred, args = self._parse_subscript(val)
            if pred and args and not self._is_first_order(args):
                self._register_ground_fact(pred, args)
                ground = self._ground_name(pred, args)
                return ast.Expr(value=ast.Name(id=ground, ctx=ast.Load()))

        if isinstance(val, ast.Call) and isinstance(val.func, ast.Name):
            if val.func.id in ("query", "evidence"):
                inner = val.args[0] if val.args else None
                if isinstance(inner, ast.Subscript):
                    pred, args = self._parse_subscript(inner)
                    if pred and not self._is_first_order(args):
                        ground = self._ground_name(pred, args)
                        val.args[0] = ast.Name(id=ground, ctx=ast.Load())
                        return node
                    # Wildcard/variable subscript: expand to one call per ground instance.
                    # Predicate domains are available because this runs in deferred_q pass.
                    if pred and self._is_first_order(args):
                        var_domains = {}
                        for i, arg in enumerate(args):
                            if self._is_variable(arg):
                                var_domains[arg] = set(
                                    self.predicate_domains.get(pred, {}).get(i, set())
                                )
                        if var_domains:
                            extra_args = val.args[1:]  # e.g. truth value for evidence
                            result = []
                            for substitution in self._enumerate_substitutions(var_domains):
                                ground_args = [substitution.get(a, a) for a in args]
                                ground_name = self._ground_name(pred, ground_args)
                                result.append(ast.Expr(value=ast.Call(
                                    func=ast.Name(id=val.func.id, ctx=ast.Load()),
                                    args=[ast.Name(id=ground_name, ctx=ast.Load())] + extra_args,
                                    keywords=[],
                                )))
                            return result
                if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name):
                    pred = inner.func.id
                    args = [a.id for a in inner.args if isinstance(a, ast.Name)]
                    if args and not self._is_first_order(args):
                        ground = self._ground_name(pred, args)
                        val.args[0] = ast.Name(id=ground, ctx=ast.Load())
                        return node

        return node


class _WildcardRenamer(ast.NodeTransformer):
    """Renames each anonymous variable `_` to a unique uppercase name ANON0, ANON1, …

    Each occurrence of `_` gets a fresh name so wildcards are independent
    (matching standard Prolog semantics where `_` is always a distinct variable).
    """
    def __init__(self):
        self._counter = 0

    def visit_Name(self, node):
        if node.id == '_':
            name = f'ANON{self._counter}'
            self._counter += 1
            return ast.Name(id=name, ctx=node.ctx)
        return node


class _PLPSubstituter(ast.NodeTransformer):

    def __init__(self, substitution, grounding):
        self.substitution = substitution
        self.grounding = grounding

    def visit_Subscript(self, node):
        self.generic_visit(node)

        if not isinstance(node.value, ast.Name):
            return node

        pred, args = self.grounding._parse_subscript(node)
        if pred is None:
            return node

        ground_args = []
        for arg in args:
            if self.grounding._is_variable(arg) and arg in self.substitution:
                ground_args.append(self.substitution[arg])
            else:
                ground_args.append(arg)

        # Don't rewrite if any variable is still unresolved
        if any(self.grounding._is_variable(a) for a in ground_args):
            return node

        # Only rewrite known predicates — avoids clobbering unrelated subscript expressions
        if pred not in self.grounding.predicate_domains:
            return node

        ground_name = self.grounding._ground_name(pred, ground_args)
        return ast.Name(id=ground_name, ctx=ast.Load())

class PLPSemantics(ast.NodeVisitor):
    """Extracts semantic formulas from the compiled PLP program AST.

    Visits the compiled (transformed) AST and extracts:
      - biconditionals: rule assignments like `b = a and (not c)` → `b <-> (a & ~c)`
      - categoricals: AD mutual exclusion asserts like `assert sum([a,b,c]) == 1`
      - evidence: sampler assignments with IfExp like `a = ... if False else True`
    """

    def __init__(self, mapper):
        self.mapper = mapper
        self._formulas = []
        self._conditionals = []
        self._categoricals = []
        self._evidence = {}

    @property
    def formulas(self):
        return self._formulas

    @property
    def conditionals(self):
        return self._conditionals

    @property
    def categoricals(self):
        return self._categoricals

    @property
    def evidence(self):
        return self._evidence

    @property
    def evidence_formula(self):
        """Build conjunction of evidence literals: e.g., {a: True, b: False} → a ∧ ¬b"""
        if not self._evidence:
            return None
        formula = None
        for name, truth in self._evidence.items():
            lit = self.mapper.Symbol(name, return_val=True)
            if not truth:
                lit = self.mapper.Not(lit)
            formula = lit if formula is None else self.mapper.And(formula, lit)
        return formula

    def _is_sampler(self, node):
        """Check if an Assign value is a np.random.random() < weight expression."""
        if isinstance(node, ast.Compare):
            return isinstance(node.left, ast.Call)
        if isinstance(node, ast.IfExp):
            return self._is_sampler(node.body)
        return False

    def visit_Assign(self, node):
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            return
        name = node.targets[0].id
        val = node.value

        # evidence: a = np.random.random() < 0.9 if False else True
        if isinstance(val, ast.IfExp) and isinstance(val.test, ast.Constant) and val.test.value is False:
            assert isinstance(val.orelse, ast.Constant)
            self._evidence[name] = val.orelse.value
            return

        # sampler: a = np.random.random() < 0.9 — skip, weight already tracked
        if self._is_sampler(val):
            return

        # rule assignment: b = <bool_expr> → biconditional b <-> rhs
        # ast.Constant excluded: fixpoint init `head = False` must not create head ↔ False
        if isinstance(val, (ast.BoolOp, ast.UnaryOp, ast.Name)):
            lhs = extract_cond(self.mapper, test=ast.Name(id=name, ctx=ast.Load()), relax=True)
            rhs = extract_cond(self.mapper, test=val, relax=True)
            self._formulas.append(self.mapper.Biconditional(lhs, rhs))

    def _build_exactly_one(self, names):
        """Build exactly-one constraint: (a ∨ b ∨ c) ∧ (¬a ∨ ¬b) ∧ (¬a ∨ ¬c) ∧ (¬b ∨ ¬c)"""
        syms = [self.mapper.Symbol(n, return_val=True) for n in names]

        # at least one
        formula = syms[0]
        for s in syms[1:]:
            formula = self.mapper.Or(formula, s)

        # at most one: pairwise mutual exclusion
        for i in range(len(syms)):
            for j in range(i + 1, len(syms)):
                clause = self.mapper.Or(self.mapper.Not(syms[i]), self.mapper.Not(syms[j]))
                formula = self.mapper.And(formula, clause)

        return formula

    def visit_Assert(self, node):
        # AD mutual exclusion: assert sum([a, b, c]) == 1
        if isinstance(node.test, ast.Compare):
            left = node.test.left
            if isinstance(left, ast.Call) and isinstance(left.func, ast.Name) and left.func.id == 'sum':
                if left.args and isinstance(left.args[0], ast.List):
                    names = [
                        elt.id for elt in left.args[0].elts
                        if isinstance(elt, ast.Name)
                    ]


                    self._categoricals.append(names)
                    self._formulas.append(self._build_exactly_one(names))
                    return

        # evidence assertions on rule heads: assert b / assert not b
        if isinstance(node.test, (ast.Name, ast.UnaryOp)):
            formula = extract_cond(self.mapper, test=node.test, relax=True)
            self._conditionals.append(formula)

    def visit_While(self, node):
        for stmt in node.body:
            if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                continue
            target = stmt.targets[0]
            if not isinstance(target, ast.Name) or not target.id.endswith('__new'):
                continue
            base_name = target.id.removesuffix('__new')
            val = stmt.value
            if isinstance(val, (ast.BoolOp, ast.UnaryOp, ast.Name, ast.Constant)):
                lhs = extract_cond(self.mapper, test=ast.Name(id=base_name, ctx=ast.Load()), relax=True)
                rhs = extract_cond(self.mapper, test=val, relax=True)
                self._formulas.append(self.mapper.Biconditional(lhs, rhs))

    def visit_FunctionDef(self, node):
        self.generic_visit(node)


class PLPTransformer(ast.NodeTransformer):
    """
    Transforms the raw AST of the input program to a Python AST that can be executed. 
    Serves as the foundation of the transformation pipeline for the constraint language.

    Attributes: 
        fact_weights (dict): 
            A dictionary mapping ground facts to their weights. E.g., {"s__a": 0.9}
        rule_heads (list): 
            A list of the heads of all rules in the program. E.g., ["alarm"]
        rule_weights (dict): 
            A dictionary mapping rule heads to their weights. E.g., {"alarm": 0.5}
        queries (list): 
            A list of all queries in the program. E.g., ["alarm"]
        evidence (dict): 
            A dictionary mapping ground facts to their observed values. E.g., {"s__a": True}


    """
    def __init__(self, variable_weights: dict[str, float] | None = None):
        self.variable_weights = variable_weights or {}
        self.fact_weights = {}
        self.clause_weights = {}  # {"b__w0": 0.4} — separate sampler per weighted rule
        self.rules = {}          # {"b": [body_expr1, body_expr2, ...]}
        self.neg_rules = {}      # {"b": [inh_body1, ...]} from b[False] :- ...
        self.rule_order = []     # preserve first-seen order of heads
        self._weight_counter = {}  # {"b": 0, ...} — counter for naming b__w0, b__w1
        self.queries = []
        self.evidence = {}
        self._ad_variables = set()
        self._categorical_groups = []  # [["a", "b", "c"], ...] — AD groups

    def _parse_head(self, node):
        """Returns (name, weight, inhibitory).

        e1             → ('e1', None, False)
        e1[0.8]        → ('e1', 0.8, False)
        e1[1/9]        → ('e1', 0.111…, False)   # fraction literal
        e1[False]      → ('e1', None, True)
        e1[0.2, False] → ('e1', 0.2, True)
        """
        if isinstance(node, ast.Subscript):
            assert isinstance(node.value, ast.Name)
            slc = node.slice
            if isinstance(slc, ast.Tuple):
                vals = []
                for e in slc.elts:
                    if isinstance(e, ast.Constant):
                        vals.append(e.value)
                    else:
                        v = _eval_numeric(e)
                        if v is not None:
                            vals.append(v)
                weight = next((v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)), None)
                inhibitory = False in vals
                return node.value.id, weight, inhibitory
            if isinstance(slc, ast.Constant):
                if slc.value is False:
                    return node.value.id, None, True
                return node.value.id, slc.value, False
            # Numeric expression weight (e.g. 1/9, 3/4)
            v = _eval_numeric(slc)
            if v is not None:
                return node.value.id, v, False
            raise ValueError(f"Unsupported head subscript: {ast.dump(slc)}")
        assert isinstance(node, ast.Name)
        return node.id, None, False
    
    def _body_to_expr(self, node):
        """Convert & | ~ to and / or / not."""
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd):
            left = self._body_to_expr(node.left)
            right = self._body_to_expr(node.right)
            return ast.BoolOp(op=ast.And(), values=[left, right])
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            left = self._body_to_expr(node.left)
            right = self._body_to_expr(node.right)
            return ast.BoolOp(op=ast.Or(), values=[left, right])
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
            return ast.UnaryOp(op=ast.Not(), operand=self._body_to_expr(node.operand))
        return node
    
    def _add_rule(self, head_name, weight, body_expr):
        if weight is not None:
            idx = self._weight_counter.get(head_name, 0)
            self._weight_counter[head_name] = idx + 1
            w_name = f"{head_name}__w{idx}"
            self.clause_weights[w_name] = weight
            self.variable_weights[w_name] = weight
            body_expr = ast.BoolOp(
                op=ast.And(),
                values=[
                    ast.Name(id=w_name, ctx=ast.Load()),
                    body_expr,
                ],
            )
        if head_name not in self.rules:
            self.rules[head_name] = []
            self.rule_order.append(head_name)
        self.rules[head_name].append(body_expr)

    def _add_neg_rule(self, head_name, body_expr, weight=None):
        if weight is not None:
            idx = self._weight_counter.get(head_name, 0)
            self._weight_counter[head_name] = idx + 1
            w_name = f"{head_name}__w{idx}"
            self.clause_weights[w_name] = weight
            self.variable_weights[w_name] = weight
            body_expr = ast.BoolOp(
                op=ast.And(),
                values=[
                    ast.Name(id=w_name, ctx=ast.Load()),
                    body_expr,
                ],
            )
        if head_name not in self.neg_rules:
            self.neg_rules[head_name] = []
        self.neg_rules[head_name].append(body_expr)
        if head_name not in self.rules:
            self.rules[head_name] = []
            self.rule_order.append(head_name)

    def _strip_usub(self, node):
        """Strip the USub introduced by :- syntax, handling operator precedence."""
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return node.operand
        # USub consumed by first operand due to precedence — strip from leftmost leaf
        if isinstance(node, ast.BinOp):
            node.left = self._strip_usub(node.left)
            return node
        return node
    
    def _collect_ad_pairs(self, node):
        """Recursively collect (name, weight) from chained BitOr of Subscripts."""
        if isinstance(node, ast.Subscript):
            name, weight, _ = self._parse_head(node)
            return [(name, weight)]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return self._collect_ad_pairs(node.left) + self._collect_ad_pairs(node.right)
        raise ValueError(f"Unexpected node in annotated disjunction: {ast.dump(node)}")


    def _process_annotated_disjunction(self, node):
        """a[0.3] | b[0.5] | c[0.1] → collect weights, return mutual exclusion assert.
        
        
        """
        pairs = self._collect_ad_pairs(node)  # [("a", 0.3), ("b", 0.5), ("c", 0.1)]

        prob_sum = 0.0
        self._categorical_groups.append([name for name, _ in pairs])
        for name, weight in pairs:
            self.fact_weights[name] = weight
            self.variable_weights[name] = weight  # type: ignore
            self._ad_variables.add(name)
            assert isinstance(weight, (int, float)) and 0.0 <= weight <= 1.0, f"Invalid weight {weight} for {name} in annotated disjunction"
            prob_sum += weight

        _WEIGHT_TOL = 1e-9
        if prob_sum > 1.0 + _WEIGHT_TOL:
            raise ValueError(f"Probabilities in annotated disjunction sum to > 1: {prob_sum}")

        names = [name for name, _ in pairs]

        if 1.0 - prob_sum > _WEIGHT_TOL:
            none_name = "__".join(names) + "__none"
            none_weight = round(1.0 - prob_sum, 10)
            self.fact_weights[none_name] = none_weight
            self.variable_weights[none_name] = none_weight
            self._ad_variables.add(none_name)
            names.append(none_name)
            self._categorical_groups[-1].append(none_name)

        return ast.Assert(
            test=ast.Compare(
                left=ast.Call(
                    func=ast.Name(id='sum', ctx=ast.Load()),
                    args=[ast.List(
                        elts=[ast.Name(id=n, ctx=ast.Load()) for n in names],
                        ctx=ast.Load(),
                    )],
                    keywords=[],
                ),
                ops=[ast.Eq()],
                comparators=[ast.Constant(1)],
            ),
        )
    def _is_annotated_disjunction(self, node):
        """Check if a BinOp(BitOr) chain consists entirely of Subscript nodes (weighted heads)."""
        if isinstance(node, ast.Subscript):
            return True
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return self._is_annotated_disjunction(node.left) and self._is_annotated_disjunction(node.right)
        return False

    def _process_stmt(self, stmt):
        """Process a single statement, collecting into self state."""
        # rule: a :- (b & c) or a[0.8] :- b, a:0.8 (a weighted fact without a body)
        if isinstance(stmt, ast.AnnAssign):
            head_name, weight, inhibitory = self._parse_head(stmt.target)

            ### weighted fact, no body (annotation is a numeric literal or fraction like 1/9)
            ann_weight = _eval_weight(stmt.annotation)
            if ann_weight is not None:
                self.fact_weights[head_name] = ann_weight
                self.variable_weights[head_name] = ann_weight
                return None

            body = self._strip_usub(stmt.annotation)  # strip USub
            body_expr = self._body_to_expr(body)
            if inhibitory:
                self._add_neg_rule(head_name, body_expr, weight)
            else:
                self._add_rule(head_name, weight, body_expr)
            return None

        if isinstance(stmt, ast.Expr):
            val = stmt.value

            # query(a) / evidence(b, True)
            if isinstance(val, ast.Call) and isinstance(val.func, ast.Name):
                if val.func.id == "query":
                    assert isinstance(val.args[0], ast.Name)
                    self.queries.append(val.args[0].id)
                    return None
                if val.func.id == "evidence":
                    assert isinstance(val.args[0], ast.Name)
                    name = val.args[0].id
                    assert isinstance(val.args[1], ast.Constant)
                    truth = val.args[1].value if len(val.args) > 1 else True
                    self.evidence[name] = truth
                    return None
                
            ## annotated disjunction: a[0.3] | b[0.5] | c[0.1]
            if isinstance(val, ast.BinOp) and isinstance(val.op, ast.BitOr):
                if self._is_annotated_disjunction(val):
                    return self._process_annotated_disjunction(val)

            # weighted fact: b[0.9]
            if isinstance(val, ast.Subscript):
                name, weight, _ = self._parse_head(val)
                self.fact_weights[name] = weight
                self.variable_weights[name] = weight  # type: ignore
                return None

            # bare fact: b  (always true)
            if isinstance(val, ast.Name):
                self.fact_weights[val.id] = 1.0
                self.variable_weights[val.id] = 1.0
                return None
                # )

        return stmt
    
    def _find_sccs(self, pos_deps, neg_deps):
        """An implementation of Tarjan's algorithm for strongly connected components. 
        Used for finding cycles in the underlying program and rules. 

        Arguments: 
            pos_deps: 
                A dictionary mapping rule heads to the set of rule heads they depend on positively.
            neg_deps: 
                A dictionary mapping rule heads to the set of rule heads they depend on negatively. 
        Returns: 
            A list of strongly connected components, where each component is a list of rule heads. 
            The components are returned in reverse topological order (i.e., if component A depends on B, 
            then A appears before B in the list).  
        
        """
        all_deps = {h: pos_deps[h] | neg_deps[h] for h in self.rules}
        
        index_counter = [0]
        stack = []
        on_stack = set()
        index = {}
        lowlink = {}
        sccs = []

        def strongconnect(v):
            index[v] = lowlink[v] = index_counter[0]
            index_counter[0] += 1
            stack.append(v)
            on_stack.add(v)

            for w in all_deps.get(v, set()):
                if w not in index:
                    strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif w in on_stack:
                    lowlink[v] = min(lowlink[v], index[w])

            if lowlink[v] == index[v]:
                scc = []
                while True:
                    w = stack.pop()
                    on_stack.remove(w)
                    scc.append(w)
                    if w == v:
                        break
                sccs.append(scc)

        for v in self.rules:
            if v not in index:
                strongconnect(v)

        return sccs  # returned in reverse topological order
    
    def _build_dependency_graph(self):
        """Build positive and negative dependency graphs from collected rules."""
        pos_deps = {h: set() for h in self.rules}
        neg_deps = {h: set() for h in self.rules}
        for head, bodies in self.rules.items():
            for body in bodies:
                p, n = self._extract_deps(body)
                pos_deps[head] |= p
                neg_deps[head] |= n
        for head, bodies in self.neg_rules.items():
            for body in bodies:
                p, n = self._extract_deps(body)
                neg_deps[head] |= p
                pos_deps[head] |= n
        return pos_deps, neg_deps
    
    def _extract_deps(self, node, positive=True):
        """Extract variable names from a body expression, tracking polarity."""
        pos, neg = set(), set()
        if isinstance(node, ast.Name):
            if node.id in self.rules:  # only track rule heads, not facts
                (pos if positive else neg).add(node.id)
        elif isinstance(node, ast.BoolOp):
            for val in node.values:
                p, n = self._extract_deps(val, positive)
                pos |= p; neg |= n
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            p, n = self._extract_deps(node.operand, not positive)
            pos |= p; neg |= n
        elif isinstance(node, ast.Compare):
            pass  # random() < weight, skip
        return pos, neg
    
    def _check_stratification(self, sccs, neg_deps):
        """Reject programs with negation within an SCC (negative cycle)."""
        node_to_scc = {}
        for i, scc in enumerate(sccs):
            for node in scc:
                node_to_scc[node] = i
        for head, negs in neg_deps.items():
            for dep in negs:
                if dep in node_to_scc and node_to_scc[dep] == node_to_scc[head]:
                    raise ValueError(
                        f"Negative cycle: '{head}' and '{dep}' depend on each "
                        f"other through negation (not stratifiable)"
                    )
                
    def _has_cycle(self, scc, pos_deps):
        """Check if an SCC has a real cycle (multi-node or self-loop)."""
        if len(scc) > 1:
            return True
        v = scc[0]
        return v in pos_deps.get(v, set())
    
    def _make_rule_value(self, head):
        """Build the combined value for a rule head: (pos1 or pos2) and not inh1 and not inh2."""
        pos_bodies = self.rules.get(head, [])
        if not pos_bodies:
            return ast.Constant(False)

        if len(pos_bodies) == 1:
            pos_expr = pos_bodies[0]
        else:
            pos_expr = ast.BoolOp(op=ast.Or(), values=pos_bodies)

        neg_bodies = self.neg_rules.get(head, [])
        if not neg_bodies:
            return pos_expr

        inhibitors = [ast.UnaryOp(op=ast.Not(), operand=b) for b in neg_bodies]
        return ast.BoolOp(op=ast.And(), values=[pos_expr] + inhibitors) # type: ignore 
    
    def _make_fixpoint_loop(self, scc):
        """Emit a while-True loop that iterates SCC to fixpoint.
        
        Generates:
            a = False; b = False
            while True:
                a_new = <body_a>
                b_new = <body_b>
                if a_new == a and b_new == b:
                    break
                a, b = a_new, b_new
        """
        stmts = []
        
        # initialize all to False
        for head in scc:
            stmts.append(ast.Assign(
                targets=[ast.Name(id=head, ctx=ast.Store())],
                value=ast.Constant(False),
            ))
        
        # build loop body
        loop_body = []
        new_names = {h: f"{h}__new" for h in scc}
        
        # assignments: a_new = <body>
        for head in scc:
            loop_body.append(ast.Assign(
                targets=[ast.Name(id=new_names[head], ctx=ast.Store())],
                value=self._make_rule_value(head),
            ))
        
        # convergence check: if a_new == a and b_new == b: break
        comparisons = []
        for head in scc:
            comparisons.append(ast.Compare(
                left=ast.Name(id=new_names[head], ctx=ast.Load()),
                ops=[ast.Eq()],
                comparators=[ast.Name(id=head, ctx=ast.Load())],
            ))
        if len(comparisons) == 1:
            test = comparisons[0]
        else:
            test = ast.BoolOp(op=ast.And(), values=comparisons)
        
        loop_body.append(ast.If(
            test=test,
            body=[ast.Break()],
            orelse=[],
        ))
        
        # update: a, b = a_new, b_new
        loop_body.append(ast.Assign(
            targets=[ast.Tuple(
                elts=[ast.Name(id=h, ctx=ast.Store()) for h in scc],
                ctx=ast.Store(),
            )],
            value=ast.Tuple(
                elts=[ast.Name(id=new_names[h], ctx=ast.Load()) for h in scc],
                ctx=ast.Load(),
            ),
        ))
        
        # while True: ...
        stmts.append(ast.While(
            test=ast.Constant(True),
            body=loop_body,
            orelse=[],
        ))
        
        return stmts
    
    def _has_undefined(self, node, defined):
        """Check if a body expression references any undefined atom (closed-world → always false)."""
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            return node.id not in defined
        for child in ast.iter_child_nodes(node):
            if self._has_undefined(child, defined):
                return True
        return False

    def _make_simple_assignment(self, head):
        """Emit: head = body1 or body2 or ..."""
        return ast.Assign(
            targets=[ast.Name(id=head, ctx=ast.Store())],
            value=self._make_rule_value(head),
        )


    def visit_FunctionDef(self, node):
        first_pass_stmts = []
        # first pass: collect facts, rules, evidence, queries
        for stmt in node.body:
            result = self._process_stmt(stmt)
            if result is not None:
                if isinstance(result, list):
                    first_pass_stmts.extend(result)
                else:
                    first_pass_stmts.append(result)

        new_body = []

        ## all fact samplers (including AD variables), with evidence baked in
        for name, weight in self.fact_weights.items():
            if name in self.evidence:
                new_body.append(make_sampler(name, weight, evidence=self.evidence[name]))
            else:
                new_body.append(make_sampler(name, weight))

        ## clause weight samplers
        for name, weight in self.clause_weights.items():
            new_body.append(make_sampler(name, weight))

        ## first-pass results (AD mutual exclusion asserts, etc.)
        new_body.extend(first_pass_stmts)

        ## closed-world: prune rule bodies with undefined atoms (fixed point),
        ## then emit zero-weight samplers for pruned heads
        all_heads = set(self.rules) | set(self.neg_rules)
        changed = True
        while changed:
            changed = False
            defined = set(self.fact_weights) | set(self.clause_weights) | set(self.rules)
            for head in list(self.rules):
                pruned = [b for b in self.rules[head] if not self._has_undefined(b, defined)]
                if len(pruned) != len(self.rules[head]):
                    changed = True
                    self.rules[head] = pruned
                if not self.rules[head]:
                    del self.rules[head]
                    self.rule_order = [h for h in self.rule_order if h != head]
            for head in list(self.neg_rules):
                pruned = [b for b in self.neg_rules[head] if not self._has_undefined(b, defined)]
                if len(pruned) != len(self.neg_rules[head]):
                    changed = True
                    self.neg_rules[head] = pruned
                if not self.neg_rules[head]:
                    del self.neg_rules[head]

        surviving = set(self.fact_weights) | set(self.clause_weights) | set(self.rules)
        for name in sorted(all_heads - surviving):
            new_body.append(make_sampler(name, 0))
            self.variable_weights[name] = 0
            self.fact_weights[name] = 0

        ## do dependency analysis of program and rules.
        pos_deps, neg_deps = self._build_dependency_graph()
        sccs = self._find_sccs(pos_deps, neg_deps)
        self._check_stratification(sccs, neg_deps)

        ## write out rules 
        for scc in sccs:
            if self._has_cycle(scc, pos_deps):
                new_body.extend(self._make_fixpoint_loop(scc))
            else:
                new_body.append(self._make_simple_assignment(scc[0]))
        
        ## make evidence into assertions 
        for name, truth in self.evidence.items():
            if name not in self.fact_weights:
                if truth:
                    test = ast.Name(id=name, ctx=ast.Load())
                else:
                    test = ast.UnaryOp(op=ast.Not(), operand=ast.Name(id=name, ctx=ast.Load()))
                new_body.append(ast.Assert(test=test))

        ## queries 
        if self.queries:
            new_body.append(ast.Return(
                value=ast.Dict(
                    keys=[ast.Constant(q) for q in self.queries],
                    values=[ast.Name(id=q, ctx=ast.Load()) for q in self.queries],
                ),
            ))

        node.body = new_body
        return node
    
@LanguageRegistry(name="plp")
class PLPLanguage(ASTModelingLanguage):
    """Implements a small fragment of a Problog-like probabilistic logic programming (plp) language.
    
    The language supports probabilistic facts and rules, as well as probabilistic inference.
    The language is designed to be simple and illustrative, and is not intended to be a full 
    implementation of Problog. The language is also designed to be easily extensible, allowing 
    for additional features to be added in the future.
    
    """
    def __init__(self, **solver_kwargs):
        super().__init__(**solver_kwargs)     
        self._variable_weights = {}
        self._deterministic_vars = set()
        self._categoricals = set() 
        self._evidence_literals = set()
        self._query_literals = set() 

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
    def evidence_literals(self) -> set[str]: 
        return self._evidence_literals
    @property 
    def query_literals(self) -> set[str]: 
        return self._query_literals
            
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
        return set(self._variable_weights.keys()) | self._deterministic_vars | set(itertools.chain(*self._categoricals))
    

    def parse(self, tree: ast.AST) -> ast.AST:
        """Transforms the raw AST of the input program to a Python AST that can be executed. 
        Serves as the foundation of the transformation pipeline for the constraint language.
        
        Arguments: 
            tree: The raw AST of the input program.
        Returns:
            The transformed AST of the input program.
        """
        ## validate general correctness and safety
        ProgramValidator().visit(tree)

        ## perform grounding (if first-order constructs used)
        grounder = PLPGroundingTransformer()
        tree = grounder.visit(tree)

        ## builds Boolean python program
        transformer = PLPTransformer(self._variable_weights)
        tree = transformer.visit(tree)
        ast.fix_missing_locations(tree)

        ## collect the variables 
        self._variable_weights = transformer.variable_weights
        self._deterministic_vars = set(transformer.rules.keys())
        self._categoricals = [frozenset(g) for g in transformer._categorical_groups]

        ## collect evidence and query variables 
        evidence_literals = [] 
        for evid, polarity in transformer.evidence.items():
            evidence_literals.append(
                str(evid) if polarity is True else f"~{evid}"
            )
        self._evidence_literals = set(evidence_literals)
        self._query_literals = set(transformer.queries)

        # Non-derivable query atoms: never registered as variables → probability 0.
        # Register them with weight 0 so the SDD correctly returns 0 for the query.
        defined = set(self._variable_weights) | self._deterministic_vars
        for q in self._query_literals:
            if q not in defined:
                self._variable_weights[q] = 0.0

        return tree

    def semantics(self, program):
        """Executes the transformed AST of the input program and returns the result. 
        
        Arguments: 
            program: 
                The raw python native program to extra semantics from and execute. 
                For the constraint language, this will be a string containing the raw program.
        Returns:
            The result of executing the input program. For the constraint language, 
            this will be a boolean value indicating whether the constraints are satisfied.
        
        """
        ## collect solver variables 
        vars = [str(v) for v in self.variables]
        aux_vars = set([i for i in vars if i not in self._variable_weights])
        categoricals = set(itertools.chain(*self._categoricals))

        ## initialize variables 
        solver = Solver.from_kwargs(
            var_list=vars, # type: ignore 
            var_weights=self.variable_weights, # type: ignore 
            aux_vars=aux_vars, # type: ignore         
            categoricals=categoricals, # type: ignore
            **self.solver_kwargs, # type: ignore 
        )

        ## build the program formula 
        mapper = solver.solver
        semantics = PLPSemantics(mapper)
        tree = ast.parse(program)
        semantics.visit(tree)

        ## add formulas (biconditionals from rules + exactly-one from ADs)
        solver.add(semantics.formulas)
        
        return solver
