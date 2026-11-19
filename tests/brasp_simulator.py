# type: ignore
"""A Python port of the B-RASP simulator at https://b-rasp.github.io (`brasp.js`), with its
preset programs, copied verbatim from the site's `index.html`.

It is an independent reference for the `brasp` language: it runs the presets from their
original strings, with the simulator's semantics:

- an operation is `("init", symbol)`, `("bool", name, expression)`, or
  `("attention", name, tie, mask, score, value, default)`;
- expressions use `f(i)`, `f(j)`, `&`, `|`, `not`, `1` and `0` (as in math.js, `not` binds
  tighter than `&` and `|`);
- the mask is one of `j<i`, `i<j`, `j<=i`, `i<=j` or `1` (no mask);
- attention takes the leftmost or rightmost j where `(score) & (mask)` holds, and returns the
  value there, or the default if there is none;
- a word is accepted when the last operation is 1 at the last position (the empty word is
  rejected).
"""
import re

__all__ = ["PRESETS", "run", "accepts"]


PRESETS = {
    "dyck_1_2": ("lr", [
        ("init", "l"),
        ("init", "r"),
        ("attention", "P_l", "rightmost", "j<i", "1", "Q_l(j)", "0"),
        ("attention", "S_r", "leftmost", "i<j", "1", "Q_r(j)", "0"),
        ("bool", "I", "(Q_l(i) & S_r(i)) | (P_l(i) & Q_r(i))"),
        ("attention", "A_r", "leftmost", "i<j", "not I(j)", "Q_r(j)", "0"),
        ("attention", "B_l", "rightmost", "j<i", "not I(j)", "Q_l(j)", "0"),
        ("bool", "C", "I(i) | (Q_l(i) & A_r(i)) | (Q_r(i) & B_l(i))"),
        ("attention", "Y", "rightmost", "1", "not C(j)", "0", "1"),
    ]),
    "stair_3": ("abc", [
        ("init", "a"),
        ("init", "b"),
        ("init", "c"),
        ("attention", "A_2", "rightmost", "j<i", "Q_a(j) | Q_b(j)", "Q_a(i) & Q_a(j)", "0"),
        ("attention", "A_3", "rightmost", "j<i", "Q_a(j) | Q_b(j)", "Q_a(i) & A_2(j)", "0"),
        ("attention", "Y", "rightmost", "1", "A_3(j)", "1", "0"),
    ]),
    "ab_plus": ("ab", [
        ("init", "a"),
        ("init", "b"),
        ("attention", "AB", "rightmost", "j<i", "1", "Q_a(j) & Q_b(i)", "0"),
        ("attention", "BA", "leftmost", "i<j", "1", "Q_a(i) & Q_b(j)", "0"),
        ("bool", "I", "AB(i) | BA(i)"),
        ("attention", "Y", "rightmost", "1", "not I(j)", "0", "1"),
    ]),
    "b_prec_a_no_succ_c": ("abc", [
        ("init", "a"),
        ("init", "b"),
        ("init", "c"),
        ("attention", "P_a", "rightmost", "j<i", "Q_a(j)", "1", "0"),
        ("attention", "S_c", "leftmost", "i<j", "Q_c(j)", "1", "0"),
        ("bool", "I", "Q_b(i) & P_a(i) & not S_c(i)"),
        ("attention", "Y", "rightmost", "1", "I(j)", "1", "0"),
    ]),
    "induction_heads": ("abc", [
        ("init", "a"),
        ("init", "b"),
        ("init", "c"),
        ("attention", "PREV_a", "rightmost", "j<i", "1", "Q_a(j)", "0"),
        ("attention", "PREV_b", "rightmost", "j<i", "1", "Q_b(j)", "0"),
        ("attention", "PREV_c", "rightmost", "j<i", "1", "Q_c(j)", "0"),
        ("attention", "PREDICT_a", "rightmost", "j<i", "(Q_a(i) & PREV_a(j)) | (Q_b(i) & PREV_b(j)) | (Q_c(i) & PREV_c(j))", "Q_a(j)", "0"),
        ("attention", "PREDICT_b", "rightmost", "j<i", "(Q_a(i) & PREV_a(j)) | (Q_b(i) & PREV_b(j)) | (Q_c(i) & PREV_c(j))", "Q_b(j)", "0"),
        ("attention", "PREDICT_c", "rightmost", "j<i", "(Q_a(i) & PREV_a(j)) | (Q_b(i) & PREV_b(j)) | (Q_c(i) & PREV_c(j))", "Q_c(j)", "0"),
        ("attention", "Y", "rightmost", "j<i", "1", "(PREDICT_a(j) & Q_a(i)) | (PREDICT_b(j) & Q_b(i)) | (PREDICT_c(j) & Q_c(i))", "0"),
    ]),
    "retrieval": ("abc123", [
        ("init", "a"),
        ("init", "b"),
        ("init", "c"),
        ("init", "1"),
        ("init", "2"),
        ("init", "3"),
        ("attention", "P_a", "rightmost", "j<i", "1", "Q_a(j)", "0"),
        ("attention", "P_b", "rightmost", "j<i", "1", "Q_b(j)", "0"),
        ("attention", "P_c", "rightmost", "j<i", "1", "Q_c(j)", "0"),
        ("attention", "Y_1", "rightmost", "j<i", "(P_a(i) & P_a(j)) | (P_b(i) & P_b(j)) | (P_c(i) & P_c(j))", "Q_1(j)", "0"),
        ("attention", "Y_2", "rightmost", "j<i", "(P_a(i) & P_a(j)) | (P_b(i) & P_b(j)) | (P_c(i) & P_c(j))", "Q_2(j)", "0"),
        ("attention", "Y_3", "rightmost", "j<i", "(P_a(i) & P_a(j)) | (P_b(i) & P_b(j)) | (P_c(i) & P_c(j))", "Q_3(j)", "0"),
        ("bool", "Y_?", "(P_a(i) | P_b(i) | P_c(i)) & not (Y_1(i) | Y_2(i) | Y_3(i))"),
    ]),
}


class Bit:
    """A Boolean with `&`, `|` and `~` (for `not`), mixing with Python Booleans."""
    def __init__(self, value):
        self.value = bool(value)

    def __and__(self, other):
        return Bit(self.value and bool(other))
    __rand__ = __and__

    def __or__(self, other):
        return Bit(self.value or bool(other))
    __ror__ = __or__

    def __invert__(self):
        return Bit(not self.value)

    def __bool__(self):
        return self.value


def _translate(expression: str):
    """`f(i)` -> `Bit(T['f'][i])`, `1`/`0` -> `Bit(1)`/`Bit(0)`, `not ` -> `~` (a tight unary
    operator, as in math.js), and compile. Every operand is a `Bit`, so `~` is negation (on a
    Python bool, `~True` would be -2)."""
    expression = re.sub(r"([A-Za-z0-9_?]+)\s*\(\s*([ij])\s*\)", r"Bit(T['\1'][\2])", expression)
    expression = re.sub(r"\b([01])\b", r"Bit(\1)", expression)
    expression = expression.replace("not ", "~")
    return compile(expression, "<brasp>", "eval")


def _name(operation) -> str:
    return "Q_" + operation[1] if operation[0] == "init" else operation[1]


def run(operations, word: str) -> dict[str, list[bool]]:
    """The trace of every operation on `word` (positions 0..n-1, as in the simulator)."""
    n, T = len(word), {}
    for operation in operations:
        kind = operation[0]
        if kind == "init":
            T[_name(operation)] = [c == operation[1] for c in word]
        elif kind == "bool":
            code = _translate(operation[2])
            T[operation[1]] = [bool(eval(code, {"T": T, "Bit": Bit, "i": i})) for i in range(n)]
        else:
            _, name, tie, mask, score, value, default = operation
            score = _translate(f"({score}) & ({'1' if mask == '1' else mask})")
            value, default = _translate(value), _translate(default)
            out = []
            for i in range(n):
                matches = [j for j in range(n) if eval(score, {"T": T, "Bit": Bit, "i": i, "j": j})]
                if not matches:
                    out.append(bool(eval(default, {"T": T, "Bit": Bit, "i": i})))
                else:
                    j = matches[0] if tie == "leftmost" else matches[-1]
                    out.append(bool(eval(value, {"T": T, "Bit": Bit, "i": i, "j": j})))
            T[name] = out
    return T


def accepts(operations, word: str) -> bool:
    """Whether the last operation is 1 at the last position."""
    return len(word) > 0 and run(operations, word)[_name(operations[-1])][-1]
