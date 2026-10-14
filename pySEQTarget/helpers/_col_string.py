import ast


def _term_columns(expression):
    """Column names a patsy right-hand-side formula refers to (function-call
    terms like ``cr(followup, df=4)`` contribute the variable they wrap)."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return set(expression.replace("+", " ").replace("*", " ").split())

    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # The function, or the module it comes from, as in np.log(x)
            func = node.func
            while isinstance(func, ast.Attribute):
                func = func.value
            called.add(func)
    return {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node not in called
    }


def _col_string(expressions):
    cols = set()
    for expression in expressions:
        if expression is None:
            continue
        # numerator/denominator may be a list of per-treatment-level formulas;
        # gather the referenced columns across every element.
        parts = expression if isinstance(expression, (list, tuple)) else [expression]
        for part in parts:
            if part is not None:
                cols.update(_term_columns(part))
    return cols
