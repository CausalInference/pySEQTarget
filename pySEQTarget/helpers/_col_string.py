import ast


def _term_columns(expression):
    """
    The column names a patsy right-hand-side formula refers to.

    The formula is parsed as a Python expression so that function-call terms —
    ``cr(followup, df=4)``, ``cr(followup, knots=[...], lower_bound=0.0, ...)``,
    ``C(sex)`` — contribute the variables they wrap rather than fragments of
    the call itself, and so that a literal intercept (``"1"``) contributes
    nothing. Anything that will not parse (e.g. a column name that is not a
    valid Python identifier) falls back to splitting on the formula operators,
    as before.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return set(expression.replace("+", " ").replace("*", " ").split())

    called = {
        node.func
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
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
