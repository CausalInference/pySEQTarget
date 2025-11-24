def _col_string(expressions):
    cols = set()
    for expression in expressions:
        if expression is not None:
            cols.update(expression.replace("+", " ").replace("*", " ").split())
    return cols
