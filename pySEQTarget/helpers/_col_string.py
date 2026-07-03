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
                cols.update(part.replace("+", " ").replace("*", " ").split())
    return cols
