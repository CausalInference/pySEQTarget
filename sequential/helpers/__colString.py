def __colString(expressions):
    cols = set()
    for expression in expressions:
        cols.update(expression.replace("+", " ").replace("*", " ").split())
    
    return cols