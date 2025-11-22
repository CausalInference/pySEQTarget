def _pad(a, b):
    len_a, len_b = len(a), len(b)
    if len_a < len_b:
        a = a + [None] * (len_b - len_a)
    elif len_b < len_a:
        b = b + [None] * (len_a - len_b)
    return a, b
