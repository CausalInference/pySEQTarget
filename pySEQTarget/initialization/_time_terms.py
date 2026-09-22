def _time_terms(self, var: str) -> str:
    """
    Default weight-model terms for a time axis ``var``: a quadratic
    (``var``, ``var_sq``), or a centred ``cr()`` spline of ``weight_spline_df``
    df when ``weight_spline`` is set. Knots are fixed later from the fit data.
    """
    if self.weight_spline:
        return f'cr({var}, df={self.weight_spline_df}, constraints="center")'
    return "+".join([var, f"{var}{self.indicator_squared}"])
