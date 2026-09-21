def _time_terms(self, var: str) -> str:
    """
    The terms a default weight model uses for a time axis ``var``.

    By default this is a quadratic (``var`` and ``var_sq``), which only lets
    the baseline hazard of treatment rise or flatten off. With
    ``weight_spline`` it becomes a natural cubic spline basis of
    ``weight_spline_df`` degrees of freedom instead, so the hazard can take a
    flexible shape over time.

    The spline is emitted in patsy's ``df=`` form; ``SEQuential.fit()`` fixes
    its knots from the data the weight models are fit on, so the basis is the
    same on the main fit and on every bootstrap resample. It is centred, since
    an unconstrained ``cr()`` basis spans the constant function and so is
    collinear with the model intercept by exactly one dimension.
    """
    if self.weight_spline:
        return f'cr({var}, df={self.weight_spline_df}, constraints="center")'
    return "+".join([var, f"{var}{self.indicator_squared}"])
