import re

import numpy as np

# patsy's data-dependent spline form — the only one whose basis depends on the
# rows it happens to be built from, and so the only one worth baking.
_CR_DF = re.compile(
    r"""\bcr\(\s*(?P<var>[A-Za-z_.][A-Za-z0-9_.]*)\s*,\s*df\s*=\s*(?P<df>\d+)\s*
        (?:,\s*constraints\s*=\s*(?P<quote>["'])(?P<constraints>center)(?P=quote)\s*)?\)""",
    re.VERBOSE,
)


def _compute_spline_knots(arr, df=3):
    """
    Knot positions patsy's ``cr()`` would place over ``arr`` for ``df``
    degrees of freedom: ``df - 2`` interior knots at equally spaced percentiles
    of the unique values, plus the two boundary knots.
    """
    lower = float(np.min(arr))
    upper = float(np.max(arr))
    n_inner = df - 2
    if n_inner <= 0:
        inner_knots = []
    else:
        # Replicate patsy's knot placement: percentiles of unique values in [lower, upper]
        x = np.unique(arr[(lower <= arr) & (arr <= upper)])
        q = np.linspace(0, 100, n_inner + 2)[1:-1]
        inner_knots = np.percentile(x, q.tolist()).tolist()
    return inner_knots, lower, upper


def _cr_term(var, inner_knots, lower, upper, constraints=None):
    """A ``cr()`` term with its knots written out."""
    constraint = f', constraints="{constraints}"' if constraints else ""
    return (
        f"cr({var}, knots={list(inner_knots)}, "
        f"lower_bound={lower}, upper_bound={upper}{constraint})"
    )


def _bake_spline_knots(formula, data):
    """
    Rewrite every ``cr(x, df=N)`` term of ``formula`` to an explicit
    ``cr(x, knots=[...], lower_bound=..., upper_bound=...)``, with the knots
    taken once from the full ``x`` column of ``data`` — the percentiles patsy
    itself would use.

    ``formula`` may be ``None``, a single right-hand-side formula string, or a
    list of them (the per-treatment-level weight models), and is returned in
    the same shape. Terms whose variable is missing from ``data``, non-numeric
    or all-null are left alone, as are terms that already carry explicit knots.
    """
    if formula is None:
        return formula
    if isinstance(formula, (list, tuple)):
        return [_bake_spline_knots(part, data) for part in formula]

    tokens = {
        match.group(0): (
            match.group("var"),
            int(match.group("df")),
            match.group("constraints"),
        )
        for match in _CR_DF.finditer(formula)
    }
    if not tokens:
        return formula

    for token, (var, df, constraints) in tokens.items():
        if var not in data.columns:
            continue
        column = data[var]
        if not column.dtype.is_numeric():
            continue
        arr = column.drop_nulls().to_numpy()
        if arr.size == 0:
            continue

        # A centering constraint absorbs one degree of freedom, so patsy
        # places one more interior knot than the unconstrained form would
        inner_knots, lower, upper = _compute_spline_knots(
            arr, df=df + 1 if constraints else df
        )
        # Too few distinct values ties knots to each other or to a boundary;
        # patsy rejects that basis, without naming the term that caused it.
        if len(set(inner_knots)) < len(inner_knots) or not all(
            lower < knot < upper for knot in inner_knots
        ):
            raise ValueError(
                f"'{var}' has too few distinct values to place "
                f"{len(inner_knots)} interior knot(s) for cr(df={df}); use a "
                "smaller df, or model this term without a spline."
            )
        formula = formula.replace(
            token, _cr_term(var, inner_knots, lower, upper, constraints)
        )

    return formula


def _bake_model_formulas(self):
    """
    Fix the knots of every ``cr(x, df=N)`` term in the model formulas, taking
    them from the full data each model is fit on.

    patsy's ``cr()`` is a stateful transform, so a fitted model already reuses
    its own fit-time knots when predicting. What explicit knots add is a basis
    that is identical across bootstrap resamples — each of which would
    otherwise memorise knots from its own rows — and one that survives the
    formula re-parse the glum backend performs when a fitted model is unpickled
    for the offload and parallel paths.
    """
    if not self.followup_spline:
        # followup_spline rewrites the outcome's followup terms (with fixed
        # knots) at fit time, and would mangle a cr() term baked in here.
        self.covariates = _bake_spline_knots(self.covariates, self.DT)

    if self.weighted:
        weight_data = self.data if self.weight_preexpansion else self.DT
        for name in (
            "numerator",
            "denominator",
            "cense_numerator",
            "cense_denominator",
        ):
            setattr(self, name, _bake_spline_knots(getattr(self, name), weight_data))
