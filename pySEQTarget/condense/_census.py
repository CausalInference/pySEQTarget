import numpy as np
import patsy
import polars as pl
from pandas.api.types import CategoricalDtype

from ..analysis._outcome_fit import _apply_spline_formula, _cast_categories
from ..helpers._spline import _CR_DF, _compute_spline_knots, _cr_term
from ._eof import _eof_last_period
from ._frame import _expanded, _record_frame

_WEIGHT_FORMULAS = ("numerator", "denominator", "cense_numerator", "cense_denominator")


def _tokens(formula):
    if formula is None:
        return {}
    if isinstance(formula, (list, tuple)):
        return {k: v for part in formula for k, v in _tokens(part).items()}
    return {
        m.group(0): (m.group("var"), int(m.group("df")), m.group("constraints"))
        for m in _CR_DF.finditer(formula)
    }


def _count(acc, frame, names):
    for var in names:
        if var not in frame.columns or not frame[var].dtype.is_numeric():
            continue
        values, counts = np.unique(
            frame[var].drop_nulls().to_numpy(), return_counts=True
        )
        acc.setdefault(var, []).append((values, counts))


def _merged(parts):
    values, inverse = np.unique(
        np.concatenate([v for v, _ in parts]), return_inverse=True
    )
    return values, np.bincount(inverse, weights=np.concatenate([c for _, c in parts]))


def _levels(acc, frame):
    for col in frame.columns:
        if isinstance(frame[col].dtype, CategoricalDtype):
            acc[col] = sorted(set(acc.get(col, [])) | set(frame[col].cat.categories))


def _bake(formula, values):
    if formula is None:
        return formula
    if isinstance(formula, (list, tuple)):
        return [_bake(part, values) for part in formula]
    for token, (var, df, constraints) in _tokens(formula).items():
        if var not in values:
            continue
        v, n = values[var]
        inner_knots, lower, upper = _compute_spline_knots(
            v, df=df + 1 if constraints else df
        )
        if len(set(inner_knots)) < len(inner_knots) or not all(
            lower < knot < upper for knot in inner_knots
        ):
            raise ValueError(
                f"'{var}' has too few distinct values to place "
                f"{len(inner_knots)} interior knot(s) for cr(df={df}); use a "
                "smaller df, or model this term without a spline."
            )
        if constraints:
            basis = patsy.dmatrix(
                _cr_term("x", inner_knots, lower, upper) + "-1", {"x": v}
            )
            constraints = (n @ np.asarray(basis) / n.sum()).tolist()
        formula = formula.replace(
            token, _cr_term(var, inner_knots, lower, upper, constraints)
        )
    return formula


class _Census:
    def __init__(self, self_, store, people_per_batch):
        weighted = self_.weighted
        record_vars = (
            {
                var
                for name in _WEIGHT_FORMULAS
                for var, _, _ in _tokens(getattr(self_, name)).values()
            }
            if weighted
            else set()
        )
        if self_.followup_spline:
            outcome_vars = {"followup"}
        else:
            outcome_vars = {var for var, _, _ in _tokens(self_.covariates).values()}

        self.subgroups = set()
        self.last_period = None
        self.levels = {"record": {}, "outcome": {}}
        values = {"record": {}, "outcome": {}}
        self.n_people = 0
        for record in store.batches(people_per_batch):
            b = _expanded(self_, record)
            self.n_people += record[self_.id_col].n_unique()
            # knots come from the data the weight models are fit on
            _count(
                values["record"],
                b.data if self_.weight_preexpansion else b.DT,
                record_vars,
            )
            _count(values["outcome"], b.DT, outcome_vars)
            if self_.subgroup_colname is not None:
                self.subgroups |= set(b.DT[self_.subgroup_colname].unique().to_list())
            if self_.end_of_fup:
                last = _eof_last_period(self_, b)
                if last is not None:
                    self.last_period = max(last, self.last_period or last)
            df = b.DT
            if self_.method == "censoring":
                df = df.filter(pl.col("switch") != 1)
            _levels(self.levels["outcome"], _cast_categories(b, df.to_pandas()))
            if weighted:
                _levels(self.levels["record"], _record_frame(self_, b))

        self.subgroups = sorted(self.subgroups)
        values = {
            frame: {var: _merged(parts) for var, parts in found.items()}
            for frame, found in values.items()
        }
        self.formulas = {
            name: _bake(getattr(self_, name), values["record"])
            for name in _WEIGHT_FORMULAS
        }
        if self_.followup_spline:
            self.followup_knots = _compute_spline_knots(
                values["outcome"]["followup"][0], df=self_.followup_spline_df
            )
            self.covariates = _apply_spline_formula(
                self_.covariates, self_.indicator_squared, self.followup_knots
            )
        else:
            self.covariates = _bake(self_.covariates, values["outcome"])
