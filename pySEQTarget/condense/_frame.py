import copy

import numpy as np
import polars as pl

from ..analysis._outcome_fit import _cast_categories
from ..expansion._diagnostics import _diagnostics
from ..helpers._fix_categories import _cast_fixed_categories
from ..weighting import _weight_bind, _weight_predict, _weight_setup
from ..weighting._weight_stats import _weight_bounds

# what fit leaves for _weight_predict
_WEIGHT_MODELS = (
    "numerator_model",
    "denominator_model",
    "cense_numerator_model",
    "cense_denominator_model",
    "visit_numerator_model",
    "visit_denominator_model",
    "_is_binary_treatment",
)


def _expanded(self, record):
    b = copy.copy(self)
    b.data = record
    b.verbose = False
    b.condense = False
    # clear a leftover bootstrap index
    b._current_boot_idx = None
    # no random trial selection per batch
    b.selection_random = False
    # keep batch levels off the shared dict
    b._covariate_categories = dict(getattr(self, "_covariate_categories", {}))
    b.expand()
    if self.selection_random:
        # hash keeps the same control trials in every pass
        tx_bas = f"{self.treatment_col}{self.indicator_baseline}"
        u = pl.struct([self.id_col, "trial"]).hash(self.seed) / 2.0**64
        b.DT = b.DT.filter(
            (pl.col(tx_bas) != self.treatment_level[0]) | (u < self.selection_sample)
        )
        # counted over the kept trials, as the package does
        _diagnostics(b)
    return b


def _recorded(self, record):
    b = copy.copy(self)
    b.data = record
    b.DT = None
    b.verbose = False
    b._current_boot_idx = None
    return b


def _weight_rows(self, b):
    WDT = _weight_setup(b)
    if not self.weight_preexpansion and not self.excused:
        WDT = WDT.filter(pl.col("followup") > 0)
    return WDT


def _bind_weights(self, b):
    for name in _WEIGHT_MODELS:
        if hasattr(self, name):
            setattr(b, name, getattr(self, name))
    _weight_bind(b, _weight_predict(b, _weight_rows(self, b)))


def _record_frame(self, b):
    return _cast_fixed_categories(self.fixed_cols, _weight_rows(self, b).to_pandas())


def _outcome_frame(self, b):
    df = b.DT
    if self.weighted:
        lower, upper = _weight_bounds(self)
        df = df.with_columns(
            pl.col("weight").clip(lower_bound=lower, upper_bound=upper)
        )
    if self.method == "censoring":
        df = df.filter(pl.col("switch") != 1)
    return _cast_categories(b, df.to_pandas())


def _segments(ids):
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    seg = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(ids)]))
    return seg, starts
