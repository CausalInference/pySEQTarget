import warnings

import polars as pl

from ._outcome_fit import _outcome_fit


def _warn_missing_subgroup(self, val):
    """Warn when bootstrap replicates have no rows in subgroup ``val``."""
    n = sum(val not in models for models in self.outcome_model[1:])
    if n:
        warnings.warn(
            f"{n} bootstrap replicate(s) had no rows in {self.subgroup_colname} = "
            f"{val}; they are left out of its confidence intervals.",
            UserWarning,
            stacklevel=3,
        )


def _subgroup_fit(self, start_params=None):
    # A resample can miss a subgroup, so key the models by subgroup value
    subgroups = sorted(self.DT[self.subgroup_colname].unique().to_list())
    if getattr(self, "_current_boot_idx", None) is None:
        self._unique_subgroups = subgroups

    models_by_group = {}
    for val in subgroups:
        subDT = self.DT.filter(pl.col(self.subgroup_colname) == val)
        sg_start = (start_params or {}).get(val, {}) or {}

        models = {
            "outcome": _outcome_fit(
                self,
                subDT,
                self.outcome_col,
                self.covariates,
                self.weighted,
                "weight",
                start_params=sg_start.get("outcome"),
            )
        }

        if self.compevent_colname is not None:
            models["compevent"] = _outcome_fit(
                self,
                subDT,
                self.compevent_colname,
                self.covariates,
                self.weighted,
                "weight",
                start_params=sg_start.get("compevent"),
            )
        models_by_group[val] = models
    return models_by_group
