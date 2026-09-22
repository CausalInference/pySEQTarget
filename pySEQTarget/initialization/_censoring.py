from ._time_terms import _time_terms


def _cense_numerator(self) -> str:
    trial = _time_terms(self, "trial") if self.trial_include else None
    followup = _time_terms(self, "followup") if self.followup_include else None
    time = _time_terms(self, self.time_col)
    tv_bas = (
        "+".join([f"{v}{self.indicator_baseline}" for v in self.time_varying_cols])
        if self.time_varying_cols
        else None
    )
    fixed = "+".join(self.fixed_cols) if self.fixed_cols else None

    if self.weight_preexpansion:
        out = "+".join(filter(None, ["tx_lag", time, fixed]))
    else:
        out = "+".join(filter(None, ["tx_lag", trial, followup, fixed, tv_bas]))

    return out


def _cense_denominator(self) -> str:
    trial = _time_terms(self, "trial") if self.trial_include else None
    followup = _time_terms(self, "followup") if self.followup_include else None
    time = _time_terms(self, self.time_col)
    tv = "+".join(self.time_varying_cols) if self.time_varying_cols else None
    tv_bas = (
        "+".join([f"{v}{self.indicator_baseline}" for v in self.time_varying_cols])
        if self.time_varying_cols
        else None
    )
    fixed = "+".join(self.fixed_cols) if self.fixed_cols else None

    if self.weight_preexpansion:
        out = "+".join(filter(None, ["tx_lag", time, fixed, tv]))
    else:
        out = "+".join(filter(None, ["tx_lag", trial, followup, fixed, tv, tv_bas]))

    return out
