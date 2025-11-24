def _cense_numerator(self) -> str:
    trial = (
        "+".join(["trial", f"trial{self.indicator_squared}"])
        if self.trial_include
        else None
    )
    followup = (
        "+".join(["followup", f"followup{self.indicator_squared}"])
        if self.followup_include
        else None
    )
    time = "+".join([self.time_col, f"{self.time_col}{self.indicator_squared}"])
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
    trial = (
        "+".join(["trial", f"trial{self.indicator_squared}"])
        if self.trial_include
        else None
    )
    followup = (
        "+".join(["followup", f"followup{self.indicator_squared}"])
        if self.followup_include
        else None
    )
    time = "+".join([self.time_col, f"{self.time_col}{self.indicator_squared}"])
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
