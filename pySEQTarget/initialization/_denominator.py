from ._time_terms import _time_terms


def _denominator(self) -> str:
    if self.method == "ITT":
        return
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
        if self.method == "dose-response":
            out = "+".join(filter(None, [fixed, tv, time]))
        elif self.method == "censoring" and not self.excused:
            out = "+".join(filter(None, [fixed, tv, time]))
        elif self.method == "censoring" and self.excused:
            out = "+".join(filter(None, [fixed, tv, time]))
    else:
        if self.method == "dose-response":
            out = "+".join(filter(None, [fixed, tv, tv_bas, followup, trial]))
        elif self.method == "censoring" and not self.excused:
            out = "+".join(filter(None, [fixed, tv, tv_bas, followup, trial]))
        elif self.method == "censoring" and self.excused:
            out = "+".join(filter(None, [fixed, tv, tv_bas, followup, trial]))

    return out
