def _numerator(self) -> str:
    if self.method == "ITT":
        return
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
        if self.method == "dose-response":
            out = "+".join(filter(None, [fixed, time]))
        elif self.method == "censoring" and not self.excused:
            out = "+".join(filter(None, [fixed, time]))
        elif self.method == "censoring" and self.excused:
            out = None
    else:
        if self.method == "dose-response":
            out = "+".join(filter(None, [fixed, tv_bas, followup, trial]))
        elif self.method == "censoring" and not self.excused:
            out = "+".join(filter(None, [fixed, tv_bas, followup, trial]))
        elif self.method == "censoring" and self.excused:
            out = "+".join(filter(None, [fixed, tv_bas, followup, trial]))
    return out
