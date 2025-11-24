def _outcome(self) -> str:
    tx_bas = f"{self.treatment_col}{self.indicator_baseline}"
    dose = "+".join(["dose", f"dose{self.indicator_squared}"])
    interaction = f"{tx_bas}*followup"
    interaction_dose = "+".join(
        ["followup*dose", f"followup*dose{self.indicator_squared}"]
    )

    if self.hazard or not self.km_curves:
        interaction = interaction_dose = None

    tv_bas = (
        "+".join([f"{v}_bas" for v in self.time_varying_cols])
        if self.time_varying_cols
        else None
    )
    fixed = "+".join(self.fixed_cols) if self.fixed_cols else None
    trial = (
        "+".join(["trial", f"trial{self.indicator_squared}"])
        if self.trial_include
        else None
    )

    if self.followup_include:
        followup = "+".join(["followup", f"followup{self.indicator_squared}"])
    elif (self.followup_spline or self.followup_class) and not self.followup_include:
        followup = "followup"
    else:
        followup = None

    if self.method == "ITT":
        parts = [tx_bas, followup, trial, fixed, tv_bas, interaction]
        return "+".join(filter(None, parts))

    if self.weighted:
        if self.weight_preexpansion:
            if self.method == "dose-response":
                parts = [dose, followup, trial, fixed, interaction_dose]
            elif self.method == "censoring":
                if self.excused:
                    parts = [tx_bas, followup, trial, interaction]
                else:
                    parts = [tx_bas, followup, trial, fixed, interaction]
        else:
            if self.method == "dose-response":
                parts = [dose, followup, trial, fixed, tv_bas, interaction_dose]
            elif self.method == "censoring":
                parts = [tx_bas, followup, trial, fixed, tv_bas, interaction]
        return "+".join(filter(None, parts))

    if self.method == "dose-response":
        parts = [dose, followup, trial, fixed, tv_bas, interaction_dose]
    elif self.method == "censoring":
        parts = [tx_bas, followup, trial, fixed, tv_bas, interaction]

    return "+".join(filter(None, parts))
