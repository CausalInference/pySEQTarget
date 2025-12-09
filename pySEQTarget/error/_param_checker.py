from ..helpers import _pad


def _param_checker(self):
    if (
        self.subgroup_colname is not None
        and self.subgroup_colname not in self.fixed_cols
    ):
        raise ValueError("subgroup_colname must be included in fixed_cols.")

    if self.followup_max is None:
        self.followup_max = self.data.select(self.time_col).to_series().max()

    if len(self.excused_colnames) == 0 and self.excused:
        self.excused = False
        raise Warning(
            "Excused column names not provided but excused is set to True. Automatically set excused to False"
        )

    if len(self.excused_colnames) > 0 and not self.excused:
        self.excused = True
        raise Warning(
            "Excused column names provided but excused is set to False. Automatically set excused to True"
        )

    if self.km_curves and self.hazard_estimate:
        raise ValueError("km_curves and hazard cannot both be set to True.")

    if sum([self.followup_class, self.followup_include, self.followup_spline]) > 1:
        raise ValueError(
            "Only one of followup_class or followup_include can be set to True."
        )

    if (
        self.weighted
        and self.method == "ITT"
        and self.cense_colname is None
        and self.visit_colname is None
    ):
        raise ValueError(
            "For weighted ITT analyses, cense_colname or visit_colname must be provided."
        )

    if self.excused:
        _, self.excused_colnames = _pad(self.treatment_level, self.excused_colnames)
    _, self.weight_eligible_colnames = _pad(
        self.treatment_level, self.weight_eligible_colnames
    )

    return
