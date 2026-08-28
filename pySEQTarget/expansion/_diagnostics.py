import polars as pl


def _diagnostics(self):
    # Outcome tables count outcome == 1 rows — meaningless for a continuous
    # end-of-follow-up outcome, so they are suppressed there.
    out = {}
    if not (self.end_of_fup and self.end_of_fup_type == "continuous"):
        out["unique_outcomes"] = _outcome_diag(self, unique=True)
        out["nonunique_outcomes"] = _outcome_diag(self, unique=False)

    unique_fu = _followup_diag(self, unique=True)
    nonunique_fu = _followup_diag(self, unique=False)
    out.update({"unique_followup": unique_fu, "nonunique_followup": nonunique_fu})

    if self.compevent_colname is not None:
        unique_ce = _compevent_diag(self, unique=True)
        nonunique_ce = _compevent_diag(self, unique=False)
        out.update({"unique_compevent": unique_ce, "nonunique_compevent": nonunique_ce})

    if self.method == "censoring":
        unique_switch = _switch_diag(self, unique=True)
        nonunique_switch = _switch_diag(self, unique=False)
        out.update(
            {"unique_switches": unique_switch, "nonunique_switches": nonunique_switch}
        )

    self.diagnostics = out


def _outcome_diag(self, unique):
    if unique:
        data = (
            self.DT.select([self.id_col, self.treatment_col, self.outcome_col])
            .group_by(self.id_col)
            .last()
        )
    else:
        data = self.DT
    out = data.group_by([self.treatment_col, self.outcome_col]).len()

    return out


def _followup_diag(self, unique):
    """
    Follow-up per treatment arm, grouped by the baseline treatment value, over
    the rows the outcome model is fit on (under method="censoring" the switched
    rows are dropped, matching _outcome_fit). ``unique`` counts distinct subjects
    contributing follow-up to the arm; otherwise counts follow-up intervals
    (rows / person-time), so non-unique outcome counts divided by these give
    per-arm event rates. A subject appearing in both arms is counted in each.
    """
    tx_bas = f"{self.treatment_col}{self.indicator_baseline}"
    data = self.DT
    if self.method == "censoring":
        data = data.filter(pl.col("switch") != 1)

    if unique:
        out = data.group_by(tx_bas).agg(pl.col(self.id_col).n_unique().alias("len"))
    else:
        out = data.group_by(tx_bas).len()

    return out.sort(tx_bas)


def _compevent_diag(self, unique):
    """
    Competing-event counts per treatment arm. Counts rows where the configured
    compevent column is 1, grouped by the baseline treatment value. ``unique``
    counts distinct subjects who experience the competing event in each arm;
    otherwise counts the intervals (rows). A subject appearing with a competing
    event in both arms is counted in each.
    """
    tx_bas = f"{self.treatment_col}{self.indicator_baseline}"
    data = self.DT.filter(pl.col(self.compevent_colname) == 1)
    if unique:
        out = data.group_by(tx_bas).agg(pl.col(self.id_col).n_unique().alias("len"))
    else:
        out = data.group_by(tx_bas).len()
    return out.sort(tx_bas)


def _switch_diag(self, unique):
    if not self.excused:
        data = self.DT.with_columns(pl.lit(False).alias("isExcused"))
    else:
        data = self.DT

    if unique:
        data = (
            data.select([self.id_col, self.treatment_col, "switch", "isExcused"])
            .with_columns(
                pl.when((pl.col("switch") == 0) & (pl.col("isExcused")))
                .then(1)
                .otherwise(pl.col("switch"))
                .alias("switch")
            )
            .group_by(self.id_col)
            .last()
        )

    out = data.group_by([self.treatment_col, "isExcused", "switch"]).len()
    return out
