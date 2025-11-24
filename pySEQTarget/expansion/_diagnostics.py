import polars as pl


def _diagnostics(self):
    unique_out = _outcome_diag(self, unique=True)
    nonunique_out = _outcome_diag(self, unique=False)
    out = {"unique_outcomes": unique_out, "nonunique_outcomes": nonunique_out}

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
