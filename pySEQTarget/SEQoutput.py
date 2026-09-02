import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

import matplotlib.figure
import matplotlib.pyplot as plt
import polars as pl
from statsmodels.base.wrapper import ResultsWrapper

from .helpers import Offloader, _build_md, _build_pdf
from .SEQopts import SEQopts


@dataclass
class SEQoutput:
    """
    Collector class for results from ``SEQuential``

    :param options: Options used in the SEQuential process
    :type options: SEQopts or None
    :param method: Method of analysis ['ITT', 'dose-response', or 'censoring']
    :type method: str
    :param numerator_models: Numerator models, if applicable, from the weighting process
    :type numerator_models: List[ResultsWrapper] or None
    :param denominator_models: Denominator models, if applicable, from the weighting process
    :type denominator_models: List[ResultsWrapper] or None
    :param compevent_models: Competing event models, if applicable
    :type compevent_models: List[ResultsWrapper] or None
    :param weight_statistics: Weight statistics once returned back to the expanded dataset
    :type weight_statistics: dict or None
    :param hazard: Hazard ratio if applicable
    :type hazard: pl.DataFrame or None
    :param km_data: Dataframe of risk, survival, and incidence data if applicable at all followups
    :type km_data: pl.DataFrame or None
    :param km_graph: Figure of survival, risk, or incidence over followup times
    :type km_graph: matplotlib.figure.Figure or None
    :param risk_ratio: Dataframe of risk ratios, compared between treatments and subgroups
    :type risk_ratio: pl.DataFrame or None
    :param risk_difference: Dataframe of risk differences, compared between treatments and subgroups
    :type risk_difference: pl.DataFrame or None
    :param eof_data: Per-arm end-of-follow-up estimates (``end_of_fup=True``): the
        weighted proportion (binary) or mean (continuous) read at
        ``end_of_fup_time``, with bootstrap confidence intervals when
        bootstrapped, the eligible trial-periods partitioned into analysed,
        censored (measured, but not within the window) and never measured, the
        censoring share, and the distinct contributing subjects
    :type eof_data: pl.DataFrame or None
    :param eof_comparison: Pairwise between-arm end-of-follow-up contrasts: the
        difference in proportions/means with its bootstrap SE and confidence
        interval (paired by iteration), plus — for a binary outcome only — the
        ratio of proportions with a log-scale interval and ``log(Ratio) SE``
    :type eof_comparison: pl.DataFrame or None
    :param time: Timings for every step of the process completed thus far
    :type time: dict or None
    :param diagnostic_tables: Diagnostic tables (outcome, follow-up, switch, and
        competing-event counts where applicable), each split by baseline treatment
        arm. The "unique" tables count distinct subjects; the "nonunique" tables
        count rows: total outcome events for the outcome tables, and total
        person-time intervals (expanded follow-up rows) for the follow-up tables.
        For a one-time (terminal) outcome the unique and nonunique outcome counts
        coincide, since each subject contributes at most one event row.
    :type diagnostic_tables: dict or None
    """

    options: SEQopts = None
    method: str = None
    numerator_models: List[ResultsWrapper] = None
    denominator_models: List[ResultsWrapper] = None
    outcome_models: List[List[ResultsWrapper]] = None
    compevent_models: List[List[ResultsWrapper]] = None
    weight_statistics: pl.DataFrame = None
    hazard: pl.DataFrame = None
    km_data: pl.DataFrame = None
    km_graph: matplotlib.figure.Figure = None
    risk_ratio: pl.DataFrame = None
    risk_difference: pl.DataFrame = None
    eof_data: pl.DataFrame = None
    eof_comparison: pl.DataFrame = None
    time: dict = None
    diagnostic_tables: dict = None

    def plot(self) -> None:
        """
        Displays the Kaplan-Meier graph
        """
        if self.km_graph is None:
            raise ValueError(
                "No plot available. Ensure km_curves=True and run SEQuential.plot() before collect()."
            )
        plt.figure(self.km_graph)
        plt.show()

    def summary(
        self,
        type: Optional[
            Literal["numerator", "denominator", "outcome", "compevent"]
        ] = None,
    ) -> List:
        """
        Returns a list of model summaries of either the numerator, denominator, outcome, or competing event models

        :param type: Indicator for which model list you would like returned
        :type type: str
        """
        match type:
            case "numerator":
                models = self.numerator_models
            case "denominator":
                models = self.denominator_models
            case "compevent":
                models = self.compevent_models
            case _:
                models = self.outcome_models

        if models is None:
            return []

        # Under offload=True the stored entries are path refs; load them back.
        loader = None
        if self.options is not None and self.options.offload:
            loader = Offloader(enabled=True, dir=self.options.offload_dir)

        summaries = []
        for model in models:
            if model is None:
                continue
            if loader is not None:
                model = loader.load_model(model)
            summaries.append(model.summary())
        return summaries

    def retrieve_data(
        self,
        type: Optional[
            Literal[
                "km_data",
                "hazard",
                "risk_ratio",
                "risk_difference",
                "unique_outcomes",
                "nonunique_outcomes",
                "unique_followup",
                "nonunique_followup",
                "unique_compevent",
                "nonunique_compevent",
                "unique_switches",
                "nonunique_switches",
                "eof_data",
                "eof_comparison",
                "unique_eof",
                "nonunique_eof",
                "eof_summary",
            ]
        ] = None,
    ) -> pl.DataFrame:
        """
        Getter for data stored within ``SEQoutput``

        The diagnostic tables come in "unique" and "nonunique" variants that count
        different things, each broken down by baseline treatment arm:

        - ``unique_outcomes`` / ``nonunique_outcomes``: distinct subjects who had
          the outcome vs. the total number of outcome events. These coincide for a
          one-time (terminal) outcome, since each subject contributes at most one
          event row.
        - ``unique_followup`` / ``nonunique_followup``: distinct subjects
          contributing follow-up vs. the total number of person-time intervals
          (expanded rows). The nonunique count is much larger because each subject
          contributes one row per follow-up period; it is the denominator that,
          with ``nonunique_outcomes``, gives the per-arm event rate.
        - ``unique_eof`` / ``nonunique_eof`` (``end_of_fup=True`` only): account
          for every trial-period at the end-of-follow-up time across four
          mutually exclusive categories — measured ``At k``, measured
          ``In window``, ``Excluded (outside window)`` and ``Excluded (no
          measurement)`` — against the ``Eligible`` total. The nonunique
          (trial-period) counts partition ``Eligible``; the unique (subject)
          counts may overlap, since one subject can fall into different
          categories for different trials.
        - ``eof_summary`` (continuous ``end_of_fup`` only): N/Mean/SD of the
          analysed measurements per arm, standing in for the suppressed outcome
          count tables.

        :param type: Data which you would like to access, ['km_data', 'hazard',
            'risk_ratio', 'risk_difference', 'unique_outcomes',
            'nonunique_outcomes', 'unique_followup', 'nonunique_followup',
            'unique_compevent', 'nonunique_compevent',
            'unique_switches', 'nonunique_switches',
            'eof_data', 'eof_comparison', 'unique_eof', 'nonunique_eof',
            'eof_summary']
        :type type: str
        """
        match type:
            case "hazard":
                data = self.hazard
            case "risk_ratio":
                data = self.risk_ratio
            case "risk_difference":
                data = self.risk_difference
            case "unique_outcomes":
                # Absent for continuous end-of-follow-up outcomes
                data = self.diagnostic_tables.get("unique_outcomes")
            case "nonunique_outcomes":
                data = self.diagnostic_tables.get("nonunique_outcomes")
            case "unique_followup":
                data = self.diagnostic_tables["unique_followup"]
            case "nonunique_followup":
                data = self.diagnostic_tables["nonunique_followup"]
            case "unique_compevent":
                data = self.diagnostic_tables.get("unique_compevent")
            case "nonunique_compevent":
                data = self.diagnostic_tables.get("nonunique_compevent")
            case "unique_switches":
                data = self.diagnostic_tables.get("unique_switches")
            case "nonunique_switches":
                data = self.diagnostic_tables.get("nonunique_switches")
            case "eof_data":
                data = self.eof_data
            case "eof_comparison":
                data = self.eof_comparison
            case "unique_eof":
                data = self.diagnostic_tables.get("unique_eof")
            case "nonunique_eof":
                data = self.diagnostic_tables.get("nonunique_eof")
            case "eof_summary":
                data = self.diagnostic_tables.get("eof_summary")
            case _:
                data = self.km_data
        if data is None:
            raise ValueError(f"Data {type} was not created in the SEQuential process")
        return data

    def to_md(self, filename="SEQuential_results.md") -> None:
        """Generates a markdown report of the SEQuential analysis results."""

        img_path = None
        if self.options.km_curves and self.km_graph is not None:
            img_path = Path(filename).with_suffix(".png")
            self.km_graph.savefig(img_path, dpi=300, bbox_inches="tight")
            img_path = img_path.name

        with open(filename, "w") as f:
            f.write(_build_md(self, img_path))

        print(f"Results saved to {filename}")

    def to_pdf(self, filename="SEQuential_results.pdf") -> None:
        """Generates a PDF report of the SEQuential analysis results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_md = Path(tmpdir) / "report.md"
            self.to_md(str(tmp_md))

            with open(tmp_md, "r") as f:
                md_content = f.read()

            tmp_img = tmp_md.with_suffix(".png")
            img_abs_path = str(tmp_img.absolute()) if tmp_img.exists() else None

            _build_pdf(md_content, filename, img_abs_path)

        print(f"Results saved to {filename}")
