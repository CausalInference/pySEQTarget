import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

import matplotlib.figure
import polars as pl
from statsmodels.base.wrapper import ResultsWrapper

from .helpers import _build_md, _build_pdf
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
    :param time: Timings for every step of the process completed thus far
    :type time: dict or None
    :param diagnostic_tables: Diagnostic tables for unique and nonunique outcome events and treatment switches
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
    time: dict = None
    diagnostic_tables: dict = None

    def plot(self) -> None:
        """
        Prints the kaplan-meier graph
        """
        print(self.km_graph)

    def summary(
        self, type=Optional[Literal["numerator", "denominator", "outcome", "compevent"]]
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

        return [model.summary() for model in models]

    def retrieve_data(
        self,
        type=Optional[
            Literal[
                "km_data",
                "hazard",
                "risk_ratio",
                "risk_difference",
                "unique_outcomes",
                "nonunique_outcomes",
                "unique_switches",
                "nonunique_switches",
            ]
        ],
    ) -> pl.DataFrame:
        """
        Getter for data stored within ``SEQoutput``
        :param type: Data which you would like to access, ['km_data', 'hazard', 'risk_ratio', 'risk_difference', 'unique_outcomes', 'nonunique_outcomes', 'unique_switches', 'nonunique_switches']
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
                data = self.diagnostic_tables["unique_outcomes"]
            case "nonunique_outcomes":
                data = self.diagnostic_tables["nonunique_outcomes"]
            case "unique_switches":
                if self.diagnostic_tables.has_key("unique_switches"):
                    data = self.diagnostic_tables["unique_switches"]
                else:
                    data = None
            case "nonunique_switches":
                if self.diagnostic_tables.has_key("nonunique_switches"):
                    data = self.diagnostic_tables["nonunique_switches"]
                else:
                    data = None
            case _:
                data = self.km_data
        if data is None:
            raise ValueError("Data {type} was not created in the SEQuential process")
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
