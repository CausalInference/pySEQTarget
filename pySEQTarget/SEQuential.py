import datetime
import time
from collections import Counter
from dataclasses import asdict
from typing import List, Literal, Optional

import numpy as np
import polars as pl

from .analysis import (_calculate_hazard, _calculate_survival, _clamp,
                       _outcome_fit, _pred_risk, _risk_estimates,
                       _subgroup_fit)
from .error import _data_checker, _param_checker
from .expansion import _binder, _diagnostics, _dynamic, _random_selection
from .helpers import _col_string, _format_time, bootstrap_loop
from .initialization import (_cense_denominator, _cense_numerator,
                             _denominator, _numerator, _outcome)
from .plot import _survival_plot
from .SEQopts import SEQopts
from .SEQoutput import SEQoutput
from .weighting import (_fit_denominator, _fit_LTFU, _fit_numerator,
                        _fit_visit, _weight_bind, _weight_predict,
                        _weight_setup, _weight_stats)


class SEQuential:
    """
    Primary class initializer for SEQuentially nested target trial emulation

    :param data: Data for analysis
    :type data: pl.DataFrame
    :param id_col: Column name for unique patient IDs
    :type id_col: str
    :param time_col: Column name for observational time points
    :type time_col: str
    :param eligible_col: Column name for analytical eligibility
    :type eligible_col: str
    :param treatment_col: Column name specifying treatment per time_col
    :type treatment_col: str
    :param outcome_col: Column name specifying outcome per time_col
    :type outcome_col: str
    :param time_varying_cols: Time-varying column names as covariates (BMI, Age, etc.)
    :type time_varying_cols: Optional[List[str]] or None
    :param fixed_cols: Fixed column names as covariates (Sex, YOB, etc.)
    :type fixed_cols: Optional[List[str]] or None
    :param method: Method for analysis ['ITT', 'dose-response', or 'censoring']
    :type method: str
    :param parameters: Parameters to augment analysis, specified with ``pySEQTarget.SEQopts``
    :type parameters: Optional[SEQopts] or None
    """

    def __init__(
        self,
        data: pl.DataFrame,
        id_col: str,
        time_col: str,
        eligible_col: str,
        treatment_col: str,
        outcome_col: str,
        time_varying_cols: Optional[List[str]] = None,
        fixed_cols: Optional[List[str]] = None,
        method: Literal["ITT", "dose-response", "censoring"] = "ITT",
        parameters: Optional[SEQopts] = None,
    ) -> None:
        self.data = data
        self.id_col = id_col
        self.time_col = time_col
        self.eligible_col = eligible_col
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.time_varying_cols = time_varying_cols
        self.fixed_cols = fixed_cols
        self.method = method

        self._time_initialized = datetime.datetime.now()

        if parameters is None:
            parameters = SEQopts()

        for name, value in asdict(parameters).items():
            setattr(self, name, value)

        self._rng = (
            np.random.RandomState(self.seed) if self.seed is not None else np.random
        )

        if self.covariates is None:
            self.covariates = _outcome(self)

        if self.weighted:
            if self.numerator is None:
                self.numerator = _numerator(self)

            if self.denominator is None:
                self.denominator = _denominator(self)

            if self.cense_colname is not None or self.visit_colname is not None:
                if self.cense_numerator is None:
                    self.cense_numerator = _cense_numerator(self)

                if self.cense_denominator is None:
                    self.cense_denominator = _cense_denominator(self)

        _param_checker(self)
        _data_checker(self)

    def expand(self) -> None:
        """
        Creates the sequentially nested, emulated target trial structure
        """
        start = time.perf_counter()
        kept = [
            self.cense_colname,
            self.cense_eligible_colname,
            self.compevent_colname,
            self.visit_colname,
            *self.weight_eligible_colnames,
            *self.excused_colnames,
        ]

        self.data = self.data.with_columns(
            [
                pl.when(pl.col(self.treatment_col).is_in(self.treatment_level))
                .then(self.eligible_col)
                .otherwise(0)
                .alias(self.eligible_col),
                pl.col(self.treatment_col).shift(1).over([self.id_col]).alias("tx_lag"),
                pl.lit(False).alias("switch"),
            ]
        ).with_columns(
            [
                pl.when(pl.col(self.time_col) == 0)
                .then(pl.lit(False))
                .otherwise(
                    (pl.col("tx_lag").is_not_null())
                    & (pl.col("tx_lag") != pl.col(self.treatment_col))
                )
                .cast(pl.Int8)
                .alias("switch")
            ]
        )

        self.DT = _binder(
            self,
            kept_cols=_col_string(
                [
                    self.covariates,
                    self.numerator,
                    self.denominator,
                    self.cense_numerator,
                    self.cense_denominator,
                ]
            ).union(kept),
        ).with_columns(pl.col(self.id_col).cast(pl.Utf8).alias(self.id_col))

        self.data = self.data.with_columns(
            pl.col(self.id_col).cast(pl.Utf8).alias(self.id_col)
        )

        if self.method != "ITT":
            _dynamic(self)
        if self.selection_random:
            _random_selection(self)
        _diagnostics(self)

        end = time.perf_counter()
        self._expansion_time = _format_time(start, end)

    def bootstrap(self, **kwargs) -> None:
        """
        Internally sets up bootstrapping - creating a list of IDs to use per iteration
        """
        allowed = {
            "bootstrap_nboot",
            "bootstrap_sample",
            "bootstrap_CI",
            "bootstrap_method",
        }
        for key, value in kwargs.items():
            if key in allowed:
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown argument: {key}")
        UIDs = self.DT.select(pl.col(self.id_col)).unique().to_series().to_list()
        NIDs = len(UIDs)

        self._boot_samples = []
        for _ in range(self.bootstrap_nboot):
            sampled_IDs = self._rng.choice(
                UIDs, size=int(self.bootstrap_sample * NIDs), replace=True
            )
            id_counts = Counter(sampled_IDs)
            self._boot_samples.append(id_counts)

    @bootstrap_loop
    def fit(self) -> None:
        """
        Fits weight models (numerator, denominator, censoring) and outcome models (outcome, competing event)
        """
        if self.bootstrap_nboot > 0 and not hasattr(self, "_boot_samples"):
            raise ValueError(
                "Bootstrap sampling not found. Please run the 'bootstrap' method before fitting with bootstrapping."
            )

        if self.weighted:
            WDT = _weight_setup(self)
            if not self.weight_preexpansion and not self.excused:
                WDT = WDT.filter(pl.col("followup") > 0)

            WDT = WDT.to_pandas()
            for col in self.fixed_cols:
                if col in WDT.columns:
                    WDT[col] = WDT[col].astype("category")

            _fit_LTFU(self, WDT)
            _fit_visit(self, WDT)
            _fit_numerator(self, WDT)
            _fit_denominator(self, WDT)

            WDT = pl.from_pandas(WDT)
            WDT = _weight_predict(self, WDT)
            _weight_bind(self, WDT)
            self.weight_stats = _weight_stats(self)

        if self.subgroup_colname is not None:
            return _subgroup_fit(self)

        models = {
            "outcome": _outcome_fit(
                self,
                self.DT,
                self.outcome_col,
                self.covariates,
                self.weighted,
                "weight",
            )
        }
        if self.compevent_colname is not None:
            models["compevent"] = _outcome_fit(
                self,
                self.DT,
                self.compevent_colname,
                self.covariates,
                self.weighted,
                "weight",
            )
        return models

    def survival(self, **kwargs) -> None:
        """
        Uses fit outcome models (outcome, competing event) to estimate risk, survival, and incidence curves
        """
        allowed = {"bootstrap_CI", "bootstrap_CI_method"}
        for key, val in kwargs.items():
            if key in allowed:
                setattr(self, key, val)
            else:
                raise ValueError(f"Unknown or misplaced arugment: {key}")

        if not hasattr(self, "outcome_model") or not self.outcome_model:
            raise ValueError(
                "Outcome model not found. Please run the 'fit' method before calculating survival."
            )

        start = time.perf_counter()

        risk_data = _pred_risk(self)
        surv_data = _calculate_survival(self, risk_data)
        self.km_data = _clamp(pl.concat([risk_data, surv_data]))
        self.risk_estimates = _risk_estimates(self)

        end = time.perf_counter()
        self._survival_time = _format_time(start, end)

    def hazard(self) -> None:
        """
        Uses fit outcome models (outcome, competing event) to estimate hazard ratios
        """
        start = time.perf_counter()

        if not hasattr(self, "outcome_model") or not self.outcome_model:
            raise ValueError(
                "Outcome model not found. Please run the 'fit' method before calculating hazard ratio."
            )
        self.hazard_ratio = _calculate_hazard(self)

        end = time.perf_counter()
        self._hazard_time = _format_time(start, end)

    def plot(self, **kwargs) -> None:
        """
        Shows a plot specific to plot_type
        """
        allowed = {"plot_type", "plot_colors", "plot_title", "plot_labels"}
        for key, val in kwargs.items():
            if key in allowed:
                setattr(self, key, val)
            else:
                raise ValueError(f"Unknown or misplaced arugment: {key}")
        self.km_graph = _survival_plot(self)

    def collect(self) -> SEQoutput:
        """
        Collects all results current created into ``SEQoutput`` class
        """
        self._time_collected = datetime.datetime.now()

        generated = [
            "numerator_model",
            "denominator_model",
            "outcome_model",
            "hazard_ratio",
            "risk_estimates",
            "km_data",
            "km_graph",
            "diagnostics",
            "_survival_time",
            "_hazard_time",
            "_model_time",
            "_expansion_time",
            "weight_stats",
        ]
        for attr in generated:
            if not hasattr(self, attr):
                setattr(self, attr, None)

        # Options ==========================
        base = SEQopts()

        for name, value in vars(self).items():
            if name in asdict(base).keys():
                setattr(base, name, value)

        # Timing =========================
        time = {
            "start_time": self._time_initialized,
            "expansion_time": self._expansion_time,
            "model_time": self._model_time,
            "survival_time": self._survival_time,
            "hazard_time": self._hazard_time,
            "collection_time": self._time_collected,
        }

        if self.compevent_colname is not None:
            compevent_models = [model["compevent"] for model in self.outcome_model]
        else:
            compevent_models = None

        if self.outcome_model is not None:
            outcome_models = [model["outcome"] for model in self.outcome_model]

        if self.risk_estimates is None:
            risk_ratio = risk_difference = None
        else:
            risk_ratio = self.risk_estimates["risk_ratio"]
            risk_difference = self.risk_estimates["risk_difference"]

        output = SEQoutput(
            options=base,
            method=self.method,
            numerator_models=self.numerator_model,
            denominator_models=self.denominator_model,
            outcome_models=outcome_models,
            compevent_models=compevent_models,
            weight_statistics=self.weight_stats,
            hazard=self.hazard_ratio,
            km_data=self.km_data,
            km_graph=self.km_graph,
            risk_ratio=risk_ratio,
            risk_difference=risk_difference,
            time=time,
            diagnostic_tables=self.diagnostics,
        )

        return output
