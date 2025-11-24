from typing import Optional, List, Literal
import time
from dataclasses import asdict
from collections import Counter
import polars as pl
import numpy as np
import datetime

from .SEQopts import SEQopts
from .SEQoutput import SEQoutput
from .error import _param_checker
from .helpers import _col_string, bootstrap_loop, _format_time
from .initialization import _outcome, _numerator, _denominator, _cense_numerator, _cense_denominator
from .expansion import _binder, _dynamic, _random_selection, _diagnostics
from .weighting import _weight_setup, _fit_LTFU, _fit_numerator, _fit_denominator, _weight_bind, _weight_predict, _weight_stats
from .analysis import _outcome_fit, _pred_risk, _calculate_survival, _subgroup_fit, _calculate_hazard, _risk_estimates
from .plot import _survival_plot


class SEQuential:
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
            parameters: Optional[SEQopts] = None
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
            
        self._rng = np.random.RandomState(self.seed) if self.seed is not None else np.random
        
        if self.covariates is None:
            self.covariates = _outcome(self)

        if self.weighted:
            if self.numerator is None:
                self.numerator = _numerator(self)

            if self.denominator is None:
                self.denominator = _denominator(self)

            if self.cense_colname is not None:
                if self.cense_numerator is None:
                    self.cense_numerator = _cense_numerator(self)

                if self.cense_denominator is None:
                    self.cense_denominator = _cense_denominator(self)
        
        _param_checker(self)

    def expand(self):
        start = time.perf_counter()
        kept = [self.cense_colname, self.cense_eligible_colname,
                self.compevent_colname,
                *self.weight_eligible_colnames,
                *self.excused_colnames]
        
        self.data = self.data.with_columns([
            pl.when(pl.col(self.treatment_col).is_in(self.treatment_level))
            .then(self.eligible_col)
            .otherwise(0)
            .alias(self.eligible_col),
            pl.col(self.treatment_col)
            .shift(1)
            .over([self.id_col])
            .alias("tx_lag"),
            pl.lit(False).alias("switch")
        ]).with_columns([
            pl.when(pl.col(self.time_col) == 0)
            .then(pl.lit(False))
            .otherwise(
                (pl.col("tx_lag").is_not_null()) &
                (pl.col("tx_lag") != pl.col(self.treatment_col))
            ).cast(pl.Int8)
            .alias("switch")
        ])
        
        self.DT = _binder(self, kept_cols= _col_string([self.covariates, 
                                                        self.numerator, 
                                                        self.denominator, 
                                                        self.cense_numerator, 
                                                        self.cense_denominator]).union(kept)) \
                            .with_columns(
                                pl.col(self.id_col)
                                .cast(pl.Utf8)
                                .alias(self.id_col)
                                )
                            
        self.data = self.data.with_columns(
            pl.col(self.id_col)
            .cast(pl.Utf8)
            .alias(self.id_col)
            )
        
        if self.method != "ITT":
            _dynamic(self)
        if self.selection_random:
            _random_selection(self)
        _diagnostics(self)
            
        end = time.perf_counter()
        self._expansion_time = _format_time(start, end)
                    
    def bootstrap(self, **kwargs):
        allowed = {"bootstrap_nboot", "bootstrap_sample", 
                   "bootstrap_CI", "bootstrap_method"}
        for key, value in kwargs.items():
            if key in allowed:
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown argument: {key}")
        
        UIDs = self.DT.select(pl.col(self.id_col)).unique().to_series().to_list()
        NIDs = len(UIDs)
        
        self._boot_samples = []
        for _ in range(self.bootstrap_nboot):
            sampled_IDs = self._rng.choice(UIDs, size=int(self.bootstrap_sample * NIDs), replace=True)
            id_counts = Counter(sampled_IDs)
            self._boot_samples.append(id_counts)
        return self
    
    @bootstrap_loop      
    def fit(self):
        if self.bootstrap_nboot > 0 and not hasattr(self, "_boot_samples"):
            raise ValueError("Bootstrap sampling not found. Please run the 'bootstrap' method before fitting with bootstrapping.")
        
        if self.weighted:
            WDT = _weight_setup(self)
            if not self.weight_preexpansion and not self.excused:
                WDT = WDT.filter(pl.col("followup") > 0)
                
            WDT = WDT.to_pandas()
            for col in self.fixed_cols:
                if col in WDT.columns:
                    WDT[col] = WDT[col].astype("category")
            
            _fit_LTFU(self, WDT)
            _fit_numerator(self, WDT)
            _fit_denominator(self, WDT)
            
            WDT = pl.from_pandas(WDT)
            WDT = _weight_predict(self, WDT)
            _weight_bind(self, WDT)
            self.weight_stats = _weight_stats(self)
        
        if self.subgroup_colname is not None:
            return _subgroup_fit(self)
        
        models = {'outcome': _outcome_fit(self, self.DT, 
                                          self.outcome_col, 
                                          self.covariates, 
                                          self.weighted, 
                                          "weight")}
        if self.compevent_colname is not None:
            models['compevent'] = _outcome_fit(self, self.DT, 
                                               self.compevent_colname, 
                                               self.covariates, 
                                               self.weighted, 
                                               "weight")
        return models
        
    def survival(self):
        if not hasattr(self, "outcome_model") or not self.outcome_model:
            raise ValueError("Outcome model not found. Please run the 'fit' method before calculating survival.")
        
        start = time.perf_counter()
        
        risk_data = _pred_risk(self)
        surv_data = _calculate_survival(self, risk_data)
        self.km_data = pl.concat([risk_data, surv_data])
        self.risk_estimates = _risk_estimates(self)
        
        end = time.perf_counter()
        self._survival_time = _format_time(start, end)
  
    def hazard(self):
        start = time.perf_counter()
        
        if not hasattr(self, "outcome_model") or not self.outcome_model:
            raise ValueError("Outcome model not found. Please run the 'fit' method before calculating hazard ratio.")
        self.hazard_ratio = _calculate_hazard(self)
        
        end = time.perf_counter()
        self._hazard_time = _format_time(start, end)

    def plot(self):
        self.km_graph = _survival_plot(self)
        
    def collect(self):
        self._time_collected = datetime.datetime.now()
        
        generated = [
            "numerator_model", "denominator_model",
            "outcome_model",
            "hazard_ratio", "risk_estimates",
            "km_data", "diagnostics", 
            "_survival_time", "_hazard_time", 
            "_model_time", "_expansion_time",
            "weight_stats"
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
        time = {"start_time": self._time_initialized,
                "expansion_time": self._expansion_time,
                "model_time": self._model_time,
                "survival_time": self._survival_time,
                "hazard_time": self._hazard_time,
                "collection_time": self._time_collected}
        
        if self.compevent_colname is not None:
            compevent_models = [model["compevent"] for model in self.outcome_models]
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
            diagnostic_tables=self.diagnostics
        )
        
        return output
                
