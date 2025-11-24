from dataclasses import dataclass
from typing import List, Optional, Literal
from .SEQopts import SEQopts
import statsmodels.formula.api as smf
import polars as pl
import matplotlib.figure

@dataclass
class SEQoutput:
    options: SEQopts = None
    method: str = None
    numerator_models: List[smf.MNLogit] = None
    denominator_models: List[smf.MNLogit] = None
    outcome_models: List[List[smf.glm]] = None
    compevent_models: List[List[smf.glm]] = None
    weight_statistics: dict = None
    hazard: pl.DataFrame = None
    km_data: pl.DataFrame = None
    km_graph: matplotlib.figure.Figure = None
    risk_ratio: pl.DataFrame = None
    risk_difference: pl.DataFrame = None
    time: dict = None
    diagnostic_tables: dict = None
    
    def plot(self):
        print(self.km_graph)
        
    def summary(self,
                type = Optional[Literal[
                    "numerator",
                    "denominator",
                    "outcome",
                    "compevent"]]):
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
    
    def retrieve_data(self, 
                      type = Optional[Literal[
                          "km_data",
                          "hazard",
                          "risk_ratio",
                          "risk_difference",
                          "unique_outcomes",
                          "nonunique_outcomes",
                          "unique_switches",
                          "nonunique_switches"
                      ]]):
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
            ValueError("Data {type} was not created in the SEQuential process")
        return data
                
        