import polars as pl
from ._outcome_fit import _outcome_fit

def _subgroup_fit(self):    
    subgroups = sorted(self.DT[self.subgroup_colname].unique().to_list())
    self._unique_subgroups = subgroups
    
    models = []
    for val in subgroups:
        subDT = self.DT.filter(pl.col(self.subgroup_colname) == val)
        
        model = _outcome_fit(self,
                            subDT,
                            self.outcome_col,
                            self.covariates,
                            self.weighted,
                            "weight")
        models.append(model)
    
    return models
