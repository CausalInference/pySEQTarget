import polars as pl
from pySEQ import SEQuential, SEQopts
import os

csv_path = os.path.expanduser("/Users/ryanodea/Documents/GitHub/pySEQ/tests/SEQdata.csv")
data = pl.read_csv(csv_path)
params = SEQopts(bootstrap_nboot=0, weighted = True, weight_preexpansion = False)
s = SEQuential(data, "ID", "time", "eligible", "tx_init", "outcome", 
               ["N", "L", "P"], ["sex"], "censoring", params)
s.expand()

dt = s.DT
print(dt)
s.fit()
print(s.outcome_model[0].summary())
s.survival()
s.plot()

