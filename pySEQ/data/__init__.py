from importlib.resources import files
import polars as pl

def load_data(name: str = "SEQdata") -> pl.DataFrame:
    
    if name == "SEQdata":
        data_path = files("pySEQ.data").joinpath("SEQdata.csv")
        return pl.read_csv(data_path)
    
    else:
        raise ValueError(f"Dataset '{name}' not available. Options: ['SEQdata']")