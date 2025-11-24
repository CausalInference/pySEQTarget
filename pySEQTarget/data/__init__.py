from importlib.resources import files

import polars as pl


def load_data(name: str = "SEQdata") -> pl.DataFrame:
    loc = files("pySEQTarget.data")
    if name in ["SEQdata", "SEQdata_multitreatment", "SEQdata_LTFU"]:
        if name == "SEQdata":
            data_path = loc.joinpath("SEQdata.csv")
        elif name == "SEQdata_multitreatment":
            data_path = loc.joinpath("SEQdata_multitreatment.csv")
        else:
            data_path = loc.joinpath("SEQdata_LTFU.csv")
        return pl.read_csv(data_path)
    else:
        raise ValueError(
            f"Dataset '{name}' not available. Options: ['SEQdata', 'SEQdata_multitreatment', 'SEQdata_LTFU']"
        )
