import polars as pl

def _binder(DT, data, id_col, time_col, eligible_col, outcome_col, treatment_col, kept_cols, 
            baseline_indicator, squared_indicator):
    """
    Internal function to bind data to the map created by __mapper
    """
    excluded = {"dose",
                f"dose{squared_indicator}",
                "followup",
                f"followup{squared_indicator}",
                "tx_lag",
                "trial",
                f"trial{squared_indicator}",
                time_col, 
                f"{time_col}{squared_indicator}"}
    
    cols = kept_cols.union({eligible_col, outcome_col, treatment_col})
    cols = {col for col in cols if col is not None}

    regular = {col for col in cols if not (baseline_indicator in col or squared_indicator in col) and col not in excluded}
    
    baseline = {col for col in cols if baseline_indicator in col and col not in excluded}
    bas_kept = {col.replace(baseline_indicator, "") for col in baseline}
    
    squared = {col for col in cols if squared_indicator in col and col not in excluded}
    sq_kept = {col.replace(squared_indicator, "") for col in squared}
    
    kept = list(regular.union(bas_kept).union(sq_kept))
    
    DT = DT.join(
        data.select([id_col, time_col] + kept),
        left_on=[id_col, 'period'],
        right_on=[id_col, time_col],
        how='left'
    )  
    DT = DT.sort([id_col, "trial", "followup"]) \
        .with_columns([
            (pl.col("trial") ** 2).alias(f"trial{squared_indicator}"),
            (pl.col("followup") ** 2).alias(f"followup{squared_indicator}")
        ])
    
    if squared:
        squares = []
        for sq in squared:
            col = sq.replace(squared_indicator, '')
            squares.append((pl.col(col) ** 2).alias(f"{col}{squared_indicator}"))
        DT = DT.with_columns(squares)
        
    baseline_cols = {bas.replace(baseline_indicator, '') for bas in baseline}
    needed = {eligible_col, treatment_col}
    baseline_cols.update({c for c in needed})
    
    bas = [
        pl.col(c).first().over([id_col, 'trial']).alias(f"{c}{baseline_indicator}")
        for c in baseline_cols
    ]
    
    DT = DT.with_columns(bas).filter(pl.col(f"{eligible_col}{baseline_indicator}") == 1) \
           .drop([f"{eligible_col}{baseline_indicator}", eligible_col])         
    
    return DT