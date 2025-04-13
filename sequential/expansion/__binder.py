import polars as pl

def __binder(DT, data, id_col, time_col,
            eligible_col, excused_col0, excused_col1, cols, 
            baseline_indicator, squared_indicator):
    """
    Internal function to bind data to the map created by __mapper
    """
    excluded = {'dose',
                f'dose{squared_indicator}',
                time_col,
                f'{time_col}{squared_indicator}',
                'tx_lag'}
    cols = cols.union({eligible_col, excused_col0, excused_col1})
    cols = {col for col in cols if col is not None}
   
    regular = {col for col in cols if not (baseline_indicator in col or squared_indicator in col) and col not in excluded}
    baseline = {col for col in cols if baseline_indicator in col and col not in excluded}
    squared = {col for col in cols if squared_indicator in col and col not in excluded}
    
    DT = DT.join(
        data.select([time_col] + list(regular)),
        left_on=['period'],
        right_on=[time_col],
        how='left'
    )

    if squared:
        for sq in squared:
            col = sq.replace(squared_indicator, '')
            DT = DT.with_columns(
                (pl.col(col) ** 2).alias(f"{col}{squared_indicator}")
            )
    
    if baseline:
        base_cols = [bas.replace(baseline_indicator, '') for bas in baseline]
        
        baseline_df = DT.group_by([id_col, "trial"]).agg(
            [pl.col(col).first().alias(f"{col}{baseline_indicator}") for col in base_cols]
        )

        DT = DT.join(baseline_df, on=[id_col, "trial"], how='left')
    
    return DT
