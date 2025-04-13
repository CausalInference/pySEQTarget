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
        data.select([id_col, time_col] + list(regular)),
        left_on=[id_col, 'period'],
        right_on=[id_col, time_col],
        how='left'
    )

    if squared:
        for sq in squared:
            col = sq.replace(squared_indicator, '')
            DT = DT.with_columns(
                (pl.col(col) ** 2).alias(f"{col}{squared_indicator}")
            )
    
    if baseline:
        base = [bas.replace(baseline_indicator, '') for bas in baseline] + [eligible_col]

        for col in base:
            DT = DT.with_columns(
                pl.col(col).over([id_col, 'trial']).first().alias(f"{col}{baseline_indicator}")
            )
    
    DT = DT.filter(pl.col(f"{eligible_col}{baseline_indicator}") == 1) \
    .drop([f"{eligible_col}{baseline_indicator}", eligible_col])         
    
    return DT