import pickle
import re

import pandas as pd

from ..helpers._fix_categories import _model_design_info
from ._survival_pred import _same_state, _transform_state

# Columns with at most this many distinct values contribute one row per value
# to the reference frame, so the levels patsy infers from data are all present
_MAX_LEVELS = 1000


def _reference_frame(frame, formula):
    """
    A few rows of the formula's columns of ``frame``, holding every value of
    each low-cardinality column.
    """
    names = set(re.findall(r"\w+", formula))
    frame = frame[[c for c in frame.columns if c in names]]
    keep = pd.Index(frame.index[:2])
    for col in frame.columns:
        values = frame[col]
        if values.nunique(dropna=False) <= _MAX_LEVELS:
            keep = keep.union(values.drop_duplicates().index)
    return frame.loc[keep]


def _same_design(a, b):
    return list(a.column_names) == list(b.column_names) and _same_state(
        _transform_state(a), _transform_state(b)
    )


def _slim_model(model):
    """
    Drop a fitted outcome model's per-row data in place, keeping what
    prediction, ``bse``, ``summary()`` and pickling need.

    One outcome model is kept per bootstrap replicate until the end of the run,
    and a statsmodels GLM holds its full data frame, design matrix, weights and
    fitted values — about 260 bytes per expanded row — so memory grew with the
    expanded data times ``bootstrap_nboot``. The glum and jax wrappers keep the
    design matrix and weights in the same way.
    """
    if hasattr(model, "_X_design"):
        # glum/jax wrappers: cache the covariance before dropping its inputs
        model.remove_data()
        return model

    # Fill summary()'s caches first; several of its statistics (e.g. the
    # null-model log-likelihood) are computed from the data being dropped.
    model.summary()
    model.remove_data()

    # remove_data() keeps the formula frame, which statsmodels re-evaluates the
    # formula on when unpickling (process pool, offload, users saving results).
    # Swap it for a few reference rows, but only if the design rebuilt from them
    # matches the fitted one; stateful transforms such as a centred spline
    # depend on all the data, so those models keep the full frame.
    data = model.model.data
    data.orig_endog = None
    data.orig_exog = None
    frame = data.frame
    design_info = _model_design_info(model)
    data.frame = _reference_frame(frame, str(data.formula))
    try:
        restored = _model_design_info(pickle.loads(pickle.dumps(model)))
        ok = restored is not None and _same_design(restored, design_info)
    except Exception:
        ok = False
    if not ok:
        data.frame = frame
    return model
