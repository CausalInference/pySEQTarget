from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
import patsy


@dataclass
class _Model:
    name: str
    outcome: str
    rhs: str
    rows: Optional[Callable] = None
    # outcome values of a multinomial model with the reference first
    classes: Optional[list] = None

    def index(self, y):
        # class index of each outcome value
        return y if self.classes is None else np.searchsorted(self.classes, y)

    @property
    def m(self):
        return 1 if self.classes is None else len(self.classes) - 1


@dataclass
class _Rows:
    ids: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    member: np.ndarray
    w: np.ndarray
    keep: np.ndarray


class _Block:
    def __init__(self, models, gold, categories, id_col, weight_col=None):
        self.models = models
        self.categories = categories
        self.id_col = id_col
        self.weight_col = weight_col
        gold = self._cast(gold.reset_index(drop=True))
        self.y_infos, self.X_infos, columns = [], [], []
        for m in models:
            y, X = patsy.dmatrices(
                f"{m.outcome} ~ {m.rhs}",
                gold[self._filter(m, gold)],
                return_type="dataframe",
            )
            self.y_infos.append(y.design_info)
            self.X_infos.append(X.design_info)
            columns += [c for c in X.columns if c not in columns]
        self.columns = columns
        self.model_cols = [
            np.array([columns.index(c) for c in info.column_names])
            for info in self.X_infos
        ]

    def _cast(self, frame):
        for col, levels in self.categories.items():
            if col not in frame.columns:
                continue
            fixed = pd.Categorical(frame[col], categories=levels)
            if (pd.isna(fixed) & frame[col].notna().to_numpy()).any():
                raise ValueError(f"'{col}' has levels the census did not see")
            frame[col] = fixed
        return frame

    @staticmethod
    def _filter(m, frame):
        return np.ones(len(frame), bool) if m.rows is None else m.rows(frame)

    def design(self, frame, j):
        (X,) = patsy.build_design_matrices(
            [self.X_infos[j]], frame, return_type="dataframe"
        )
        return X.index.to_numpy(), X.to_numpy(float)

    def prepare(self, frame):
        return self._cast(frame.reset_index(drop=True))

    def matrices(self, frame):
        frame = self.prepare(frame)
        n, M = len(frame), len(self.models)
        X = np.full((n, len(self.columns)), np.nan)
        Y = np.full((n, M), np.nan)
        member = np.zeros((n, M), bool)
        for j, m in enumerate(self.models):
            pos, Xm = self.design(frame, j)
            cols = self.model_cols[j]
            before = X[np.ix_(pos, cols)]
            clash = ~np.isnan(before) & (before != Xm)
            if clash.any():
                raise ValueError(
                    f"Column '{self.columns[cols[clash.any(0)][0]]}' differs "
                    f"between the models of a block"
                )
            X[np.ix_(pos, cols)] = Xm
            (y,) = patsy.build_design_matrices(
                [self.y_infos[j]], frame, return_type="dataframe"
            )
            Y[y.index.to_numpy(), j] = y.to_numpy(float).ravel()
            defined = np.zeros(n, bool)
            defined[pos] = True
            member[:, j] = defined & ~np.isnan(Y[:, j]) & self._filter(m, frame)
        Y[~member] = np.nan
        keep = np.flatnonzero(member.any(1))
        w = (
            frame[self.weight_col].to_numpy(float)[keep]
            if self.weight_col
            else np.ones(len(keep))
        )
        ids = frame[self.id_col].to_numpy()[keep]
        return _Rows(ids, X[keep], Y[keep], member[keep], w, keep)
