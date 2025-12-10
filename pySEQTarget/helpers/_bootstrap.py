import copy
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import wraps

import numpy as np
import polars as pl
from tqdm import tqdm

from ._format_time import _format_time


def _prepare_boot_data(self, data, boot_id):
    id_counts = self._boot_samples[boot_id]

    counts = pl.DataFrame(
        {self.id_col: list(id_counts.keys()), "count": list(id_counts.values())}
    )

    bootstrapped = data.join(counts, on=self.id_col, how="inner")
    bootstrapped = (
        bootstrapped.with_columns(pl.int_ranges(0, pl.col("count")).alias("replicate"))
        .explode("replicate")
        .with_columns(
            (
                pl.col(self.id_col).cast(pl.Utf8)
                + "_"
                + pl.col("replicate").cast(pl.Utf8)
            ).alias(self.id_col)
        )
        .drop("count", "replicate")
    )

    return bootstrapped


def _bootstrap_worker(obj, method_name, original_DT, i, seed, args, kwargs):
    obj = copy.deepcopy(obj)
    obj._rng = (
        np.random.RandomState(seed + i) if seed is not None else np.random.RandomState()
    )
    obj.DT = _prepare_boot_data(obj, original_DT, i)

    # Disable bootstrapping to prevent recursion
    obj.bootstrap_nboot = 0

    method = getattr(obj, method_name)
    result = method(*args, **kwargs)
    obj._rng = None
    return result


def bootstrap_loop(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "outcome_model"):
            self.outcome_model = []
        start = time.perf_counter()

        results = []
        original_DT = self.DT

        full = method(self, *args, **kwargs)
        results.append(full)

        if getattr(self, "bootstrap_nboot") > 0 and getattr(
            self, "_boot_samples", None
        ):
            nboot = self.bootstrap_nboot
            ncores = self.ncores
            seed = getattr(self, "seed", None)
            method_name = method.__name__

            if getattr(self, "parallel", False):
                original_rng = getattr(self, "_rng", None)
                self._rng = None

                with ProcessPoolExecutor(max_workers=ncores) as executor:
                    futures = [
                        executor.submit(
                            _bootstrap_worker,
                            self,
                            method_name,
                            original_DT,
                            i,
                            seed,
                            args,
                            kwargs,
                        )
                        for i in range(nboot)
                    ]
                    for j in tqdm(
                        as_completed(futures), total=nboot, desc="Bootstrapping..."
                    ):
                        results.append(j.result())

                self._rng = original_rng
            else:
                for i in tqdm(range(nboot), desc="Bootstrapping..."):
                    self.DT = _prepare_boot_data(self, original_DT, i)
                    boot_fit = method(self, *args, **kwargs)
                    results.append(boot_fit)

            self.DT = original_DT

            end = time.perf_counter()
            self._model_time = _format_time(start, end)

        self.outcome_model = results
        return results

    return wrapper
