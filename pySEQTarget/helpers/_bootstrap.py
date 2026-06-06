import copy
import time
import warnings
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

    # Build a per-row unique ID for the resampled frame. With an integer-typed
    # ID column do this in integer arithmetic (orig_id * id_mult + replicate)
    # rather than the old string-concat form ("{orig_id}_{replicate}"), since
    # string IDs make every downstream join/groupby ~3-5x slower. id_mult is
    # the max count seen this iteration plus one, so the (orig_id, replicate)
    # pair maps to a unique int64 with room. Falls back to string concat for
    # non-integer ID columns (e.g. user-supplied string IDs).
    id_is_int = data.schema[self.id_col].is_integer()
    if id_is_int:
        id_mult = (max(id_counts.values()) if id_counts else 1) + 1
        # _weight_bind recovers orig_id with `id // id_mult` to join the
        # bootstrap-resampled self.DT back to the un-resampled WDT, so the
        # multiplier has to be discoverable downstream.
        self._boot_id_mult = id_mult
        new_id = (
            pl.col(self.id_col).cast(pl.Int64) * id_mult + pl.col("replicate")
        ).alias(self.id_col)
    else:
        new_id = (
            pl.col(self.id_col).cast(pl.Utf8)
            + "_"
            + pl.col("replicate").cast(pl.Utf8)
        ).alias(self.id_col)

    bootstrapped = (
        data.lazy()
        .join(counts.lazy(), on=self.id_col, how="inner")
        .with_columns(pl.int_ranges(0, pl.col("count")).alias("replicate"))
        .explode("replicate")
        .with_columns(new_id)
        .drop("count", "replicate")
        .collect()
    )

    return bootstrapped


def _bootstrap_worker(obj, method_name, original_DT, i, seed, args, kwargs):
    # Shallow copy the object and only deep copy mutable state that changes per-bootstrap
    obj = copy.copy(obj)
    # Deep copy only the mutable attributes that get modified during fitting
    obj.outcome_model = []
    obj.numerator_model = (
        copy.copy(obj.numerator_model)
        if hasattr(obj, "numerator_model") and obj.numerator_model
        else []
    )
    obj.denominator_model = (
        copy.copy(obj.denominator_model)
        if hasattr(obj, "denominator_model") and obj.denominator_model
        else []
    )

    obj._rng = (
        np.random.RandomState(seed + i) if seed is not None else np.random.RandomState()
    )
    original_DT = obj._offloader.load_dataframe(original_DT)
    obj.DT = _prepare_boot_data(obj, original_DT, i)
    del original_DT
    obj._current_boot_idx = i + 1

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

        seed = getattr(self, "seed", None)
        if seed is not None:
            self._rng = np.random.RandomState(seed)

        self._current_boot_idx = None
        full = method(self, *args, **kwargs)
        results.append(full)

        if getattr(self, "bootstrap_nboot") > 0 and getattr(
            self, "_boot_samples", None
        ):
            nboot = self.bootstrap_nboot
            ncores = self.ncores
            seed = getattr(self, "seed", None)
            method_name = method.__name__

            original_DT_ref = self._offloader.save_dataframe(original_DT, "_DT")

            if getattr(self, "parallel", False):
                original_rng = getattr(self, "_rng", None)
                self._rng = None
                self.DT = None

                with ProcessPoolExecutor(max_workers=ncores) as executor:
                    futures = {
                        executor.submit(
                            _bootstrap_worker,
                            self,
                            method_name,
                            original_DT_ref,
                            i,
                            seed,
                            args,
                            kwargs,
                        ): i
                        for i in range(nboot)
                    }
                    skipped = 0
                    boot_sample_idx = []
                    for j in tqdm(
                        as_completed(futures), total=nboot, desc="Bootstrapping..."
                    ):
                        boot_idx = futures[j]
                        try:
                            results.append(j.result())
                            boot_sample_idx.append(boot_idx)
                        except np.linalg.LinAlgError as e:
                            skipped += 1
                            warnings.warn(
                                f"Bootstrap iteration {boot_idx + 1} failed "
                                f"({e}); skipping replicate.",
                                UserWarning,
                                stacklevel=2,
                            )

                self._rng = original_rng
                self.DT = self._offloader.load_dataframe(original_DT_ref)
            else:
                # Keep original data in memory if offloading is disabled to avoid unnecessary I/O
                if self._offloader.enabled:
                    original_DT_ref = self._offloader.save_dataframe(original_DT, "_DT")
                    del original_DT
                else:
                    original_DT_ref = original_DT

                skipped = 0
                boot_sample_idx = []
                for i in tqdm(range(nboot), desc="Bootstrapping..."):
                    self._current_boot_idx = i + 1
                    if seed is not None:
                        self._rng = np.random.RandomState(seed + i)
                    tmp = self._offloader.load_dataframe(original_DT_ref)
                    self.DT = _prepare_boot_data(self, tmp, i)
                    if self._offloader.enabled:
                        del tmp
                    self.bootstrap_nboot = 0
                    try:
                        boot_fit = method(self, *args, **kwargs)
                        results.append(boot_fit)
                        boot_sample_idx.append(i)
                    except np.linalg.LinAlgError as e:
                        skipped += 1
                        warnings.warn(
                            f"Bootstrap iteration {i + 1} failed "
                            f"({e}); skipping replicate.",
                            UserWarning,
                            stacklevel=2,
                        )

                self.DT = self._offloader.load_dataframe(original_DT_ref)

            # Maps each fitted bootstrap model (results[1:]) back to its
            # original _boot_samples index. Skipped replicates leave gaps, so
            # downstream consumers (e.g. hazard) must use this to pair a
            # resample with its model rather than assuming a 1:1 ordering.
            self._boot_sample_idx = boot_sample_idx
            self.bootstrap_nboot = len(results) - 1
            if skipped > 0:
                warnings.warn(
                    f"{skipped} of {nboot} bootstrap replicate(s) skipped due to "
                    "singular Hessian; effective bootstrap_nboot is "
                    f"{self.bootstrap_nboot}.",
                    UserWarning,
                    stacklevel=2,
                )

            end = time.perf_counter()
            self._model_time = _format_time(start, end)

        self.outcome_model = results
        return results

    return wrapper
