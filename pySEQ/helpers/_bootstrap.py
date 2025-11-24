from functools import wraps
from concurrent.futures import ProcessPoolExecutor, as_completed
import polars as pl
from tqdm import tqdm
import copy
import time
from ._format_time import _format_time

def _prepare_boot_data(self, data, boot_id):
    id_counts = self._boot_samples[boot_id]
    
    counts = pl.DataFrame({
        self.id_col: list(id_counts.keys()),
        "count": list(id_counts.values())
    })
    
    bootstrapped = data.join(counts, on=self.id_col, how="inner")
    bootstrapped = bootstrapped.with_columns(
        pl.int_ranges(0, pl.col("count")).alias("replicate")
    ).explode("replicate").with_columns(
        (pl.col(self.id_col).cast(pl.Utf8) + "_" + pl.col("replicate").cast(pl.Utf8)).alias(self.id_col)
    ).drop("count", "replicate")
    
    return bootstrapped

def bootstrap_loop(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "outcome_model"):
            self.outcome_model = []
        start = time.perf_counter()
            
        results = []
        full = method(self, *args, **kwargs)
        results.append(full)
        
        if getattr(self, "bootstrap_nboot") > 0 and getattr(self, "_boot_samples", None):
            original_DT = self.DT
            nboot = self.bootstrap_nboot
            ncores = self.ncores
            
            def _worker(i):
                obj = copy.deepcopy(self)
                obj.DT = _prepare_boot_data(obj, original_DT, i)
                return method(obj, *args, **kwargs)

            if getattr(self, "parallel", False):
                with ProcessPoolExecutor(max_workers=ncores) as executor:
                    futures = [executor.submit(_worker, i) for i in range(nboot)]
                    for j in tqdm(as_completed(futures), total=nboot, desc="Bootstrapping..."):
                        results.append(j.result())
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
