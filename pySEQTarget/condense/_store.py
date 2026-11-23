import json
from pathlib import Path

import polars as pl


class RecordStore:
    """
    Person-time record on disk as parquet shards of whole people
    """

    def __init__(self, path):
        self.path = Path(path)
        meta = json.loads((self.path / "meta.json").read_text())
        self.id_col = meta["id_col"]
        self.n_shards = meta["n_shards"]
        self._index = pl.read_parquet(self.path / "index.parquet")

    @classmethod
    def write(cls, data, id_col, path, people_per_shard=10_000):
        """
        Writes a dataframe to a new store

        :param data: Record to write
        :param id_col: Column name of the ID
        :param path: Directory to write to
        :param people_per_shard: People per parquet file
        :rtype: RecordStore
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        index = (
            data.select(pl.col(id_col).unique().sort())
            .with_row_index("_pos")
            .with_columns((pl.col("_pos") // people_per_shard).alias("shard"))
            .drop("_pos")
        )
        shards = (
            data.join(index, on=id_col, how="left", maintain_order="left")
            .sort("shard", id_col, maintain_order=True)
            .partition_by("shard", as_dict=True, maintain_order=True)
        )
        for (k,), part in shards.items():
            part.drop("shard").write_parquet(path / f"shard_{k:05d}.parquet")

        index.write_parquet(path / "index.parquet")
        meta = {"id_col": id_col, "n_shards": len(shards)}
        (path / "meta.json").write_text(json.dumps(meta))
        return cls(path)

    def ids(self):
        """
        IDs of every person in the store
        """
        return self._index[self.id_col].to_list()

    def _shard(self, k):
        return pl.read_parquet(self.path / f"shard_{k:05d}.parquet")

    def batches(self, people_per_batch=None):
        """
        Yields the record a batch of whole people at a time

        :param people_per_batch: People per batch, default one shard
        """
        for k in range(self.n_shards):
            shard = self._shard(k)
            if people_per_batch is None:
                yield shard
                continue
            ids = shard[self.id_col].unique(maintain_order=True)
            for start in range(0, len(ids), people_per_batch):
                batch = ids.slice(start, people_per_batch)
                yield shard.filter(pl.col(self.id_col).is_in(batch.implode()))

    def read(self, ids):
        """
        Record of the given people

        :param ids: IDs to read
        """
        wanted = self._index.filter(pl.col(self.id_col).is_in(list(ids)))
        parts = [
            self._shard(k).filter(
                pl.col(self.id_col).is_in(
                    wanted.filter(pl.col("shard") == k)[self.id_col].implode()
                )
            )
            for k in wanted["shard"].unique().sort().to_list()
        ]
        return pl.concat(parts)
