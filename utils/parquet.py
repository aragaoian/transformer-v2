import pyarrow.parquet as pq


def read_parquet(path: str) -> list[dict[str, str]]:
    pq_file = pq.ParquetFile(path)
    return pq_file.to_pylist()
