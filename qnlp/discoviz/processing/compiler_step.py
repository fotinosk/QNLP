import hashlib
from pathlib import Path

import lmdb
import polars as pl


def mock_ccg_compile(text: str) -> bytes:
    raise NotImplementedError


class CompilerStep:
    def __init__(self, lmdb_path: Path | str, text_column: str = "processed_text"):
        self.lmdb_path = Path(lmdb_path)
        self.lmdb_path.parent.mkdir(parents=True, exist_ok=True)
        self.text_column = text_column

        # Initialize LMDB environment. max_readers high for concurrent access.
        self.env = lmdb.open(
            str(self.lmdb_path),
            max_readers=128,
            map_size=10 * 1024 * 1024 * 1024,  # 10GB
            create=True,
        )

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.text_column not in df.columns:
            raise ValueError(f"Column '{self.text_column}' not found in DataFrame.")

        # 1. Generate text_hash
        def hash_string(s: str) -> str:
            return hashlib.sha256(s.encode("utf-8")).hexdigest()

        df = df.with_columns(
            pl.col(self.text_column).map_elements(hash_string, return_dtype=pl.String).alias("text_hash")
        )

        hashes = df["text_hash"].to_list()
        texts = df[self.text_column].to_list()

        # 2. Filter batch to isolate the "delta" (unseen hashes)
        unseen_pairs = []
        with self.env.begin(write=False) as txn:
            for text, h in zip(texts, hashes):
                if txn.get(h.encode("utf-8")) is None:
                    unseen_pairs.append((text, h))

        # 3. & 4. Pass delta to CCG parser and commit to LMDB
        if unseen_pairs:
            with self.env.begin(write=True) as txn:
                for text, h in unseen_pairs:
                    compiled_tree = mock_ccg_compile(text)
                    txn.put(h.encode("utf-8"), compiled_tree)

        # 5. Return the batch (now containing text_hash)
        return df

    def close(self):
        self.env.close()
