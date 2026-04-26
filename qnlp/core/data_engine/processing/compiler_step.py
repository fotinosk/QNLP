import gc
import hashlib
import json
import multiprocessing as mp
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import lmdb
import polars as pl

from qnlp.core.data_engine.processing.pipeline import PipelineStep
from qnlp.utils.logging import setup_logger

_worker_processor = None
logger = setup_logger("ccg_parser")


def _worker_init(bond_dim: int, embedding_dim: int, device: str, rules: list[str]):
    """Initialise heavy CCG objects ONCE per worker process."""
    global _worker_processor
    from lambeq import AtomicType, Rewriter
    from lambeq.backend.tensor import Dim

    from qnlp.discoviz.models.bobcat_text_processor import BobcatTextProcessor
    from qnlp.discoviz.parser.asnsatz import CustomMPSAnsatz
    from qnlp.discoviz.parser.cached_bobcat import CachedBobcatParser

    ansatz = CustomMPSAnsatz(
        {
            AtomicType.SENTENCE: Dim(embedding_dim),
            AtomicType.NOUN: Dim(embedding_dim),
            AtomicType.PREPOSITIONAL_PHRASE: Dim(embedding_dim),
        },
        bond_dim=bond_dim,
    )
    parser = CachedBobcatParser(device=device)
    _worker_processor = BobcatTextProcessor(
        ccg_parser=parser,
        ansatz=ansatz,
        rewriter=Rewriter(rules),
    )


def _worker_process_batch(texts: list[str]) -> list[dict]:
    """Process a small batch of texts inside a worker. Returns serialisable dicts."""
    global _worker_processor
    if _worker_processor is None:
        raise RuntimeError("Worker processor not initialised.")

    results = []
    for text in texts:
        try:
            out = _worker_processor([text])
            einsum_inputs = out["einsum_inputs"][0]
            diagram = einsum_inputs[0]
            symbols = [[asdict(x[0]), x[1]] for x in einsum_inputs[1]]
            results.append({"text": text, "diagram": diagram, "symbols": symbols, "error": None})
        except Exception as e:
            results.append({"text": text, "diagram": None, "symbols": None, "error": str(e)})

    # Aggressive cleanup inside worker to prevent RSS growth
    gc.collect()
    return results


class CCGCompilerStep(PipelineStep):
    """
    Compiles text into CCG tensor diagrams using a multiprocessing pool.
    Caches results in LMDB to skip already-compiled sentences.
    Designed for memory-constrained environments with heavy parsers.
    """

    def __init__(
        self,
        lmdb_path: Path | str,
        text_column: str = "processed_text",
        bond_dim: int = 10,
        embedding_dim: int = 50,
        device: str = "mps",
        max_workers: int = 2,
        worker_batch_size: int = 1000,
        max_tasks_per_child: int = 5,
    ):
        self.lmdb_path = Path(lmdb_path)
        self.text_column = text_column
        self.bond_dim = bond_dim
        self.embedding_dim = embedding_dim
        self.device = device
        self.max_workers = max_workers
        self.worker_batch_size = worker_batch_size
        self.max_tasks_per_child = max_tasks_per_child
        self.rules = [
            "auxiliary",
            "connector",
            "determiner",
            "postadverb",
            "preadverb",
            "prepositional_phrase",
            "coordination",
            "object_rel_pronoun",
            "subject_rel_pronoun",
        ]
        self._pool: Optional[mp.Pool] = None

    def _get_pool(self) -> mp.Pool:
        """Lazy-initialise multiprocessing pool with worker recycling."""
        if self._pool is None:
            self._pool = mp.Pool(
                processes=self.max_workers,
                initializer=_worker_init,
                initargs=(self.bond_dim, self.embedding_dim, self.device, self.rules),
                maxtasksperchild=self.max_tasks_per_child,  # Forces worker restart to free RAM
            )
        return self._pool

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        logger.info(f"Starting CCGCompilerStep for chunk of size {len(df)}")
        if self.text_column not in df.columns:
            raise ValueError(f"Column '{self.text_column}' not found in DataFrame.")

        # 1. Hash texts for deduplication & LMDB keys
        logger.debug("Hashing text column for deduplication...")
        df = df.with_columns(
            pl.col(self.text_column)
            .map_elements(
                lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest(),
                return_dtype=pl.String,
                skip_nulls=True,
            )
            .alias("text_hash")
        )

        hashes = df["text_hash"].to_list()
        texts = df[self.text_column].to_list()

        # 2. Check LMDB cache to skip already-compiled sentences
        unseen_mask = [True] * len(hashes)
        if self.lmdb_path.exists():
            logger.debug(f"Checking LMDB cache at {self.lmdb_path} for existing hashes...")
            env = lmdb.open(str(self.lmdb_path), max_readers=128, readonly=True, lock=False)
            with env.begin() as txn:
                for i, h in enumerate(hashes):
                    if txn.get(h.encode("utf-8")) is not None:
                        unseen_mask[i] = False
            env.close()

        unseen_indices = [i for i, m in enumerate(unseen_mask) if m]
        unseen_texts = [texts[i] for i in unseen_indices]

        cached_count = len(texts) - len(unseen_texts)
        logger.info(f"Cache check complete: {cached_count} already compiled, {len(unseen_texts)} need compilation.")

        # 3. Compile only unseen texts using recycled worker pool
        compiled_map: dict[str, bytes] = {}
        if unseen_texts:
            logger.info(
                f"Distributing {len(unseen_texts)} texts across {self.max_workers} workers "
                "(batch size: {self.worker_batch_size})..."
            )
            batches = [
                unseen_texts[i : i + self.worker_batch_size]
                for i in range(0, len(unseen_texts), self.worker_batch_size)
            ]

            pool = self._get_pool()
            # imap_unordered yields results as workers finish, keeping memory low
            processed_batches = 0
            for batch_results in pool.imap_unordered(_worker_process_batch, batches, chunksize=1):
                processed_batches += 1
                if processed_batches % max(1, len(batches) // 5) == 0:
                    logger.debug(f"Compiled {processed_batches}/{len(batches)} batches...")

                for res in batch_results:
                    h = hashlib.sha256(res["text"].encode("utf-8")).hexdigest()
                    payload = {
                        "diagram": res["diagram"],
                        "symbols": res["symbols"],
                        "error": res["error"],
                    }
                    if res["error"]:
                        logger.warning(f"Compilation error for text '{res['text'][:50]}...': {res['error']}")
                    compiled_map[h] = json.dumps(payload).encode("utf-8")
            logger.info("Compilation complete.")

        # 4. Attach compiled bytes (None for cached/failed rows)
        compiled_bytes = [compiled_map.get(h) for h in hashes]
        df = df.with_columns(pl.Series("compiled_bytes", compiled_bytes, dtype=pl.Binary))

        # 5. Main process cleanup
        gc.collect()
        return df

    def teardown(self) -> None:
        """Gracefully shut down the worker pool. Call after pipeline.run()."""
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def __del__(self):
        self.teardown()
