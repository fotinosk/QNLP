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

logger = setup_logger(log_name="ccg_parser")

# Worker globals (post-parse only; CPU-bound; one set per worker process).
_w_ansatz = None
_w_rewriter = None
_w_tn_to_einsum = None


def _post_parse_init(bond_dim: int, embedding_dim: int, rules: list[str]):
    """Initialise CPU-only post-parse machinery in a worker process.

    Workers never touch the parser or the GPU — they only handle the
    per-tree work: tree.to_diagram → rewriter → ansatz → tn_to_einsum.
    """
    global _w_ansatz, _w_rewriter, _w_tn_to_einsum
    from lambeq import AtomicType, Rewriter
    from lambeq.backend.tensor import Dim

    from qnlp.discoviz.models.bobcat_text_processor import tn_to_einsum
    from qnlp.discoviz.parser.asnsatz import CustomMPSAnsatz

    _w_ansatz = CustomMPSAnsatz(
        {
            AtomicType.SENTENCE: Dim(embedding_dim),
            AtomicType.NOUN: Dim(embedding_dim),
            AtomicType.PREPOSITIONAL_PHRASE: Dim(embedding_dim),
        },
        bond_dim=bond_dim,
    )
    _w_rewriter = Rewriter(rules)
    _w_tn_to_einsum = tn_to_einsum


def _post_parse_one(args, timeout_s: int = 30) -> dict:
    """Compile one (text, CCGTree) pair to einsum + symbols.

    SIGALRM bounds the work so a pathological sentence cannot hang the worker.
    Safe here because workers never touch CUDA (signals reliably interrupt
    pure-Python lambeq operations).
    """
    text, tree = args
    if tree is None:
        return {"text": text, "diagram": None, "symbols": None, "error": "parse failed"}

    import signal

    def _on_alarm(signum, frame):
        raise TimeoutError(f"post-parse timed out after {timeout_s}s")

    old_handler = signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(timeout_s)
    try:
        diagram = tree.to_diagram()
        if _w_rewriter is not None:
            diagram = _w_rewriter(diagram).remove_snakes()
        circuit = _w_ansatz(diagram)
        einsum_str, tensors = _w_tn_to_einsum(circuit)
        symbols = [[asdict(x[0]), x[1]] for x in tensors]
        return {"text": text, "diagram": einsum_str, "symbols": symbols, "error": None}
    except TimeoutError as e:
        return {"text": text, "diagram": None, "symbols": None, "error": str(e)}
    except Exception as e:
        return {"text": text, "diagram": None, "symbols": None, "error": f"{type(e).__name__}: {e}"}
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


class CCGCompilerStep(PipelineStep):
    """
    Hybrid CCG compiler.

    Architecture:
      • Parsing (BobcatParser → CCGTree, BERT-bound) runs in the **main
        process** on the configured ``device`` (e.g. CUDA), batched at
        ``parser_batch_size``.
      • Post-parse work (tree → diagram → rewriter → ansatz → tn_to_einsum,
        CPU-bound, per-sentence) runs across ``max_workers`` worker processes.
      • Trees stream from the parser into ``pool.imap_unordered`` so the GPU
        and CPU workers pipeline naturally.

    Caches results in LMDB keyed by SHA-256 of the text. Already-compiled
    texts are skipped via an anti-join against LMDB at the start of every
    chunk, so re-runs are cheap.
    """

    def __init__(
        self,
        lmdb_path: Path | str,
        text_column: str = "processed_text",
        bond_dim: int = 10,
        embedding_dim: int = 50,
        device: str = "cpu",
        parser_batch_size: int = 64,
        max_workers: int = 4,
        max_tasks_per_child: int = 100,
    ):
        self.lmdb_path = Path(lmdb_path)
        self.text_column = text_column
        self.bond_dim = bond_dim
        self.embedding_dim = embedding_dim
        self.device = device
        self.parser_batch_size = parser_batch_size
        self.max_workers = max_workers
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
        self._processor = None  # main-process BobcatTextProcessor

    def _get_processor(self):
        """Lazy-init the BobcatTextProcessor in the main process.

        The parser is loaded once and stays warm for the whole pipeline run.
        """
        if self._processor is None:
            from lambeq import AtomicType, Rewriter
            from lambeq.backend.tensor import Dim

            from qnlp.discoviz.models.bobcat_text_processor import BobcatTextProcessor
            from qnlp.discoviz.parser.asnsatz import CustomMPSAnsatz
            from qnlp.discoviz.parser.cached_bobcat import CachedBobcatParser

            # The processor's ansatz/rewriter aren't used here — we only call
            # sentences2trees() — but BobcatTextProcessor requires both.
            ansatz = CustomMPSAnsatz(
                {
                    AtomicType.SENTENCE: Dim(self.embedding_dim),
                    AtomicType.NOUN: Dim(self.embedding_dim),
                    AtomicType.PREPOSITIONAL_PHRASE: Dim(self.embedding_dim),
                },
                bond_dim=self.bond_dim,
            )
            parser = CachedBobcatParser(device=self.device, batch_size=self.parser_batch_size)
            self._processor = BobcatTextProcessor(
                ccg_parser=parser,
                ansatz=ansatz,
                rewriter=Rewriter(self.rules),
            )
        return self._processor

    def _get_pool(self) -> mp.Pool:
        """Lazy-initialise the CPU pool for post-parse work."""
        if self._pool is None:
            self._pool = mp.Pool(
                processes=self.max_workers,
                initializer=_post_parse_init,
                initargs=(self.bond_dim, self.embedding_dim, self.rules),
                maxtasksperchild=self.max_tasks_per_child,
            )
        return self._pool

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        logger.info(f"Starting CCGCompilerStep for chunk of size {len(df)}")
        if self.text_column not in df.columns:
            raise ValueError(f"Column '{self.text_column}' not found in DataFrame.")

        # 1. Hash texts for dedup + LMDB keys
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

        # 2. Anti-join against LMDB to skip cached entries
        unseen_mask = [True] * len(hashes)
        if self.lmdb_path.exists():
            try:
                env = lmdb.open(str(self.lmdb_path), max_readers=128, readonly=True, lock=False)
                with env.begin() as txn:
                    for i, h in enumerate(hashes):
                        if txn.get(h.encode("utf-8")) is not None:
                            unseen_mask[i] = False
                env.close()
            except lmdb.Error:
                # path exists but no LMDB env yet — treat as empty cache
                pass

        unseen_indices = [i for i, m in enumerate(unseen_mask) if m]
        unseen_texts = [texts[i] for i in unseen_indices]
        cached_count = len(texts) - len(unseen_texts)
        logger.info(f"Cache check: {cached_count} cached, {len(unseen_texts)} need compilation.")

        # 3. Parse on (one) device, post-parse on N workers, streamed
        compiled_map: dict[str, bytes] = {}
        if unseen_texts:
            logger.info(
                f"Parsing on {self.device} (parser_batch_size={self.parser_batch_size}); "
                f"post-parse across {self.max_workers} CPU workers."
            )
            processor = self._get_processor()
            pool = self._get_pool()

            n_batches = (len(unseen_texts) + self.parser_batch_size - 1) // self.parser_batch_size

            def _stream_pairs():
                """Yield (text, tree) pairs lazily, batching BERT calls.

                Pool workers consume from this generator via imap_unordered;
                while one BERT batch is in flight on the device, workers can
                still be processing trees from the previous batch.
                """
                for b_idx, start in enumerate(range(0, len(unseen_texts), self.parser_batch_size)):
                    batch = unseen_texts[start:start + self.parser_batch_size]
                    try:
                        out = processor.sentences2trees(batch, suppress_exceptions=True)
                        trees = out["lemma_trees"]
                    except Exception as e:
                        logger.warning(f"Batch {b_idx + 1}/{n_batches} parse failed entirely: {e}")
                        trees = [None] * len(batch)
                    logger.info(f"Parsed batch {b_idx + 1}/{n_batches} ({len(batch)} sentences)")
                    for text, tree in zip(batch, trees):
                        yield (text, tree)

            processed = 0
            for res in pool.imap_unordered(_post_parse_one, _stream_pairs(), chunksize=4):
                processed += 1
                h = hashlib.sha256(res["text"].encode("utf-8")).hexdigest()
                payload = {
                    "diagram": res["diagram"],
                    "symbols": res["symbols"],
                    "error": res["error"],
                }
                if res["error"]:
                    logger.warning(
                        f"Compilation error for text '{res['text'][:50]}...': {res['error']}"
                    )
                compiled_map[h] = json.dumps(payload).encode("utf-8")

            logger.info(f"Compilation complete ({processed} sentences).")

        # 4. Attach compiled bytes (None for cached/failed rows)
        compiled_bytes = [compiled_map.get(h) for h in hashes]
        df = df.with_columns(pl.Series("compiled_bytes", compiled_bytes, dtype=pl.Binary))

        gc.collect()
        return df

    def teardown(self) -> None:
        """Gracefully shut down the worker pool. Call after pipeline.run()."""
        if self._pool is not None:
            try:
                self._pool.close()
                self._pool.join()
            except Exception:
                pass
            finally:
                self._pool = None

    def __del__(self):
        try:
            import sys

            if mp is not None and getattr(sys, "is_finalizing", lambda: False)():
                return
            self.teardown()
        except Exception:
            pass
