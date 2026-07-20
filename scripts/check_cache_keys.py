import itertools

import diskcache
import polars as pl
from nltk.tokenize.treebank import TreebankWordTokenizer


def main():
    c = diskcache.Cache("/SAN/intelsys/discoviz/fotinos/QNLP/.cache/lambeq/bobcat/diskcache")

    print("--- sample raw keys ---", flush=True)
    for k in itertools.islice(c.iterkeys(), 5):
        print(repr(k)[:200], flush=True)

    df = (
        pl.read_parquet(
            "/SAN/intelsys/discoviz/fotinos/QNLP/data/datasets/coco_single_caption_nlc_train.parquet",
            columns=["text_hash", "processed_text"],
        )
        .unique(subset=["text_hash"])
        .sort("text_hash")
        .head(8)
    )

    tk = TreebankWordTokenizer()
    print("--- our reconstructed keys ---", flush=True)
    for th, text in df.iter_rows():
        key = str((tk.tokenize(text), True, False))
        hit = key in c
        print(("HIT " if hit else "MISS"), repr(key)[:180], flush=True)


if __name__ == "__main__":
    main()
