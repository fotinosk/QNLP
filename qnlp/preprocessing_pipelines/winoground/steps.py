from pathlib import Path

import polars as pl


class WinogroundFlattenStep:
    """
    Expands each Winoground item (2 images, 2 captions) into 2 rows.

    The pairing is strictly enforced:
        caption_0  <->  local_image_0_path  ->  sample_id suffix __0
        caption_1  <->  local_image_1_path  ->  sample_id suffix __1

    The original sample_id is recoverable by splitting on "__" so that
    pairs can be reconstructed after the pipeline runs.
    """

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        resolve = lambda p: str(Path(p).resolve())  # noqa: E731

        rows_0 = df.select(
            [
                (pl.col("sample_id") + "__0").alias("sample_id"),
                pl.col("local_image_0_path").map_elements(resolve, return_dtype=pl.String).alias("local_image_path"),
                pl.col("caption_0").alias("processed_text"),
            ]
        )

        rows_1 = df.select(
            [
                (pl.col("sample_id") + "__1").alias("sample_id"),
                pl.col("local_image_1_path").map_elements(resolve, return_dtype=pl.String).alias("local_image_path"),
                pl.col("caption_1").alias("processed_text"),
            ]
        )

        # Interleave so both rows of a pair stay adjacent in the output chunk
        return pl.concat([rows_0, rows_1]).sort("sample_id")
