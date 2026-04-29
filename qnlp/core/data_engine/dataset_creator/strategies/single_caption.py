import polars as pl

_OUTPUT_COLUMNS = ["sample_id", "local_image_path", "processed_text", "text_hash", "diagram", "symbols"]


class SingleCaptionStrategy:
    """
    Pass-through strategy. Each atom becomes one sample.

    Selects the standard atom columns in a fixed order, dropping any
    pipeline-specific extras (e.g. label). Works for any atlas source
    since atoms are already one (image, text) pair per row.

    Output schema:
        sample_id, local_image_path, processed_text, text_hash, diagram, symbols
    """

    def compose(self, atoms: pl.DataFrame) -> pl.DataFrame:
        available = [c for c in _OUTPUT_COLUMNS if c in atoms.columns]
        return atoms.select(available)
