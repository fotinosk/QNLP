from typing import Protocol

import polars as pl


class CompositionStrategy(Protocol):
    def compose(self, atoms: pl.DataFrame) -> pl.DataFrame:
        """
        Transform a DataFrame of enriched atoms into composite samples.

        Input columns (always present):
            sample_id, local_image_path, processed_text, text_hash, diagram, symbols

        Additional columns may be present depending on the preprocessing pipeline
        (e.g. `label` for datasets with explicit positive/negative pairs).

        Returns a DataFrame in the task-specific schema. The output schema is
        entirely defined by the strategy — the dataset creator infrastructure
        does not inspect or enforce it.
        """
        ...
