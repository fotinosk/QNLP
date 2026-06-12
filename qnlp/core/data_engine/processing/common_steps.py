import polars as pl


class SchemaMappingStep:
    """Renames raw columns to standard pipeline column names."""

    def __init__(self, column_mapping: dict[str, str]):
        self.column_mapping = column_mapping

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        rename_dict = {k: v for k, v in self.column_mapping.items() if k in df.columns}
        return df.rename(rename_dict) if rename_dict else df


class FlattenContrastivePairStep:
    """Flattens each row holding a (true_caption, false_caption) pair into two
    atoms sharing the same sample_id: one labelled True (the positive) and one
    labelled False (the negative).

    Downstream, ContrastivePairStrategy's labeled mode regroups by sample_id to
    reconstruct the contrastive pair. All other columns (sample_id,
    local_image_path, split, …) are carried onto both atoms.
    """

    def __init__(
        self,
        true_column: str = "true_caption",
        false_column: str = "false_caption",
        text_column: str = "processed_text",
        label_column: str = "label",
    ):
        self.true_column = true_column
        self.false_column = false_column
        self.text_column = text_column
        self.label_column = label_column

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.true_column not in df.columns or self.false_column not in df.columns:
            return df

        shared = [c for c in df.columns if c not in (self.true_column, self.false_column)]
        positives = (
            df.select(shared + [self.true_column])
            .rename({self.true_column: self.text_column})
            .with_columns(pl.lit(True).alias(self.label_column))
        )
        negatives = (
            df.select(shared + [self.false_column])
            .rename({self.false_column: self.text_column})
            .with_columns(pl.lit(False).alias(self.label_column))
        )
        return pl.concat([positives, negatives], how="vertical_relaxed")


class RemoveTrailingDotsStep:
    """Strips trailing dots from text — prevents Bobcat CCG parsing failures."""

    def __init__(self, text_column: str = "processed_text"):
        self.text_column = text_column

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.text_column not in df.columns:
            return df
        return df.with_columns(pl.col(self.text_column).str.replace(r"\.+$", "").alias(self.text_column))
