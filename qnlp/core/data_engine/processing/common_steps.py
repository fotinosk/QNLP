import polars as pl


class SchemaMappingStep:
    """Renames raw columns to standard pipeline column names."""

    def __init__(self, column_mapping: dict[str, str]):
        self.column_mapping = column_mapping

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        rename_dict = {k: v for k, v in self.column_mapping.items() if k in df.columns}
        return df.rename(rename_dict) if rename_dict else df


class RemoveTrailingDotsStep:
    """Strips trailing dots from text — prevents Bobcat CCG parsing failures."""

    def __init__(self, text_column: str = "processed_text"):
        self.text_column = text_column

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.text_column not in df.columns:
            return df
        return df.with_columns(pl.col(self.text_column).str.replace(r"\.+$", "").alias(self.text_column))
