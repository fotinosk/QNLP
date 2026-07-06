from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


class Constants(BaseSettings):
    constants_config: Any = SettingsConfigDict(env_prefix="global_constants_", frozen=True)

    embedding_dim: int = 512
    bond_dim: int = 10
    atlases_path: Path = Path("data/atlases/")
    splits_path: Path = Path("data/splits/")
    logs_path: Path = Path("runs/logs/")
    checkpoints_path: Path = Path("runs/checkpoints/")
    datasets_path: Path = Path("data/datasets/")

    # Parser generation selector. Empty string ("") = the original bobcat
    # artifacts: data/sentence_mapping/ (LMDB), derived_v1/ (per-atlas), and
    # unsuffixed datasets. Set to a short tag (e.g. "v2") — via the PARSER_VERSION
    # env var — to write FULLY PARALLEL artifacts for a new parser without touching
    # any bobcat data. The tag flows through the LMDB dir, the derived dir name,
    # the parser diskcache, and dataset output names.
    parser_version: str = ""

    @property
    def _parser_suffix(self) -> str:
        return f"_{self.parser_version}" if self.parser_version else ""

    @property
    def lmdb_path(self) -> Path:
        """Compiled diagram store (text_hash -> diagram/symbols). Versioned so a new
        parser gets its own store instead of hitting stale, text-keyed bobcat cache."""
        return Path(f"data/sentence_mapping{self._parser_suffix}/")

    @property
    def derived_name(self) -> str:
        """Per-atlas derived directory. Bobcat uses 'derived_v1'; a new parser writes
        to a parallel 'derived_<version>' dir so both coexist."""
        return f"derived_{self.parser_version}" if self.parser_version else "derived_v1"

    @property
    def artifact_suffix(self) -> str:
        """Appended to dataset output names so bobcat datasets are never overwritten."""
        return self._parser_suffix

    @property
    def bobcat_cache_path(self) -> Path:
        """Parser-level diskcache (raw parse trees). Versioned alongside the rest."""
        return Path(f"/SAN/intelsys/discoviz/fotinos/QNLP/.cache/lambeq/bobcat{self._parser_suffix}/diskcache")


constants = Constants()
