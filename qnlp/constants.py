from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


class Constants(BaseSettings):
    constants_config: Any = SettingsConfigDict(env_prefix="global_constants_", frozen=True)

    embedding_dim: int = 512
    bond_dim: int = 10
    atlases_path: Path = Path("data/atlases/")
    lmdb_path: Path = Path("data/sentence_mapping/")
    splits_path: Path = Path("data/splits/")
    logs_path: Path = Path("runs/logs/")
    checkpoints_path: Path = Path("runs/checkpoints/")
    datasets_path: Path = Path("data/datasets/")


constants = Constants()
