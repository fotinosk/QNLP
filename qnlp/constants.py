from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Constants(BaseSettings):
    constants_config = SettingsConfigDict(env_prefix="global_constants_", frozen=True)

    embedding_dim: int = 512
    atlases_path = Path("data/atlases/")
    lmdb_path = Path("sentence_mapping/")
