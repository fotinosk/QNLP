"""Build coco_contrastive_{tree}_{train,val,test}.parquet from the
derived_tree dir produced by qnlp.preprocessing_pipelines.coco.pipeline_tree.
"""
import os
from pathlib import Path

from qnlp.constants import constants
from qnlp.core.data_engine.dataset_creator.dataset_generator import create_train_val_test_datasets
from qnlp.core.data_engine.dataset_creator.strategies.contrastive_pair import ContrastivePairStrategy

# Honour PIPELINE_TREE_SCRATCH so this matches whatever the pipeline wrote.
_SCRATCH = os.environ.get("PIPELINE_TREE_SCRATCH")
if _SCRATCH:
    _ROOT = Path(_SCRATCH)
    COCO_DERIVED_DIR = _ROOT / "atlases" / "coco" / "derived_tree"
    LMDB_PATH = _ROOT / "sentence_mapping_tree"
else:
    COCO_DERIVED_DIR = constants.atlases_path / "coco" / "derived_tree"
    LMDB_PATH = constants.lmdb_path.parent / "sentence_mapping_tree"

if __name__ == "__main__":
    create_train_val_test_datasets(
        derived_dirs=[COCO_DERIVED_DIR],
        strategy=ContrastivePairStrategy(),
        output_name="coco_contrastive_tree",
        lmdb_path=LMDB_PATH,
    )
