from qnlp.constants import constants
from qnlp.core.data_engine.processing.compiler_step import CCGCompilerStep
from qnlp.core.data_engine.processing.pipeline import Pipeline
from qnlp.preprocessing_pipelines.coco.steps import COCOFlattenStep, SchemaMappingStep

flatten_step = COCOFlattenStep("sentences_raw")
schema_step = SchemaMappingStep(column_mapping={"sentences_raw": "processed_text"})
ccg_parsing_step = CCGCompilerStep(
    lmdb_path=constants.lmdb_path, bond_dim=constants.bond_dim, embedding_dim=constants.embedding_dim
)

coco_atlas = constants.atlases_path / "coco"

coco_pipeline = Pipeline(
    atlas_dir=coco_atlas,
    steps=[flatten_step, schema_step, ccg_parsing_step],
    derived_name="derived_test",
)


if __name__ == "__main__":
    coco_pipeline.run(chunk_size=100, dry_run=True)
