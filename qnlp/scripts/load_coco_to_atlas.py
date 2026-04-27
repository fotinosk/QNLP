from qnlp.core.data_engine.atlas.atlas import Atlas

if __name__ == "__main__":
    # coco_atlas = Atlas.create_atlas(
    #     name="coco", source_path_or_url="hf://datasets/Multimodal-Fatima/COCO_captions_train/data/train-*-of-*.parquet"
    # )
    coco_atlas = Atlas.load_atlas(atlas_metadata_location="data/atlases/coco/metadata.json")
    coco_atlas.ingest_data_from_remote(1000)
