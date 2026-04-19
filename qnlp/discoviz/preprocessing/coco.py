from datasets import load_dataset

if __name__ == "__main__":
    dataset = load_dataset(
        "Multimodal-Fatima/COCO_captions_train",
        # "Multimodal-Fatima/COCO_captions_validation",
        # "Multimodal-Fatima/COCO_captions_test",
        streaming=True,
        split="train",
    )
    subset = dataset.take(5)

    subset_list = list(subset)

    print(f"Downloaded {len(subset_list)} samples")
    print(subset_list[0].keys())
