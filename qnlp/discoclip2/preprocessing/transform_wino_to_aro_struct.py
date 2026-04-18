"""
Transform to ARO style with 1 image and a positive and negative samlpe
"""

import json
import os

from datasets import load_dataset

OUTPUT_PATH = "data/winoground/raw"


if __name__ == "__main__":
    ds_dict = load_dataset("facebook/winoground")
    ds = ds_dict.get("test")

    output_dir = OUTPUT_PATH
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    print(f"Saving {len(ds)} examples to {output_dir}...")
    image_count = 0

    train_list = []
    test_list = []
    val_list = []

    for example in ds:
        # Winoground examples have two images and two captions
        img0_id = str(image_count)
        img1_id = str(image_count + 1)

        # Save the Image files
        # example["image_0"].convert("RGB").save(os.path.join(images_dir, f"{img0_id}.jpg"))
        # example["image_1"].convert("RGB").save(os.path.join(images_dir, f"{img1_id}.jpg"))

        # Store the text and metadata
        sample1 = {
            "image_id": img0_id,
            "true_caption": example["caption_0"],
            "false_caption": example["caption_1"],
            "image_path": os.path.join(images_dir, f"{img0_id}.jpg"),
        }

        sample2 = {
            "image_id": img1_id,
            "true_caption": example["caption_0"],
            "false_caption": example["caption_1"],
            "image_path": os.path.join(images_dir, f"{img0_id}.jpg"),
        }
        train_list.append(sample1)
        if image_count % 4 == 0:
            test_list.append(sample2)
        else:
            val_list.append(sample2)

        image_count += 2

    with open(os.path.join(output_dir, "test.json"), "w") as f:
        json.dump(test_list, f, indent=4)

    with open(os.path.join(output_dir, "val.json"), "w") as f:
        json.dump(val_list, f, indent=4)

    with open(os.path.join(output_dir, "train.json"), "w") as f:
        json.dump(train_list, f, indent=4)

    print("Done! Check the 'winoground_local' folder.")
