import argparse
import os
from collections import defaultdict

import clip
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from qnlp.discoclip2.models.lookup_embeddings import LookupEmbedding


def encode_images(args):
    """
    Encodes images from the ARO dataset using CLIP and saves them to a lookup embedding.
    """
    clip_model_name = args.clip_model
    model, processor = clip.load(clip_model_name, device=args.device)
    image_folder = args.image_folder

    df_vgr = pd.read_json(args.vgr_path)
    df_vga = pd.read_json(args.vga_path)

    images_to_encode = defaultdict(set)

    for i, row in tqdm(df_vgr.iterrows(), total=len(df_vgr), desc="Processing VGR"):
        images_to_encode[row["image_path"]].add((row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"]))
    for i, row in tqdm(df_vga.iterrows(), total=len(df_vga), desc="Processing VGA"):
        images_to_encode[row["image_path"]].add((row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"]))

    print(f"{sum([len(bboxes) for bboxes in images_to_encode.values()])} images to encode")

    labels = []
    embeddings = []

    for filename, bboxes in tqdm(images_to_encode.items(), desc="Encoding Images"):
        path = os.path.join(image_folder, filename)
        try:
            image = Image.open(path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: File not found at {path}. Skipping.")
            continue

        inputs = processor(image).to(args.device)
        with torch.no_grad():
            embedding = model.encode_image(inputs.unsqueeze(0))
        labels.append(filename)
        embeddings.append(embedding.squeeze(0))

        for bbox in bboxes:
            x, y, w, h = bbox
            image_cropped = image.crop((x, y, x + w, y + h))
            inputs = processor(image_cropped).to(args.device)
            with torch.no_grad():
                embedding = model.encode_image(inputs.unsqueeze(0))
            file_name = filename.split(".")[0]
            labels.append(f"{file_name}_{x}_{y}_{w}_{h}.jpg")
            embeddings.append(embedding.squeeze(0))

    lookup = LookupEmbedding.from_embeddings(labels, embeddings)
    lookup.save_to_checkpoint(args.output_path)


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Encode ARO images using CLIP")
    parser.add_argument("--clip-model", type=str, default="ViT-B/32", help="CLIP model to use")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for encoding")
    parser.add_argument(
        "--image-folder",
        type=str,
        default="data/aro/raw/images",
        help="Path to ARO image folder",
    )
    parser.add_argument(
        "--vgr-path",
        type=str,
        default="data/aro/raw/visual_genome_relation.json",
        help="Path to Visual Genome Relation JSON file",
    )
    parser.add_argument(
        "--vga-path",
        type=str,
        default="data/aro/raw/visual_genome_attribution.json",
        help="Path to Visual Genome Attribution JSON file",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save the output embeddings",
    )

    args = parser.parse_args()

    if args.output_path is None:
        safe_model = args.clip_model.replace("/", "_")
        args.output_path = f"models/lookup_embedding_{safe_model}.pt"

    return args


if __name__ == "__main__":
    args = parse_args()
    encode_images(args)
