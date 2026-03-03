import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from tqdm import tqdm


def download_image(img_id, url, image_dir):
    """
    Downloads an image and saves it as {img_id}.jpg.
    Skips if the file already exists and is not empty.
    """
    if not isinstance(url, str) or not url.startswith("http"):
        return

    # Construct the local path
    path = os.path.join(image_dir, f"{img_id}.jpg")

    # --- The "Crashtolerance" Check ---
    # We check if path exists AND if the file size is greater than 0
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return

    try:
        # Use a stream to handle potential large files gracefully
        r = requests.get(url, timeout=10, stream=True)
        if r.status_code == 200:
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except Exception:
        # If it fails, we don't want a partial/corrupt file sitting there
        if os.path.exists(path):
            os.remove(path)


def download_svo_images(datasets: list[str], image_dir: str):
    os.makedirs(image_dir, exist_ok=True)

    # Using a dictionary to map IDs to URLs ensures unique downloads
    id_to_url = {}

    for ds in datasets:
        try:
            df = pd.read_csv(ds)
            # Combine pos and neg mappings
            id_to_url.update(dict(zip(df["pos_image_id"], df["pos_url"])))
            id_to_url.update(dict(zip(df["neg_image_id"], df["neg_url"])))
        except Exception as e:
            print(f"Error reading {ds}: {e}")

    # Clean up any NaN values
    id_to_url = {k: v for k, v in id_to_url.items() if pd.notna(k) and pd.notna(v)}

    print(f"Total unique images to verify/download: {len(id_to_url)}")

    # Use ThreadPoolExecutor for speed
    with ThreadPoolExecutor(max_workers=20) as executor:
        # Pass the items (ID, URL) to the downloader
        list(
            tqdm(
                executor.map(lambda item: download_image(item[0], item[1], image_dir), id_to_url.items()),
                total=len(id_to_url),
                desc="Downloading SVO Images",
            )
        )


if __name__ == "__main__":
    datasets = [
        "data/svo/processed/test.csv",
        "data/svo/processed/val.csv",
        "data/svo/processed/train.csv",
    ]
    image_dir = "data/svo/raw/images"
    download_svo_images(datasets, image_dir)
