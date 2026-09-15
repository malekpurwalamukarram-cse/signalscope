"""Download the CIFAKE train/test parquet files and generate a 384px variant."""

from pathlib import Path
import sys

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPO_ID = "dragonintelligence/CIFAKE-image-dataset"
DATA_DIR = ROOT / "data" / "cifake"
FILES = (
    "data/train-00000-of-00001.parquet",
    "data/test-00000-of-00001.parquet",
)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading CIFAKE into {DATA_DIR}", flush=True)
    for filename in FILES:
        print(f"Downloading {filename}...", flush=True)
        downloaded = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
            local_dir=str(DATA_DIR),
            force_download=True,
        )
        print(f"Saved: {downloaded}", flush=True)

    print("CIFAKE source dataset ready; training resizes images to 384px on the fly.", flush=True)


if __name__ == "__main__":
    main()
