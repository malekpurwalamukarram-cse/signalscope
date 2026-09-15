"""Train and compare matched CIFAKE models at several input resolutions."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train_cifake import (
    ParquetCIFAKEDataset,
    build_cifake_model,
    get_transforms,
    resolve_dataset_files,
)


def measure_inference(model, loader, device, warmup=10):
    model.eval()
    batches = list(loader)
    for images, _ in batches[:warmup]:
        with torch.inference_mode():
            model(images.to(device))
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    samples = 0
    with torch.inference_mode():
        for images, _ in batches:
            model(images.to(device))
            samples += images.size(0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        if device.type == "cuda" else 0.0
    )
    return {
        "inference_ms_per_image": elapsed * 1000 / samples,
        "inference_images_per_second": samples / elapsed,
        "peak_gpu_memory_mb": peak_mb,
    }


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_path, test_path = resolve_dataset_files(args.data_dir)
    results = []

    for size in args.sizes:
        out_dir = Path(args.out_dir) / str(size)
        checkpoint = out_dir / "signalscope_cifake_best.pth"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--run-training",
            "--data_dir", str(args.data_dir),
            "--img_size", str(size),
            "--train_samples", str(args.train_samples),
            "--test_samples", str(args.test_samples),
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--out_dir", str(out_dir),
        ]
        subprocess.run(command, check=True)

        checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=False)
        model = build_cifake_model().to(device)
        model.load_state_dict(checkpoint_data["model_state_dict"])
        _, val_transform = get_transforms(size)
        dataset = ParquetCIFAKEDataset(test_path, args.test_samples, val_transform)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        result = {
            "resolution": size,
            "accuracy": checkpoint_data["accuracy"],
            "macro_f1": checkpoint_data["macro_f1"],
            "roc_auc": checkpoint_data["roc_auc"],
            **measure_inference(model, loader, device),
        }
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)

    Path(args.results).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-training", action="store_true")
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--sizes", nargs="+", type=int, default=[32, 224, 384])
    parser.add_argument("--data_dir", default="data/cifake")
    parser.add_argument("--train_samples", type=int, default=10000)
    parser.add_argument("--test_samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out_dir", default="signalscope/model/weights/resolution_benchmark")
    parser.add_argument("--results", default="signalscope/model/weights/resolution_benchmark.json")
    args = parser.parse_args()
    if args.run_training:
        from train_cifake import main as train_main
        train_parser = argparse.Namespace(
            data_dir=args.data_dir, img_size=args.img_size,
            train_samples=args.train_samples, test_samples=args.test_samples,
            epochs=args.epochs, batch_size=args.batch_size, lr=5e-4,
            out_dir=args.out_dir, resume=None, num_workers=0, real_world_dir=None,
        )
        train_main(train_parser)
    else:
        main(args)       