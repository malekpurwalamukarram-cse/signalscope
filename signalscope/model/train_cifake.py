"""
SignalScope - CIFAKE Training Pipeline (PS-2)
Trains MobileNetV3-Small / EfficientNet on the CIFAKE 100k benchmark dataset
to achieve 95%+ accuracy on Real vs. AI-Generated image detection.
"""

import io
import os
import random
import sys
import time
import json
import argparse
from pathlib import Path

from PIL import Image

# Windows console UTF-8 safety
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
from PIL import Image
from huggingface_hub import hf_hub_download
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, Dataset, DataLoader, Subset
from torchvision import datasets, transforms, models


def load_cifake_frame(parquet_path: str | Path, max_samples: int | None = None) -> pd.DataFrame:
    """Load all rows or a bounded balanced sample without duplicating the full parquet."""
    if max_samples is None:
        return pd.read_parquet(parquet_path)

    import pyarrow.parquet as pq

    per_class = max(1, max_samples // 2)
    reservoirs = {0: [], 1: []}
    seen = {0: 0, 1: 0}
    rng = random.Random(42)
    parquet = pq.ParquetFile(parquet_path)
    for batch in parquet.iter_batches(batch_size=256, columns=["label", "image"]):
        for row in batch.to_pylist():
            label = int(row["label"])
            if label not in reservoirs:
                continue
            seen[label] += 1
            reservoir = reservoirs[label]
            if len(reservoir) < per_class:
                reservoir.append(row)
            else:
                position = rng.randrange(seen[label])
                if position < per_class:
                    reservoir[position] = row

    rows = reservoirs[0] + reservoirs[1]
    rng.shuffle(rows)
    return pd.DataFrame(rows)


class ParquetCIFAKEDataset(Dataset):
    """Memory-efficient dataset reading directly from CIFAKE parquet bytes."""

    def __init__(self, parquet_path: str, max_samples: int | None = None, transform=None):
        self.df = load_cifake_frame(parquet_path, max_samples)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_bytes = row["image"]["bytes"]
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        label = int(row["label"])  # 0: AI-Generated (Fake), 1: Real
        if self.transform:
            img = self.transform(img)
        return img, label


class RealWorldFolderDataset(Dataset):
    """Load labeled camera, screenshot, and web images from class folders."""

    def __init__(self, root: str | Path, transform=None):
        folder = datasets.ImageFolder(str(root), allow_empty=True)
        allowed = {"AI-Generated": 0, "AI": 0, "Fake": 0, "Real": 1}
        unknown = set(folder.class_to_idx) - set(allowed)
        if unknown:
            raise ValueError(
                f"Unsupported class folders in {root}: {sorted(unknown)}. "
                "Use Real/ and optionally AI-Generated/."
            )
        self.samples = [(path, allowed[folder.classes[label]]) for path, label in folder.samples]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def prepare_384_dataset(source_dir: str | Path, out_dir: str | Path, max_samples: int | None = None) -> tuple[Path, Path]:
    """Create a 384px version of the CIFAKE parquet dataset in a dedicated output folder."""
    src_dir = Path(source_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_src = src_dir / "train-00000-of-00001.parquet"
    test_src = src_dir / "test-00000-of-00001.parquet"
    if not train_src.exists():
        train_src = src_dir / "data" / train_src.name
    if not test_src.exists():
        test_src = src_dir / "data" / test_src.name

    if not train_src.exists() or not test_src.exists():
        raise FileNotFoundError(f"Missing CIFAKE parquet source files under {src_dir}")

    def resize_parquet(path: Path, out_path: Path):
        import pyarrow as pa
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        total = parquet.metadata.num_rows
        processed = 0
        writer = None
        try:
            for batch in parquet.iter_batches(batch_size=256, columns=["label", "image"]):
                rows = []
                for row in batch.to_pylist():
                    img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
                    img = img.resize((384, 384), Image.Resampling.BILINEAR)
                    buffer = io.BytesIO()
                    img.save(buffer, format="PNG")
                    rows.append({
                        "label": int(row["label"]),
                        "image": {"bytes": buffer.getvalue()},
                    })

                table = pa.Table.from_pylist(rows)
                if writer is None:
                    writer = pq.ParquetWriter(out_path, table.schema)
                writer.write_table(table)
                processed += len(rows)
                print(f"  {path.name}: {processed}/{total} images resized", flush=True)
        finally:
            if writer is not None:
                writer.close()

    train_out = out / "train-00000-of-00001.parquet"
    test_out = out / "test-00000-of-00001.parquet"
    resize_parquet(train_src, train_out)
    resize_parquet(test_src, test_out)
    return train_out, test_out


def resolve_dataset_files(data_dir: str) -> tuple[Path, Path]:
    """Resolve the original CIFAKE files; transforms resize images during training."""
    root = Path(data_dir)

    local_train = root / "train-00000-of-00001.parquet"
    local_test = root / "test-00000-of-00001.parquet"
    nested_train = root / "data" / local_train.name
    nested_test = root / "data" / local_test.name

    if local_train.exists() and local_test.exists():
        return local_train, local_test
    if nested_train.exists() and nested_test.exists():
        return nested_train, nested_test

    root.mkdir(parents=True, exist_ok=True)
    repo = "dragonintelligence/CIFAKE-image-dataset"
    train_path = hf_hub_download(
        repo_id=repo,
        filename="data/train-00000-of-00001.parquet",
        repo_type="dataset",
        local_dir=str(root),
    )
    test_path = hf_hub_download(
        repo_id=repo,
        filename="data/test-00000-of-00001.parquet",
        repo_type="dataset",
        local_dir=str(root),
    )
    return root / "data" / local_train.name, root / "data" / local_test.name


def get_transforms(img_size: int = 384):
    train_tf = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomResizedCrop(img_size, scale=(0.70, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.04),
        ], p=0.8),
        transforms.RandomApply([transforms.RandomRotation(10)], p=0.35),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.25),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        transforms.RandomErasing(p=0.10, scale=(0.02, 0.12), value="random"),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return train_tf, val_tf


def build_cifake_model():
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 2),
    )
    return model


def choose_ai_threshold(labels, real_probs, max_false_positive_rate: float) -> tuple[float, float]:
    """Choose the highest-F1 AI threshold subject to a real-photo FPR limit."""
    labels = np.asarray(labels)
    ai_probs = 1.0 - np.asarray(real_probs)
    real_count = max(1, int((labels == 1).sum()))
    candidates = []
    for threshold in np.linspace(0.05, 0.95, 181):
        predicted_ai = ai_probs >= threshold
        false_positive_rate = float(((predicted_ai) & (labels == 1)).sum() / real_count)
        if false_positive_rate <= max_false_positive_rate:
            predicted_labels = np.where(predicted_ai, 0, 1)
            score = f1_score(labels, predicted_labels, average="macro", zero_division=0)
            candidates.append((score, threshold, false_positive_rate))

    if not candidates:
        return 0.95, 0.0
    _, threshold, false_positive_rate = max(candidates)
    return float(threshold), float(false_positive_rate)


def evaluate_model(model, loader, device):
    model.eval()
    labels, real_probs = [], []
    with torch.no_grad():
        for imgs, batch_labels in loader:
            imgs = imgs.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                logits = model(imgs)
            real_probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
            labels.extend(batch_labels.tolist())
    return labels, real_probs


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"[SignalScope CIFAKE] Training on device: {device}")

    print(f"[SignalScope CIFAKE] Loading dataset from {args.data_dir}...")
    train_p, test_p = resolve_dataset_files(args.data_dir)
    print(f"[SignalScope CIFAKE] Dataset files resolved: train={train_p}, test={test_p}")

    train_tf, val_tf = get_transforms(args.img_size)
    print(f"[SignalScope CIFAKE] Loading {args.train_samples} train and {args.test_samples} test samples...")
    train_source = ParquetCIFAKEDataset(train_p, max_samples=args.train_samples, transform=train_tf)
    val_source   = ParquetCIFAKEDataset(train_p, max_samples=args.train_samples, transform=val_tf)
    indices = np.arange(len(train_source))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=args.validation_fraction,
        random_state=42,
        stratify=train_source.df["label"].to_numpy(),
    )
    train_ds = Subset(train_source, train_indices.tolist())
    val_ds   = Subset(val_source, val_indices.tolist())
    test_ds  = ParquetCIFAKEDataset(test_p,  max_samples=args.test_samples,  transform=val_tf)
    if args.real_world_dir:
        real_world_ds = RealWorldFolderDataset(args.real_world_dir, transform=train_tf)
        if not len(real_world_ds):
            raise ValueError(f"No images found under {args.real_world_dir}")
        train_ds = ConcatDataset([train_ds, real_world_ds])
        print(f"[SignalScope CIFAKE] Added {len(real_world_ds)} real-world labeled images from {args.real_world_dir}")
    if args.real_world_val_dir:
        real_world_val_ds = RealWorldFolderDataset(args.real_world_val_dir, transform=val_tf)
        if not len(real_world_val_ds):
            raise ValueError(f"No images found under {args.real_world_val_dir}")
        val_ds = ConcatDataset([val_ds, real_world_val_ds])
        print(f"[SignalScope CIFAKE] Added {len(real_world_val_ds)} real-world validation images")
    print(f"[SignalScope CIFAKE] Loaded train={len(train_ds)}, validation={len(val_ds)}, test={len(test_ds)} samples")
    if args.train_samples is not None and len(train_ds) < args.train_samples:
        raise ValueError(
            f"Training parquet contains only {len(train_ds)} samples, "
            f"but --train_samples requested {args.train_samples}. "
            "Rebuild the dataset with download_cifake.py."
        )
    if args.test_samples is not None and len(test_ds) < args.test_samples:
        raise ValueError(
            f"Test parquet contains only {len(test_ds)} samples, "
            f"but --test_samples requested {args.test_samples}. "
            "Rebuild the dataset with download_cifake.py."
        )

    num_workers = args.num_workers
    print(f"[SignalScope CIFAKE] Using DataLoader workers={num_workers}, device={device}")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=device.type == "cuda")
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=device.type == "cuda")
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=device.type == "cuda")

    model = build_cifake_model().to(device)
    checkpoint_path = Path(args.resume) if args.resume else None
    if checkpoint_path is not None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        print(f"[SignalScope CIFAKE] Resumed weights from {checkpoint_path}")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_auc = -1.0
    best_threshold = 0.65

    print(f"\n[SignalScope CIFAKE] Starting {args.epochs} training epochs...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        optimizer_updates = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if not device.type == "cuda" or scaler.get_scale() >= previous_scale:
                optimizer_updates += 1

            train_loss += loss.item() * imgs.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += imgs.size(0)

        if optimizer_updates:
            scheduler.step()
        train_loss /= train_total
        train_acc = train_correct / train_total

        val_labels, val_real_probs = evaluate_model(model, val_loader, device)
        val_auc = roc_auc_score(val_labels, val_real_probs)
        threshold, val_fpr = choose_ai_threshold(
            val_labels,
            val_real_probs,
            args.max_validation_fpr,
        )
        elapsed  = time.time() - t0

        print(f"Epoch {epoch:2d}/{args.epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val AUC: {val_auc:.4f} | Val FPR: {val_fpr*100:.2f}% | AI threshold: {threshold:.3f} | {elapsed:.1f}s")

        if val_auc > best_auc:
            best_auc = val_auc
            best_threshold = threshold
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "validation_roc_auc": val_auc,
                "validation_false_positive_rate": val_fpr,
                "ai_decision_threshold": threshold,
                "classes": ["AI-Generated", "Real"],
            }
            torch.save(ckpt, out_dir / "signalscope_cifake_best.pth")
            print(f"  [OK] Saved checkpoint (Val AUC={val_auc:.4f}, FPR={val_fpr*100:.2f}%, threshold={threshold:.3f})")

    best_checkpoint = torch.load(out_dir / "signalscope_cifake_best.pth", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_labels, test_real_probs = evaluate_model(model, test_loader, device)
    test_ai_probs = 1.0 - np.asarray(test_real_probs)
    test_preds = np.where(test_ai_probs >= best_checkpoint["ai_decision_threshold"], 0, 1)
    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)
    test_auc = roc_auc_score(test_labels, test_real_probs)
    cm = confusion_matrix(test_labels, test_preds)
    print("\n=== Final Evaluation on Held-Out Test Set ===")
    print(f"Accuracy: {test_acc*100:.2f}%")
    print(f"Macro-F1: {test_f1:.4f}")
    print(f"ROC-AUC: {test_auc:.4f}")
    print(f"AI threshold: {best_checkpoint['ai_decision_threshold']:.3f}")
    print("\nConfusion Matrix [Row: Ground Truth, Col: Pred]:")
    print(f"  [AI-Gen, Real]")
    print(f"AI:   {cm[0]}")
    print(f"Real: {cm[1]}")
    print("\nClassification Report:")
    print(classification_report(
        test_labels,
        test_preds,
        target_names=["AI-Generated", "Real"],
        zero_division=0,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SignalScope on CIFAKE")
    parser.add_argument("--train_samples", type=int, default=10000, help="Balanced train samples; use 0 for the full split")
    parser.add_argument("--test_samples",  type=int, default=1000,  help="Number of test samples")
    parser.add_argument("--epochs",        type=int, default=10,    help="Number of additional epochs")
    parser.add_argument("--batch_size",    type=int, default=64,    help="Batch size")
    parser.add_argument("--img_size",      type=int, default=384,   help="Image resolution")
    parser.add_argument("--lr",            type=float, default=5e-4,help="Learning rate")
    parser.add_argument("--validation_fraction", type=float, default=0.10,
                        help="Fraction of the CIFAKE training split reserved for calibration")
    parser.add_argument("--max_validation_fpr", type=float, default=0.05,
                        help="Maximum real-photo false-positive rate for threshold selection")
    parser.add_argument("--out_dir",       type=str, default="signalscope/model/weights")
    parser.add_argument("--data_dir",      type=str, default="data/cifake")
    parser.add_argument("--real_world_dir", type=str, default=None,
                        help="Optional ImageFolder added to training with Real/ and optionally AI-Generated/")
    parser.add_argument("--real_world_val_dir", type=str, default=None,
                        help="Optional separate ImageFolder for real-world validation")
    parser.add_argument("--resume",        type=str, default=None, help="Optional checkpoint to resume from")
    parser.add_argument("--num_workers",   type=int, default=0, help="DataLoader worker processes; 0 minimizes RAM use")
    args = parser.parse_args()
    if args.train_samples == 0:
        args.train_samples = None
    main(args)

