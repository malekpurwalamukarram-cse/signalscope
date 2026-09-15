# signalscope/model/inference.py
"""
SignalScope - SIH 2026 Production Inference Module.
MobileNetV3-Small classifier with calibrated threshold and metadata fusion.
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import numpy as np

# ---------------------------------------------------------------------------
# Checkpoint resolution: prefer the CIFAKE-trained checkpoint (which contains
# a properly calibrated threshold and was validated), fall back to the old
# production pickle only if the CIFAKE one doesn't exist.
# ---------------------------------------------------------------------------
_WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
_CIFAKE_WEIGHTS = _WEIGHTS_DIR / "cifake_384_finetuned" / "signalscope_cifake_best.pth"
_LEGACY_WEIGHTS = _WEIGHTS_DIR / "signalscope_production_384px.pth"

_PREPROCESS = T.Compose([
    T.Resize((384, 384)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Module-level cache — model loaded once per process.
_MODEL_CACHE = {}

# Default AI-probability threshold (used only when no calibrated value is
# stored in the checkpoint).  The training script (`train_cifake.py`) saves
# the optimal threshold under the key `ai_decision_threshold`; we prefer that.
_DEFAULT_AI_THRESHOLD = 0.55  # 55% as a safe fallback
_CALIBRATED_THRESHOLD = None  # populated on first load


def _build_mobilenet_v3_small(num_classes: int = 2) -> nn.Module:
    """Construct the same MobileNetV3-Small head used by train_cifake.py."""
    model = models.mobilenet_v3_small(weights=None)  # architecture only
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def get_model(device=None):
    """Load (once) and return the SignalScope model + the device it's on."""
    global _CALIBRATED_THRESHOLD

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device) if not isinstance(device, torch.device) else device

    cache_key = str(device)
    if cache_key not in _MODEL_CACHE:
        if _CIFAKE_WEIGHTS.exists():
            # ── Preferred path: CIFAKE-trained state_dict checkpoint ──
            checkpoint = torch.load(_CIFAKE_WEIGHTS, map_location=device, weights_only=False)
            model = _build_mobilenet_v3_small(num_classes=2)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            model.to(device)
            _MODEL_CACHE[cache_key] = model

            # Read the calibrated threshold saved during training
            if "ai_decision_threshold" in checkpoint:
                _CALIBRATED_THRESHOLD = float(checkpoint["ai_decision_threshold"])
                print(f"[SignalScope] Loaded CIFAKE checkpoint – calibrated threshold: {_CALIBRATED_THRESHOLD:.3f}")
            else:
                print("[SignalScope] Loaded CIFAKE checkpoint (no threshold stored, using default)")

        elif _LEGACY_WEIGHTS.exists():
            # ── Fallback: old-style pickled whole-model checkpoint ──
            model = torch.load(_LEGACY_WEIGHTS, map_location=device, weights_only=False)
            model.eval()
            model.to(device)
            _MODEL_CACHE[cache_key] = model
            print("[SignalScope] Loaded legacy production checkpoint (no calibrated threshold)")

        else:
            _MODEL_CACHE[cache_key] = None
            print("[SignalScope] WARNING: No model weights found!")

    return _MODEL_CACHE[cache_key], device


def predict_with_details(image_path: str, metadata: dict | None = None):
    """
    Run inference and return a detailed result dict.

    Args:
        image_path: Path to the image file.
        metadata:   Optional provenance metadata dict (from inspect_metadata()).
                    When provided, the confidence_shift is used to nudge
                    borderline verdicts.
    """
    model, device = get_model()

    if model is None:
        return {
            "class": "Weights Missing",
            "confidence": 0.0,
            "real_prob": 50.0,
            "ai_prob": 50.0,
            "model": "MobileNetV3 (Uninitialized)",
            "device": str(device),
        }

    raw_image = Image.open(image_path).convert("RGB")

    # ── MobileNet forward pass ────────────────────────────────────────────
    input_tensor = _PREPROCESS(raw_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()

    # Output indices: [0] = AI-Generated, [1] = Real
    ai_prob = float(probs[0][0] * 100.0)
    real_prob = float(probs[0][1] * 100.0)

    # ── Decision threshold ────────────────────────────────────────────────
    # Use the calibrated threshold from the checkpoint if available,
    # otherwise fall back to the default.
    threshold_frac = _CALIBRATED_THRESHOLD if _CALIBRATED_THRESHOLD is not None else _DEFAULT_AI_THRESHOLD
    decision_threshold = threshold_frac * 100.0  # convert fraction → percentage

    # ── Metadata confidence shift (secondary signal) ──────────────────────
    # If EXIF metadata is available, apply a soft shift to the AI probability.
    # This only moves the needle for genuinely borderline cases; a strong
    # model signal (e.g. 90% AI) won't be overridden by metadata alone.
    metadata_shift = 0.0
    metadata_verdict_str = None
    if metadata is not None:
        # confidence_shift: +0.35 for camera EXIF, -0.35 for AI software tags
        raw_shift = metadata.get("confidence_shift", 0.0)
        # Scale: a ±0.35 shift maps to ~±10 percentage-point adjustment
        metadata_shift = raw_shift * 28.0  # ±0.35 → ±~10pp
        metadata_verdict_str = metadata.get("metadata_verdict")

    # Apply the metadata nudge to the AI probability for decision purposes
    adjusted_ai_prob = ai_prob - metadata_shift  # camera EXIF → lower AI prob
    adjusted_ai_prob = max(0.0, min(100.0, adjusted_ai_prob))

    # ── Verdict ───────────────────────────────────────────────────────────
    if adjusted_ai_prob >= decision_threshold:
        final_class = "AI-Generated"
        confidence = ai_prob  # report raw model confidence, not adjusted
    else:
        final_class = "Real"
        confidence = real_prob

    # Flag borderline cases for manual review
    needs_review = (
        abs(ai_prob - decision_threshold) <= 10.0  # within ±10pp of threshold
    )

    result = {
        "class": final_class,
        "confidence": confidence,
        "real_prob": real_prob,
        "ai_prob": ai_prob,
        "needs_review": needs_review,
        "model": "MobileNetV3-Small (CIFAKE-Calibrated)",
        "device": str(device),
        "decision_threshold": decision_threshold,
        # Surface whether the threshold actually came from the checkpoint's
        # validation-calibrated value, or the hardcoded fallback — so callers
        # (UI/CLI) don't silently present an uncalibrated threshold as if it
        # were calibrated.
        "threshold_calibrated": _CALIBRATED_THRESHOLD is not None,
    }

    if metadata_verdict_str:
        result["metadata_verdict"] = metadata_verdict_str
    if abs(metadata_shift) > 0.01:
        result["metadata_shift_applied"] = round(metadata_shift, 2)

    return result


def predict(image_path: str) -> str:
    """Simple label-only entry point."""
    return predict_with_details(image_path)["class"]