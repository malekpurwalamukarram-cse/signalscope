"""
SignalScope - Robustness Analysis Module (Bonus C)
Evaluates verdict stability under JPEG compression and resizing degradation.
"""

import sys
import io
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

_HERE = Path(__file__).parent.parent
sys.path.insert(0, str(_HERE / "model"))


def _apply_jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _apply_resize_down(img: Image.Image, scale: float) -> Image.Image:
    w, h = img.size
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def robustness_sweep(image_path: Union[str, Path], predict_fn) -> dict:
    """
    Sweep the prediction over varying JPEG quality and resize scale.

    Args:
        image_path: Path to the image.
        predict_fn: Callable that takes an image path or PIL image and returns
                    a dict with 'class' and 'confidence' keys.

    Returns:
        dict with sweep results and stability metrics.
    """
    img = Image.open(image_path).convert("RGB")

    jpeg_qualities = [95, 80, 65, 50, 35, 20]
    resize_scales  = [1.0, 0.75, 0.5, 0.35]

    results = {"jpeg": [], "resize": [], "stable": True, "stability_score": 1.0}
    verdicts = []

    # --- JPEG sweep ---
    for q in jpeg_qualities:
        degraded = _apply_jpeg(img, q)
        # Save to temp buffer and pass to predict
        tmp = io.BytesIO()
        degraded.save(tmp, format="PNG")
        tmp.seek(0)
        tmp_img = Image.open(tmp).convert("RGB")

        # predict_fn needs a path — save temporary file
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            degraded.save(f.name)
            tmp_path = f.name
        try:
            res = predict_fn(tmp_path)
        finally:
            os.unlink(tmp_path)

        results["jpeg"].append({
            "quality": q,
            "verdict": res.get("class", res) if isinstance(res, dict) else res,
            "confidence": res.get("confidence", None) if isinstance(res, dict) else None,
        })
        verdicts.append(res.get("class", res) if isinstance(res, dict) else res)

    # --- Resize sweep ---
    for scale in resize_scales:
        degraded = _apply_resize_down(img, scale)
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            degraded.save(f.name)
            tmp_path = f.name
        try:
            res = predict_fn(tmp_path)
        finally:
            os.unlink(tmp_path)

        results["resize"].append({
            "scale": scale,
            "verdict": res.get("class", res) if isinstance(res, dict) else res,
            "confidence": res.get("confidence", None) if isinstance(res, dict) else None,
        })
        verdicts.append(res.get("class", res) if isinstance(res, dict) else res)

    # Stability: fraction of trials matching the majority vote
    if verdicts:
        majority = max(set(verdicts), key=verdicts.count)
        match_rate = verdicts.count(majority) / len(verdicts)
        results["stable"]          = match_rate >= 0.8
        results["stability_score"] = round(match_rate, 3)
        results["majority_verdict"] = majority
    else:
        results["majority_verdict"] = "Unknown"

    return results


def format_robustness_report(results: dict) -> str:
    lines = [
        "### 🛡️ Robustness Analysis (Bonus C)",
        "",
        f"**Majority Verdict**: {results.get('majority_verdict', 'N/A')}",
        f"**Stability Score**: {results.get('stability_score', 0)*100:.1f}% "
        f"({'✅ Stable' if results.get('stable') else '⚠️ Inconsistent'})",
        "",
        "#### JPEG Compression Sweep",
        "| Quality | Verdict | Confidence |",
        "|---------|---------|------------|",
    ]
    for r in results.get("jpeg", []):
        conf = f"{r['confidence']:.1f}%" if r["confidence"] else "N/A"
        lines.append(f"| {r['quality']:7d} | {r['verdict']:7s} | {conf:10s} |")

    lines += [
        "",
        "#### Resize Degradation Sweep",
        "| Scale | Verdict | Confidence |",
        "|-------|---------|------------|",
    ]
    for r in results.get("resize", []):
        conf = f"{r['confidence']:.1f}%" if r["confidence"] else "N/A"
        lines.append(f"| {r['scale']:.2f}  | {r['verdict']:7s} | {conf:10s} |")

    return "\n".join(lines)

