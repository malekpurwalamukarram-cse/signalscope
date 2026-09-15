"""
SignalScope - Generator Attribution Module (Bonus B)
Attributes AI-generated images to GAN vs. Diffusion model families.
"""

import sys
from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

try:
    from signalscope.model.architecture import compute_fft_spectrum
except ImportError:
    _HERE = Path(__file__).parent.parent
    sys.path.insert(0, str(_HERE / "model"))
    from architecture import compute_fft_spectrum

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def _extract_spectral_features(image_path: Union[str, Path]) -> np.ndarray:
    img    = Image.open(image_path).convert("RGB")
    tensor = _TRANSFORM(img).unsqueeze(0)
    freq   = compute_fft_spectrum(tensor)[0].numpy()  # [3, H, W]
    mean   = freq.mean(axis=0)

    # Extract radial profile (average energy at each frequency radius)
    h, w    = mean.shape
    cx, cy  = w // 2, h // 2
    max_r   = min(cx, cy)
    profile = np.zeros(max_r)
    for r in range(max_r):
        y_idx, x_idx = np.ogrid[:h, :w]
        mask   = (np.sqrt((x_idx - cx)**2 + (y_idx - cy)**2).astype(int) == r)
        if mask.sum() > 0:
            profile[r] = mean[mask].mean()

    return profile


def attribute_generator(image_path: Union[str, Path]) -> dict:
    """
    Heuristically attribute AI-generated image to generator family.

    Returns:
        dict with: generator_family, confidence, evidence
    """
    try:
        profile = _extract_spectral_features(image_path)
    except Exception as e:
        return {"generator_family": "Unknown", "confidence": 0.0, "evidence": str(e)}

    if len(profile) < 10:
        return {"generator_family": "Unknown", "confidence": 0.0, "evidence": "Too small"}

    # Feature: high-frequency energy slope (diffusion tends to be flatter)
    low_freq  = profile[:len(profile)//4].mean()
    high_freq = profile[3*len(profile)//4:].mean()
    hf_ratio  = float(high_freq / (low_freq + 1e-8))

    # Feature: periodicity (GAN grids produce periodic peaks in spectrum)
    fft_of_profile = np.abs(np.fft.fft(profile))
    periodicity    = float(fft_of_profile[2:len(fft_of_profile)//4].max() /
                           (fft_of_profile[1:].mean() + 1e-8))

    # Heuristic rules based on known spectral signatures
    evidence_lines = []

    if periodicity > 3.0:
        family     = "GAN (StyleGAN/ProGAN family)"
        confidence = min(0.5 + (periodicity - 3.0) * 0.05, 0.90)
        evidence_lines.append(f"Strong periodic spectral peaks (score={periodicity:.2f}) — characteristic of GAN upsampling grids.")
        evidence_lines.append("GAN architectures (StyleGAN, ProGAN) introduce aliasing artifacts from transposed convolutions.")
    elif hf_ratio < 0.15:
        family     = "Diffusion Model (Stable Diffusion/DALL-E family)"
        confidence = min(0.5 + (0.15 - hf_ratio) * 5, 0.88)
        evidence_lines.append(f"Low high-frequency energy ({hf_ratio:.3f}) — characteristic of diffusion denoising.")
        evidence_lines.append("Diffusion models suppress high-frequency noise through iterative denoising, leaving smooth spectra.")
    else:
        family     = "Uncertain (likely VAE-based or post-processed)"
        confidence = 0.45
        evidence_lines.append(f"Mixed spectral signature (HF ratio={hf_ratio:.3f}, periodicity={periodicity:.2f}).")
        evidence_lines.append("Could be VAE-GAN hybrid, heavily JPEG-compressed, or resized output.")

    return {
        "generator_family": family,
        "confidence":       round(confidence * 100, 1),
        "evidence":         "\n".join(evidence_lines),
        "features": {
            "hf_ratio":    round(hf_ratio, 4),
            "periodicity": round(periodicity, 4),
        },
    }


def format_attribution_report(image_path: Union[str, Path]) -> str:
    result = attribute_generator(image_path)
    return (
        f"### 🔎 Generator Attribution\n\n"
        f"**Attributed Family**: {result['generator_family']}\n"
        f"**Attribution Confidence**: {result['confidence']}%\n\n"
        f"**Evidence**:\n{result['evidence']}\n\n"
        f"**Spectral Features**:\n"
        f"- High-frequency ratio: `{result['features'].get('hf_ratio', 'N/A')}`\n"
        f"- Periodicity score: `{result['features'].get('periodicity', 'N/A')}`\n"
    )

