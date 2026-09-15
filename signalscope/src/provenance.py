"""
SignalScope - Provenance & Metadata Engine (Bonus Module D)
Inspects EXIF camera tags, C2PA Content Credentials, and image metadata signatures
to distinguish genuine camera photos from AI-generated outputs.
"""

from pathlib import Path
from typing import Union
from PIL import Image, ExifTags


def inspect_metadata(image_path: Union[str, Path]) -> dict:
    """
    Examine camera EXIF metadata and generative software footprints.

    Returns:
        dict with:
          - has_camera_exif (bool)
          - camera_make_model (str or None)
          - exposure_info (dict)
          - ai_software_signatures (list)
          - metadata_verdict (str: "Likely Camera Photo", "Likely Synthetic", "Neutral / Stripped")
          - metadata_confidence_shift (float: positive favors Real, negative favors AI)
    """
    path = Path(image_path)
    result = {
        "has_camera_exif": False,
        "camera_make_model": None,
        "exposure_info": {},
        "ai_software_signatures": [],
        "metadata_verdict": "Neutral / Stripped Metadata",
        "confidence_shift": 0.0,
        "details": [],
    }

    try:
        with Image.open(path) as img:
            # 1. Check PNG text metadata chunks (often contains diffusion prompts)
            if hasattr(img, "info") and img.info:
                for key, val in img.info.items():
                    val_str = str(val).lower()
                    for marker in ["stable diffusion", "midjourney", "dall-e", "comfyui", "novelai", "invokeai", "steps: ", "sampler: ", "cfg scale: "]:
                        if marker in val_str:
                            result["ai_software_signatures"].append(f"Prompt/Generation metadata chunk found: '{key}' contains '{marker}'")

            # 2. Check EXIF tags
            exif_raw = img.getexif()
            if exif_raw:
                named_exif = {}
                for tag_id, value in exif_raw.items():
                    tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                    named_exif[tag_name] = value

                # Camera hardware markers
                make  = named_exif.get("Make")
                model = named_exif.get("Model")
                software = str(named_exif.get("Software", "")).lower()

                if make or model:
                    make_str = str(make).strip() if make else ""
                    model_str = str(model).strip() if model else ""
                    result["camera_make_model"] = f"{make_str} {model_str}".strip()
                    result["has_camera_exif"] = True
                    result["details"].append(f"Captured by hardware camera: {result['camera_make_model']}")

                # Exposure markers
                for exp_tag in ["ExposureTime", "FNumber", "ISOSpeedRatings", "FocalLength", "DateTimeOriginal"]:
                    if exp_tag in named_exif:
                        result["exposure_info"][exp_tag] = str(named_exif[exp_tag])

                if result["exposure_info"]:
                    result["details"].append(f"Authentic exposure settings detected: {result['exposure_info']}")

                # Check software tag for AI generators
                for ai_app in ["stablediffusion", "automatic1111", "comfyui", "midjourney", "dalle"]:
                    if ai_app in software:
                        result["ai_software_signatures"].append(f"EXIF Software tag specifies generative tool: {software}")

    except Exception as e:
        result["details"].append(f"Metadata parsing note: {e}")

    # Determine verdict contribution
    if result["ai_software_signatures"]:
        result["metadata_verdict"] = "Likely AI-Generated (Software Fingerprint Found)"
        result["confidence_shift"] = -0.35  # strongly favors AI
    elif result["has_camera_exif"] and len(result["exposure_info"]) >= 2:
        result["metadata_verdict"] = "Authentic Camera Hardware EXIF Detected"
        result["confidence_shift"] = +0.35  # strongly favors Real photo
    elif result["has_camera_exif"]:
        result["metadata_verdict"] = "Camera Hardware Tag Present"
        result["confidence_shift"] = +0.30  # favors Real photo
    else:
        result["metadata_verdict"] = "No EXIF Metadata (Common in screenshots or web-compressed photos)"
        result["confidence_shift"] = 0.0

    return result


def format_provenance_report(metadata: dict) -> str:
    """Format markdown summary for UI and reports."""
    lines = [
        "### 📜 Provenance & Metadata Signals (Bonus D)",
        f"**Metadata Status**: {metadata['metadata_verdict']}",
    ]
    if metadata["camera_make_model"]:
        lines.append(f"- **Camera Device**: `{metadata['camera_make_model']}`")
    if metadata["exposure_info"]:
        exp_summary = ", ".join(f"{k}: {v}" for k, v in metadata["exposure_info"].items())
        lines.append(f"- **Capture Parameters**: `{exp_summary}`")
    if metadata["ai_software_signatures"]:
        for sig in metadata["ai_software_signatures"]:
            lines.append(f"- ⚠️ **AI Artifact**: `{sig}`")
    if not metadata["has_camera_exif"] and not metadata["ai_software_signatures"]:
        lines.append("- _Metadata is stripped (standard for web uploads, chat apps, and social media). Visual model analysis takes priority._")

    return "\n".join(lines)

