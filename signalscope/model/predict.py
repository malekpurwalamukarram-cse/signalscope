"""
SignalScope - Required CLI Entry Point (PS-2)
Usage: python predict.py --image <path_to_image>
"""

import argparse
import json
import sys
from pathlib import Path

# Fix Windows console encoding for Unicode/emojis
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from inference import predict, predict_with_details
from provenance import inspect_metadata


def main():
    parser = argparse.ArgumentParser(
        description="SignalScope — Real vs. AI-Generated Image Detection CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py --image photo.jpg
  python predict.py --image photo.jpg --details
  python predict.py --image photo.jpg --json
        """,
    )
    parser.add_argument("--image",   type=str, required=True, help="Path to the input image")
    parser.add_argument("--details", action="store_true", help="Show confidence scores")
    parser.add_argument("--json",    action="store_true",  help="Output as JSON")

    args = parser.parse_args()
    img_path = Path(args.image)
    if not img_path.exists():
        print(f"[ERROR] File not found: {img_path}", file=sys.stderr)
        sys.exit(1)

    if args.details or args.json:
        # Read EXIF/provenance signals first so they actually feed into the
        # verdict (previously this CLI never computed metadata at all, and
        # separately looked for a "metadata" key that predict_with_details()
        # never returns — so the camera/provenance lines below never printed).
        try:
            metadata = inspect_metadata(img_path)
        except Exception:
            metadata = None

        result = predict_with_details(img_path, metadata=metadata)
        output = {
            "image":      str(img_path),
            "verdict":    result["class"],
            "confidence": result["confidence"],
            "real_prob":  result["real_prob"],
            "ai_prob":    result["ai_prob"],
            "threshold_calibrated": result.get("threshold_calibrated", False),
        }
        if metadata is not None:
            output["metadata_verdict"] = metadata.get("metadata_verdict")
            output["camera_make_model"] = metadata.get("camera_make_model")

        if args.json:
            print(json.dumps(output, indent=2))
        else:
            verdict_icon = "✅ REAL" if result["class"] == "Real" else "🤖 AI-GENERATED"
            print(f"\n=== SignalScope — Image Authentication ===")
            print(f"  Image      : {img_path.name}")
            print(f"  Verdict    : {verdict_icon}")
            print(f"  Confidence : {result['confidence']:.1f}%")
            print(f"  Real prob  : {result['real_prob']:.1f}%")
            print(f"  AI prob    : {result['ai_prob']:.1f}%")
            if not result.get("threshold_calibrated", False):
                print(f"  Note       : decision threshold is the uncalibrated default (checkpoint has no stored threshold)")
            if metadata and metadata.get("camera_make_model"):
                print(f"  Camera     : {metadata['camera_make_model']} (Authentic EXIF verified)")
            if metadata and metadata.get("metadata_verdict"):
                print(f"  Provenance : {metadata['metadata_verdict']}")
    else:
        label = predict(img_path)
        print(label)


if __name__ == "__main__":
    main()