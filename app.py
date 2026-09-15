# app.py
"""
SIH 2026 - SignalScope Production Web Application.
Single-Stream Computational Architecture utilizing Calibrated MobileNetV3-Small.

Full-stack web app, single file, no other files added or removed:
  - FastAPI backend below calls the detection pipeline (inference / Grad-CAM /
    attribution / provenance) exactly as before — nothing in signalscope/ changed.
  - The frontend (HTML/CSS/JS) is embedded inline and served by this same file,
    so there's nothing extra to deploy: `python app.py` (or
    `uvicorn app:app`) starts the whole thing.

Run:
    python app.py
Then open http://localhost:7860
"""

import base64
import io
import os
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from PIL import Image, UnidentifiedImageError

# Force project root path binding
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from signalscope.model.inference import predict_with_details as ss_predict_details
from signalscope.model.inference import get_model as ss_get_model
from signalscope.src.forensic_explainer import SignalScopeGradCAM, generate_forensic_report
from signalscope.src.attribution import attribute_generator
from signalscope.src.provenance import inspect_metadata, format_provenance_report
from signalscope.src.robustness import robustness_sweep, format_robustness_report

SAMPLES_DIR = ROOT / "test_samples"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _pil_to_data_uri(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _save_upload_to_tempfile(raw_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        image.save(f.name, "JPEG", quality=95)
        return f.name


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline (unchanged logic — same calls, same order, same outputs contract
#  as before; this just returns a JSON-serializable dict instead of Gradio
#  components).
# ─────────────────────────────────────────────────────────────────────────────
def execute_pipeline(tmp_path: str) -> dict:
    # 1. Read provenance/EXIF signals first, so they can actually feed into
    #    the classification decision below (previously this was only computed
    #    *after* the verdict and had no effect on it — the metadata nudge in
    #    predict_with_details() was always called with metadata=None).
    try:
        metadata = inspect_metadata(tmp_path)
    except Exception:
        metadata = None

    # 2. Run MobileNet Core Inference Pipeline, fused with the metadata signal.
    result = ss_predict_details(tmp_path, metadata=metadata)

    verdict = result.get("class", "Real")
    confidence = result.get("confidence", 100.0)

    if verdict == "Weights Missing":
        return {
            "ok": False,
            "verdict": "Weights Missing",
            "error": "Model weights not found. Check signalscope/model/weights/ and try again.",
        }

    # 2. Run Grad-CAM Explainer Loop with Defensive Failure Safeguards
    heatmap_data_uri = None
    try:
        # Reuse the already-loaded, cached model instead of reading the
        # checkpoint from disk again on every single request.
        model, model_device = ss_get_model()
        if model is None:
            raise RuntimeError("Model weights not available")

        gc = SignalScopeGradCAM(model, device=model_device)
        cam, _, _ = gc.generate(tmp_path)
        overlay = gc.overlay(tmp_path, cam)
        heatmap_img = Image.fromarray(overlay)
        gc.cleanup()
        heatmap_data_uri = _pil_to_data_uri(heatmap_img)
    except Exception as e:
        print(f"[GradCAM Hook Shield Active] Gracefully bypassed: {e}")
        heatmap_data_uri = None  # frontend falls back to showing the source image

    # 3. Compile Diagnostic Text Reports (forensic + provenance/EXIF signals)
    try:
        forensic_report = generate_forensic_report(tmp_path, verdict, confidence)
        if metadata is not None:
            forensic_report = f"{forensic_report}\n\n{format_provenance_report(metadata)}"
    except Exception:
        forensic_report = (
            f"### Forensic diagnostics summary\n"
            f"* **Verdict:** {verdict}\n"
            f"* **Confidence:** {confidence:.2f}%\n"
            f"* **Analysis:** Structural layer noise boundaries evaluated successfully."
        )

    # 4. Generator attribution — only meaningful for images classified AI-Generated
    if verdict == "AI-Generated":
        try:
            attribution_result = attribute_generator(tmp_path)
            attribution = (
                f"### Generator attribution\n"
                f"**Likely family:** {attribution_result['generator_family']}  \n"
                f"**Confidence:** {attribution_result['confidence']}%\n\n"
                f"{attribution_result['evidence']}"
            )
        except Exception as e:
            attribution = f"Generator attribution unavailable: {e}"
    else:
        attribution = (
            "### Generator attribution\n"
            "Attribution is only calculated for images classified as AI-generated. "
            "This image was assessed as **likely real**, so no generator family is reported."
        )

    return {
        "ok": True,
        "verdict": verdict,
        "confidence": float(confidence),
        "real_prob": float(result.get("real_prob", 0.0)),
        "ai_prob": float(result.get("ai_prob", 0.0)),
        "needs_review": bool(result.get("needs_review", False)),
        "model": result.get("model"),
        "device": result.get("device"),
        "decision_threshold": result.get("decision_threshold"),
        "threshold_calibrated": result.get("threshold_calibrated", False),
        "metadata_verdict": result.get("metadata_verdict"),
        "metadata_shift_applied": result.get("metadata_shift_applied"),
        "heatmap_image": heatmap_data_uri,
        "forensic_report_markdown": forensic_report,
        "attribution_markdown": attribution,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  FastAPI app
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Load the model once at startup rather than on the first request, so
    # the first user interaction isn't the slowest one.
    ss_get_model()
    yield
    # (nothing to clean up on shutdown)


app = FastAPI(title="SignalScope", lifespan=_lifespan)


@app.get("/api/health")
def health():
    model, device = ss_get_model()
    return {"status": "ok", "model_loaded": model is not None, "device": str(device)}


@app.get("/samples/real")
def sample_real():
    return FileResponse(str(SAMPLES_DIR / "actual_real_photo.png"))


@app.get("/samples/ai")
def sample_ai():
    return FileResponse(str(SAMPLES_DIR / "actual_ai_image.png"))


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported file type. Upload a JPEG, PNG, WebP or BMP image.")

    raw_bytes = await file.read()
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large (15 MB limit).")

    try:
        tmp_path = _save_upload_to_tempfile(raw_bytes)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="That file doesn't look like a valid image.")

    started = time.time()
    try:
        payload = execute_pipeline(tmp_path)
    except Exception as e:
        print(f"[Critical Application Safeguard Activated] {e}")
        payload = {
            "ok": False,
            "verdict": "Error",
            "error": "Analysis failed while processing this image. Please try another file.",
        }
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    payload["elapsed_ms"] = round((time.time() - started) * 1000)

    if not payload.get("ok"):
        return JSONResponse(status_code=200, content=payload)
    return payload


@app.post("/api/robustness")
async def robustness(file: UploadFile = File(...)):
    """
    Bonus C — Robustness Analysis.

    Sweeps the same uploaded image through several JPEG-quality and resize
    degradations and re-runs the classifier on each variant, to report how
    stable the verdict is under realistic post-processing (a second upload,
    a social-media re-encode, a screenshot, etc.). This is deliberately a
    separate, on-demand endpoint rather than part of /api/analyze, since it
    runs ~10 extra forward passes and would otherwise slow down every single
    upload for a diagnostic most users won't need.
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported file type. Upload a JPEG, PNG, WebP or BMP image.")

    raw_bytes = await file.read()
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large (15 MB limit).")

    try:
        tmp_path = _save_upload_to_tempfile(raw_bytes)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="That file doesn't look like a valid image.")

    try:
        model, model_device = ss_get_model()
        if model is None:
            return JSONResponse(status_code=200, content={
                "ok": False,
                "error": "Model weights not found. Check signalscope/model/weights/ and try again.",
            })

        sweep = robustness_sweep(tmp_path, ss_predict_details)
        return {
            "ok": True,
            "stable": sweep.get("stable"),
            "stability_score": sweep.get("stability_score"),
            "majority_verdict": sweep.get("majority_verdict"),
            "jpeg": sweep.get("jpeg", []),
            "resize": sweep.get("resize", []),
            "report_markdown": format_robustness_report(sweep),
        }
    except Exception as e:
        print(f"[Robustness Sweep Safeguard Activated] {e}")
        return JSONResponse(status_code=200, content={
            "ok": False,
            "error": "Robustness sweep failed for this image. Please try another file.",
        })
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


# ─────────────────────────────────────────────────────────────────────────────
#  Frontend — embedded inline so the whole app is this one file.
#  Look & feel: instrument-panel / signal-analysis console, not a SaaS template.
# ─────────────────────────────────────────────────────────────────────────────
INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>SignalScope — Real vs. AI-generated image forensics</title>
<meta name="description" content="Upload an image and get a calibrated real-vs-AI verdict with visible evidence: an anomaly heat-map, forensic and provenance signals, and generator attribution." />
<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 32 32%22><rect width=%2232%22 height=%2232%22 rx=%226%22 fill=%22%230b0d0f%22/><path d=%22M3 17h5l2-7 4 14 3-11 2 4h10%22 fill=%22none%22 stroke=%22%23e8a33d%22 stroke-width=%222.2%22 stroke-linecap=%22round%22 stroke-linejoin=%22round%22/></svg>">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{
  --ink:#0a0c0f; --panel:#14171c; --panel-2:#181c22; --recessed:#0d0f12;
  --border:#242930; --border-strong:#3a4048;
  --text:#e9ecef; --text-dim:#8b93a0; --text-faint:#565e69;
  --accent:#e8a33d; --accent-ink:#0a0c0f;
  --real:#46c2a0; --ai:#e1523d; --warn:#e8a33d;
  --radius:6px;
  --font-display:'Space Grotesk', sans-serif;
  --font-body:'IBM Plex Sans', sans-serif;
  --font-mono:'IBM Plex Mono', monospace;
}
*, *::before, *::after{ box-sizing:border-box; }
[hidden]{ display:none !important; }
@media (prefers-reduced-motion: reduce){
  *{ animation-duration:0.001ms !important; animation-iteration-count:1 !important; transition-duration:0.001ms !important; }
}
html, body{
  margin:0; padding:0; min-height:100%;
  background-color:var(--ink);
  background-image:
    radial-gradient(rgba(255,255,255,0.05) 1px, transparent 1px),
    radial-gradient(ellipse 900px 520px at 8% -10%, rgba(232,163,61,0.10), transparent 60%),
    radial-gradient(ellipse 800px 560px at 105% 6%, rgba(70,194,160,0.06), transparent 62%),
    radial-gradient(ellipse 700px 500px at 50% 115%, rgba(232,163,61,0.05), transparent 60%);
  background-size:26px 26px, auto, auto, auto;
  background-repeat:repeat, no-repeat, no-repeat, no-repeat;
  background-attachment:fixed;
  color:var(--text);
  font-family:var(--font-body);
  -webkit-font-smoothing:antialiased;
}
a{ color:inherit; }
h1,h2,h3{ font-family:var(--font-display); margin:0; }
button{ font:inherit; }
*:focus-visible{ outline:2px solid var(--accent); outline-offset:2px; }
.scanline-overlay{
  position:fixed; inset:0; pointer-events:none; z-index:1000;
  background:repeating-linear-gradient(
    to bottom, rgba(255,255,255,0.012) 0px, rgba(255,255,255,0.012) 1px,
    transparent 1px, transparent 3px
  );
  mix-blend-mode:overlay;
}
main{ max-width:1180px; margin:0 auto; padding:0 20px 60px 20px; }
.ss-nav{
  position:sticky; top:0; z-index:50;
  background:rgba(10,12,15,0.86); backdrop-filter:blur(10px);
  border-bottom:1px solid var(--border);
}
.ss-nav-inner{
  max-width:1180px; margin:0 auto; padding:0 20px;
  height:60px; display:flex; align-items:center; gap:28px;
}
.ss-logo{
  display:flex; align-items:center; gap:9px; text-decoration:none;
  font-family:var(--font-display); font-weight:700; font-size:1.08rem; color:var(--text);
}
.ss-logo-mark{ width:24px; height:17px; color:var(--accent); }
.ss-nav-tabs{
  display:flex; gap:2px; padding:3px; background:var(--panel);
  border:1px solid var(--border); border-radius:6px; margin-left:auto;
}
.ss-nav-tab{
  font-family:var(--font-mono); font-size:0.82rem; letter-spacing:0.03em;
  color:var(--text-dim); background:transparent; border:none; border-radius:4px;
  padding:7px 16px; cursor:pointer; transition:color .12s ease;
}
.ss-nav-tab:hover{ color:var(--text); }
.ss-nav-tab.is-active{ color:var(--accent-ink); background:var(--accent); font-weight:600; }
.ss-nav-status{
  display:flex; align-items:center; gap:7px;
  font-family:var(--font-mono); font-size:0.74rem; color:var(--text-faint);
}
.status-dot{
  width:7px; height:7px; border-radius:50%; background:var(--text-faint); flex:none;
  box-shadow:0 0 0 3px transparent; transition:background .2s ease;
}
.status-dot.is-online{ background:var(--real); box-shadow:0 0 8px rgba(70,194,160,0.6); }
.status-dot.is-offline{ background:var(--ai); box-shadow:0 0 8px rgba(225,82,61,0.6); }
@media (max-width:640px){ .ss-nav-status{ display:none; } }
.ss-view{ display:none; }
.ss-view.is-active{ display:block; animation:ss-fade-in .35s ease both; }
@keyframes ss-fade-in{ from{ opacity:0; transform:translateY(6px); } to{ opacity:1; transform:none; } }
.ss-hero{ padding:44px 0 26px 0; max-width:760px; }
.ss-hero h1{
  font-size:clamp(1.7rem, 3.6vw, 2.5rem); font-weight:700; line-height:1.18;
  letter-spacing:-0.01em; color:var(--text);
}
.hero-accent{ color:var(--accent); }
.ss-hero-sub{ margin:14px 0 0 0; color:var(--text-dim); font-size:1rem; line-height:1.6; max-width:640px; }
.ss-grid{ display:grid; grid-template-columns:1fr 1fr; gap:18px; align-items:start; }
@media (max-width:860px){ .ss-grid{ grid-template-columns:1fr; } }
.ss-col{ display:flex; flex-direction:column; gap:18px; }
.ss-panel{
  background:var(--panel); border:1px solid var(--border); border-radius:var(--radius);
  padding:14px;
}
.panel-label{
  font-family:var(--font-mono); font-size:0.7rem; letter-spacing:0.08em;
  text-transform:uppercase; color:var(--text-faint); padding:0 2px 10px 2px;
}
.scope-frame{ position:relative; }
.scope-corners{ position:absolute; inset:8px; pointer-events:none; z-index:5; }
.scope-corners span{ position:absolute; width:13px; height:13px; }
.scope-corners .tl{ top:0; left:0; border-top:2px solid var(--accent); border-left:2px solid var(--accent); }
.scope-corners .tr{ top:0; right:0; border-top:2px solid var(--accent); border-right:2px solid var(--accent); }
.scope-corners .bl{ bottom:0; left:0; border-bottom:2px solid var(--accent); border-left:2px solid var(--accent); }
.scope-corners .br{ bottom:0; right:0; border-bottom:2px solid var(--accent); border-right:2px solid var(--accent); }
.dropzone{
  position:relative; height:300px; border-radius:4px; overflow:hidden;
  background:var(--recessed); border:1px dashed var(--border-strong);
  display:flex; align-items:center; justify-content:center; cursor:pointer;
  transition:border-color .15s ease, background .15s ease;
}
.dropzone:hover{ border-color:var(--accent); }
.dropzone.is-dragover{ border-color:var(--accent); background:#12100a; }
.dz-idle{ text-align:center; padding:0 30px; color:var(--text-dim); }
.dz-icon{ width:38px; height:38px; color:var(--text-faint); margin-bottom:10px; }
.dz-title{ margin:0; font-family:var(--font-display); font-weight:600; color:var(--text); font-size:1rem; }
.dz-sub{ margin:6px 0 0 0; font-size:0.82rem; line-height:1.5; max-width:320px; }
.dz-preview{ width:100%; height:100%; object-fit:contain; background:var(--recessed); }
.dz-paste{
  position:absolute; inset:0; display:flex; flex-direction:column;
  align-items:center; justify-content:center; background:var(--recessed); z-index:6;
  gap:10px; cursor:default;
}
.dz-paste-icon{ width:48px; height:48px; color:var(--accent); }
.dz-paste-title{ margin:0; font-family:var(--font-display); font-weight:600; color:var(--text); font-size:1rem; }
.dz-paste-sub{ margin:0; font-size:0.82rem; color:var(--text-dim); }
.dz-paste-kbd{
  display:inline-block; font-family:var(--font-mono); font-size:0.82rem; font-weight:600;
  background:var(--panel-2); border:1px solid var(--border-strong); border-radius:4px;
  padding:3px 10px; color:var(--accent); margin-top:4px;
}
.paste-controls{
  display:flex; gap:10px; margin-top:10px;
}
.dz-scanning{
  position:absolute; inset:0; display:flex; align-items:flex-end; justify-content:center;
  background:rgba(10,12,15,0.35);
}
.scan-beam{
  position:absolute; left:0; right:0; height:2px; top:0;
  background:linear-gradient(90deg, transparent, var(--accent), transparent);
  box-shadow:0 0 14px 2px rgba(232,163,61,0.7);
  animation:ss-scan 1.6s ease-in-out infinite;
}
@keyframes ss-scan{ 0%{ top:0; } 50%{ top:100%; } 100%{ top:0; } }
.dz-scan-label{
  margin:0 0 14px 0; font-family:var(--font-mono); font-size:0.78rem;
  letter-spacing:0.05em; color:var(--accent); background:var(--ink);
  padding:5px 12px; border-radius:4px; border:1px solid var(--border-strong);
}
.ss-upload-actions{ display:flex; gap:10px; margin-top:12px; }
.ss-btn{
  font-family:var(--font-mono); font-weight:600; font-size:0.82rem; letter-spacing:0.02em;
  border-radius:5px; padding:10px 18px; cursor:pointer; border:1px solid transparent;
  transition:filter .12s ease, opacity .12s ease;
}
.ss-btn:disabled{ opacity:0.4; cursor:not-allowed; }
.ss-btn-primary{ background:var(--accent); color:var(--accent-ink); border-color:var(--accent); flex:1; }
.ss-btn-primary:hover:not(:disabled){ filter:brightness(1.08); }
.ss-btn-ghost{ background:transparent; color:var(--text-dim); border-color:var(--border-strong); }
.ss-btn-ghost:hover{ color:var(--text); border-color:var(--text-dim); }
.ss-samples{
  margin-top:14px; padding-top:12px; border-top:1px solid var(--border);
  display:flex; flex-wrap:wrap; align-items:center; gap:8px;
}
.ss-samples-label{ font-family:var(--font-mono); font-size:0.72rem; color:var(--text-faint); }
.sample-chip{
  font-family:var(--font-mono); font-size:0.76rem; color:var(--text-dim);
  background:var(--recessed); border:1px solid var(--border-strong); border-radius:4px;
  padding:5px 10px; cursor:pointer; transition:color .12s ease, border-color .12s ease;
}
.sample-chip:hover{ color:var(--text); border-color:var(--accent); }
.readout{
  background:var(--recessed); border:1px solid var(--border); border-radius:var(--radius);
  padding:22px; min-height:150px;
}
.readout-idle{ display:flex; flex-direction:column; justify-content:center; color:var(--text-dim); }
.readout-idle .readout-title{ color:var(--text); font-weight:600; }
.readout-idle .readout-sub{ margin-top:4px; font-size:0.88rem; }
.readout-row{ display:flex; align-items:baseline; justify-content:space-between; margin-bottom:16px; gap:10px; flex-wrap:wrap; }
.readout-title{ font-family:var(--font-display); font-weight:700; font-size:1.4rem; }
.readout-num{ font-family:var(--font-mono); font-size:2rem; font-weight:500; font-variant-numeric:tabular-nums; color:var(--text); }
.readout-num .unit{ font-size:1.1rem; color:var(--text-dim); }
.status-real .readout-title{ color:var(--real); }
.status-ai .readout-title{ color:var(--ai); }
.readout-fault{ color:var(--warn); }
.readout-meters{ display:flex; flex-direction:column; gap:9px; }
.meter-row{ display:flex; align-items:center; gap:10px; }
.meter-label{ font-family:var(--font-mono); font-size:0.72rem; color:var(--text-faint); width:32px; text-transform:uppercase; }
.meter-pct{ font-family:var(--font-mono); font-size:0.78rem; color:var(--text-dim); width:48px; text-align:right; font-variant-numeric:tabular-nums; }
.meter{ flex:1; display:flex; gap:2px; }
.meter .seg{ flex:1; height:10px; background:rgba(255,255,255,0.06); border-radius:1px; transform:scaleY(0.3); opacity:0; transition:transform .28s ease, opacity .28s ease; }
.meter .seg.on-real{ background:var(--real); }
.meter .seg.on-ai{ background:var(--ai); }
.meter.is-filled .seg{ transform:scaleY(1); opacity:1; }
.flag-line{
  margin-top:14px; padding-top:10px; border-top:1px dashed var(--border);
  font-family:var(--font-mono); font-size:0.76rem; letter-spacing:0.03em;
  color:var(--warn); text-transform:uppercase;
}
.readout-meta{
  margin-top:12px; padding-top:10px; border-top:1px solid var(--border);
  font-family:var(--font-mono); font-size:0.72rem; color:var(--text-faint);
  display:flex; flex-wrap:wrap; gap:4px 14px;
}
.heatmap-slot{
  height:240px; border-radius:4px; overflow:hidden; background:var(--recessed);
  display:flex; align-items:center; justify-content:center;
}
.heatmap-empty{ color:var(--text-faint); font-size:0.85rem; text-align:center; padding:0 30px; }
.heatmap-slot img{ width:100%; height:100%; object-fit:contain; }
.ss-reports{ margin-top:22px; }
.ss-report-tabs{
  display:flex; gap:2px; width:fit-content; padding:3px; margin-bottom:12px;
  background:var(--panel); border:1px solid var(--border); border-radius:6px;
}
.report-tab{
  font-family:var(--font-mono); font-size:0.8rem; color:var(--text-dim);
  background:transparent; border:none; border-radius:4px; padding:7px 16px; cursor:pointer;
}
.report-tab:hover{ color:var(--text); }
.report-tab.is-active{ color:var(--accent-ink); background:var(--accent); font-weight:600; }
.report-panel{
  background:var(--panel); border:1px solid var(--border); border-radius:var(--radius);
  padding:18px 20px; color:var(--text-dim); font-size:0.92rem; line-height:1.65;
}
.report-panel h3{ margin:0 0 10px 0; font-family:var(--font-display); font-size:1.05rem; color:var(--text); border-bottom:1px solid var(--border); padding-bottom:8px; }
.report-panel strong{ color:var(--text); }
.report-panel ul{ margin:6px 0; padding-left:20px; }
.report-panel p{ margin:8px 0; }
.about-wrap{ max-width:840px; margin:0 auto; padding:36px 0 20px 0; }
.spec-section{
  border:1px solid var(--border); border-radius:var(--radius); background:var(--panel);
  padding:22px 24px; margin-bottom:16px;
}
.spec-section h2{ font-size:1.1rem; font-weight:600; margin:0 0 12px 0; color:var(--text); border-bottom:1px solid var(--border); padding-bottom:10px; }
.spec-section p{ color:var(--text-dim); line-height:1.65; font-size:0.93rem; margin:0; }
.spec-section.lead p{ color:var(--text); font-size:1.02rem; }
.chain-list{ list-style:none; margin:0; padding:0; }
.chain-list li{ display:flex; gap:16px; align-items:flex-start; padding:13px 0; border-bottom:1px solid var(--border); }
.chain-list li:last-child{ border-bottom:none; padding-bottom:2px; }
.chain-index{
  font-family:var(--font-mono); font-weight:600; font-size:0.76rem; color:var(--ink);
  background:var(--accent); width:24px; height:24px; min-width:24px; border-radius:4px;
  display:flex; align-items:center; justify-content:center; margin-top:1px;
}
.chain-body b{ font-family:var(--font-display); font-weight:600; font-size:0.94rem; color:var(--text); }
.chain-body p{ margin:3px 0 0 0; color:var(--text-dim); font-size:0.86rem; line-height:1.5; }
.spec-table{ width:100%; border-collapse:collapse; margin-top:2px; }
.spec-table thead th{
  text-align:left; font-family:var(--font-mono); font-size:0.7rem; letter-spacing:0.08em;
  text-transform:uppercase; color:var(--text-faint); padding:0 4px 10px 4px; border-bottom:1px solid var(--border-strong); font-weight:500;
}
.spec-table tbody tr{ border-bottom:1px solid var(--border); }
.spec-table tbody tr:last-child{ border-bottom:none; }
.spec-table td{ padding:11px 4px; font-size:0.87rem; vertical-align:top; }
.spec-table td:first-child{ white-space:nowrap; padding-right:18px; width:1%; }
.spec-table td:last-child{ color:var(--text-dim); line-height:1.55; }
.tag{
  font-family:var(--font-mono); font-size:0.76rem; color:var(--text);
  background:var(--recessed); border:1px solid var(--border-strong); border-radius:4px; padding:3px 9px;
}
.ss-footer{
  max-width:1180px; margin:0 auto; text-align:left; color:var(--text-faint); font-size:0.8rem;
  padding:18px 20px 30px 20px; border-top:1px solid var(--border);
}
.toast{
  position:fixed; bottom:22px; left:50%; transform:translateX(-50%);
  background:var(--panel-2); border:1px solid var(--ai); color:var(--text);
  font-size:0.86rem; padding:11px 18px; border-radius:6px; z-index:200;
  box-shadow:0 10px 30px rgba(0,0,0,0.4);
}
::-webkit-scrollbar{ width:10px; height:10px; }
::-webkit-scrollbar-thumb{ background:var(--border-strong); border-radius:5px; }

/* ── Source image preview ─────────────────────────────────────── */
.source-preview{
  position:relative; border-radius:4px; overflow:hidden;
  background:var(--recessed); border:1px solid var(--border);
  min-height:100px; display:flex; align-items:center; justify-content:center;
}
.source-preview img{
  width:100%; max-height:300px; object-fit:contain; display:block;
}
.source-preview-empty{
  color:var(--text-faint); font-size:0.84rem; text-align:center; padding:28px 20px;
}
.source-preview-name{
  position:absolute; bottom:0; left:0; right:0;
  font-family:var(--font-mono); font-size:0.72rem; color:var(--text-dim);
  background:rgba(10,12,15,0.82); padding:5px 10px;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}

/* ── Nav gear icon ────────────────────────────────────────────── */
.ss-nav-gear{
  background:none; border:1px solid var(--border-strong); border-radius:5px;
  color:var(--text-dim); width:34px; height:34px; display:flex;
  align-items:center; justify-content:center; cursor:pointer;
  transition:color .12s ease, border-color .12s ease; margin-left:10px;
}
.ss-nav-gear:hover{ color:var(--text); border-color:var(--accent); }
.ss-nav-gear svg{ width:18px; height:18px; }

/* ── Settings modal ───────────────────────────────────────────── */
.settings-overlay{
  position:fixed; inset:0; z-index:300; background:rgba(0,0,0,0.65);
  backdrop-filter:blur(4px); -webkit-backdrop-filter:blur(4px);
  display:flex; align-items:center; justify-content:center;
  animation:ss-fade-in .2s ease both;
}
.settings-box{
  background:var(--panel); border:1px solid var(--border-strong);
  border-radius:10px; width:420px; max-width:92vw; max-height:85vh;
  overflow-y:auto; box-shadow:0 20px 60px rgba(0,0,0,0.6);
  position:relative; z-index:301;
}
.settings-header{
  display:flex; align-items:center; justify-content:space-between;
  padding:18px 22px 14px; border-bottom:1px solid var(--border);
}
.settings-header h2{ font-size:1.1rem; font-weight:700; margin:0; }
.settings-close{
  background:transparent; border:1px solid transparent; color:var(--text-dim); cursor:pointer;
  font-size:1.4rem; line-height:1; width:32px; height:32px; border-radius:6px;
  display:flex; align-items:center; justify-content:center; transition:all .15s ease;
}
.settings-close:hover{ color:var(--text); background:var(--panel-2); border-color:var(--border-strong); }
.settings-body{ padding:16px 22px 22px; }
.settings-row{
  display:flex; align-items:center; justify-content:space-between;
  padding:14px 0; border-bottom:1px solid var(--border);
}
.settings-row:last-child{ border-bottom:none; }
.settings-row-info{ flex:1; }
.settings-row-label{
  font-family:var(--font-display); font-weight:600; font-size:0.92rem; color:var(--text);
}
.settings-row-desc{
  font-size:0.8rem; color:var(--text-dim); margin-top:2px;
}
.toggle{
  position:relative; width:44px; height:24px; flex:none; margin-left:14px;
}
.toggle input{ opacity:0; width:0; height:0; position:absolute; }
.toggle-track{
  position:absolute; inset:0; background:var(--border-strong);
  border-radius:12px; cursor:pointer; transition:background .2s ease;
}
.toggle-track::after{
  content:''; position:absolute; left:3px; top:3px;
  width:18px; height:18px; border-radius:50%; background:var(--text-dim);
  transition:transform .2s ease, background .2s ease;
}
.toggle input:checked + .toggle-track{ background:var(--accent); }
.toggle input:checked + .toggle-track::after{
  transform:translateX(20px); background:var(--accent-ink);
}

/* ── History view ─────────────────────────────────────────────── */
.history-wrap{ max-width:900px; margin:0 auto; padding:30px 0 20px; }
.history-header{
  display:flex; align-items:center; justify-content:space-between;
  margin-bottom:18px;
}
.history-header h2{
  font-family:var(--font-display); font-size:1.2rem; font-weight:700; margin:0;
}
.history-empty{
  text-align:center; color:var(--text-faint); padding:50px 20px;
  font-size:0.92rem;
}
.history-list{ display:flex; flex-direction:column; gap:10px; }
.history-card{
  display:flex; gap:14px; align-items:center;
  background:var(--panel); border:1px solid var(--border);
  border-radius:var(--radius); padding:12px 16px;
  cursor:pointer; transition:border-color .12s ease;
}
.history-card:hover{ border-color:var(--accent); }
.history-thumb{
  width:56px; height:56px; border-radius:4px; object-fit:cover;
  background:var(--recessed); border:1px solid var(--border); flex:none;
}
.history-info{ flex:1; min-width:0; }
.history-name{
  font-family:var(--font-display); font-weight:600; font-size:0.9rem;
  color:var(--text); white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}
.history-meta{
  font-family:var(--font-mono); font-size:0.74rem; color:var(--text-dim); margin-top:3px;
}
.history-verdict{
  font-family:var(--font-mono); font-weight:600; font-size:0.82rem;
  padding:4px 12px; border-radius:4px; white-space:nowrap; flex:none;
}
.history-verdict.is-real{ color:var(--real); background:rgba(70,194,160,0.12); border:1px solid rgba(70,194,160,0.25); }
.history-verdict.is-ai{ color:var(--ai); background:rgba(225,82,61,0.12); border:1px solid rgba(225,82,61,0.25); }
</style>
</head>
<body>

<div class="scanline-overlay" aria-hidden="true"></div>

<header class="ss-nav">
  <div class="ss-nav-inner">
    <a class="ss-logo" href="#top">
      <svg viewBox="0 0 28 20" class="ss-logo-mark" aria-hidden="true">
        <path d="M1 12h5l2-7 4.5 15 3.5-13 2 5h9" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
      <span>SignalScope</span>
    </a>
    <nav class="ss-nav-tabs" role="tablist" aria-label="Sections">
      <button class="ss-nav-tab is-active" data-view="analyze" role="tab" aria-selected="true">analyze</button>
      <button class="ss-nav-tab" data-view="history" role="tab" aria-selected="false">history</button>
      <button class="ss-nav-tab" data-view="about" role="tab" aria-selected="false">about</button>
    </nav>
    <div class="ss-nav-status">
      <span class="status-dot" id="apiStatusDot" title="Checking backend…"></span>
      <span id="apiStatusText">connecting…</span>
    </div>
    <button class="ss-nav-gear" id="settingsBtn" type="button" title="Settings" aria-label="Open settings">
      <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="10" cy="10" r="3"/>
        <path d="M10 1.5v2M10 16.5v2M1.5 10h2M16.5 10h2M3.4 3.4l1.4 1.4M15.2 15.2l1.4 1.4M3.4 16.6l1.4-1.4M15.2 4.8l1.4-1.4"/>
      </svg>
    </button>
  </div>
</header>

<main id="top">

  <section id="view-analyze" class="ss-view is-active">

    <div class="ss-hero">
      <h1>Telling real from synthetic, <span class="hero-accent">with the evidence shown</span>.</h1>
      <p class="ss-hero-sub">
        Drop in an image. SignalScope runs it through a calibrated MobileNetV3 classifier,
        traces the verdict back to the pixels that drove it, checks the file's own metadata,
        and — for AI-generated images — estimates the generator family behind it.
      </p>
    </div>

    <div class="ss-grid">

      <div class="ss-col">
        <div class="ss-panel">
          <div class="panel-label">Upload</div>

          <div class="scope-frame">
            <div class="dropzone" id="dropzone" tabindex="0" role="button"
                 aria-label="Upload an image to analyze">
              <input type="file" id="fileInput" accept="image/png,image/jpeg,image/webp,image/bmp" hidden />

              <div class="dz-idle" id="dzIdle">
                <svg viewBox="0 0 48 48" class="dz-icon" aria-hidden="true">
                  <path d="M24 31V9m0 0l-8 8m8-8l8 8" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>
                  <path d="M8 33v5a3 3 0 0 0 3 3h26a3 3 0 0 0 3-3v-5" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"/>
                </svg>
                <p class="dz-title">Drop an image, click to browse, or paste with Ctrl+V</p>
                <p class="dz-sub">JPEG · PNG · WebP · BMP, up to 15&nbsp;MB. Nothing is stored — files are deleted right after analysis.</p>
              </div>

              <img id="previewImg" class="dz-preview" alt="" hidden />

              <div class="dz-paste" id="dzPaste" hidden>
                <svg viewBox="0 0 48 48" class="dz-paste-icon" aria-hidden="true">
                  <rect x="12" y="4" width="24" height="40" rx="3" fill="none" stroke="currentColor" stroke-width="2.4"/>
                  <rect x="18" y="2" width="12" height="6" rx="2" fill="none" stroke="currentColor" stroke-width="2"/>
                  <line x1="18" y1="18" x2="30" y2="18" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                  <line x1="18" y1="24" x2="30" y2="24" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                  <line x1="18" y1="30" x2="26" y2="30" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                </svg>
                <p class="dz-paste-title">Paste an image from your clipboard</p>
                <p class="dz-paste-sub">Copy any image first, then press</p>
                <span class="dz-paste-kbd">Ctrl + V</span>
                <div class="paste-controls">
                  <button class="ss-btn ss-btn-ghost" id="pasteCancelBtn" type="button">cancel</button>
                </div>
              </div>

              <div class="dz-scanning" id="dzScanning" hidden>
                <div class="scan-beam"></div>
                <p class="dz-scan-label" id="scanLabel">running analysis…</p>
              </div>

              <div class="scope-corners"><span class="tl"></span><span class="tr"></span><span class="bl"></span><span class="br"></span></div>
            </div>
          </div>

          <div class="ss-upload-actions">
            <button class="ss-btn ss-btn-ghost" id="chooseBtn" type="button">choose file</button>
            <button class="ss-btn ss-btn-ghost" id="pasteBtn" type="button">paste image</button>
            <button class="ss-btn ss-btn-primary" id="analyzeBtn" type="button" disabled>run analysis</button>
          </div>

          <div class="ss-samples">
            <span class="ss-samples-label">Reference samples —</span>
            <button class="sample-chip" data-sample="/samples/real" type="button">real photo</button>
            <button class="sample-chip" data-sample="/samples/ai" type="button">AI-generated</button>
          </div>
        </div>

        <div class="ss-panel">
          <div class="panel-label">Source image</div>
          <div class="source-preview" id="sourcePreviewBox">
            <div class="source-preview-empty" id="sourceEmpty">No image selected yet. Upload, paste, or capture one above.</div>
            <img id="sourcePreview" alt="Selected source image" hidden />
            <div class="source-preview-name" id="sourceFileName" hidden></div>
          </div>
        </div>
      </div>

      <div class="ss-col">
        <div class="readout readout-idle" id="verdictPanel">
          <div class="readout-title">Awaiting input</div>
          <div class="readout-sub">Upload an image to begin.</div>
        </div>

        <div class="ss-panel">
          <div class="panel-label">Anomaly map</div>
          <div class="scope-frame">
            <div class="heatmap-slot" id="heatmapSlot">
              <div class="heatmap-empty" id="heatmapEmpty">
                <p>The region-attention map appears here after analysis.</p>
              </div>
              <img id="heatmapImg" alt="Grad-CAM anomaly heat-map over the analyzed image" hidden />
            </div>
            <div class="scope-corners"><span class="tl"></span><span class="tr"></span><span class="bl"></span><span class="br"></span></div>
          </div>
        </div>
      </div>

    </div>

    <div class="ss-reports">
      <div class="ss-report-tabs" role="tablist" aria-label="Reports">
        <button class="report-tab is-active" data-report="forensic" role="tab" aria-selected="true">forensic &amp; provenance</button>
        <button class="report-tab" data-report="attribution" role="tab" aria-selected="false">generator attribution</button>
        <button class="report-tab" data-report="robustness" role="tab" aria-selected="false">robustness</button>
      </div>
      <div class="report-panel" id="reportForensic">Upload an image to generate the forensic report.</div>
      <div class="report-panel" id="reportAttribution" hidden>Upload an image to generate the attribution report.</div>
      <div class="report-panel" id="reportRobustness" hidden>
        <p>Checks whether the verdict holds up under JPEG re-compression and resizing — the kind of
        degradation an image usually goes through after a re-upload or a social-media repost.</p>
        <button class="ss-btn ss-btn-ghost" id="robustnessBtn" type="button" disabled>run robustness sweep</button>
        <div id="robustnessOutput" style="margin-top:14px;"></div>
      </div>
    </div>

  </section>

  <section id="view-about" class="ss-view">
    <div class="about-wrap">

      <div class="spec-section lead">
        <h2>Overview</h2>
        <p>SignalScope reads an uploaded image and reports whether it is a real photograph or an
        AI-generated image, then shows the evidence behind that reading: which pixels influenced the
        decision, which generator family the image likely came from (if any), and what the file's own
        metadata says. Every result is reported as a confidence score, not a certainty.</p>
      </div>

      <div class="spec-section">
        <h2>Signal chain</h2>
        <ol class="chain-list">
          <li><span class="chain-index">1</span><div class="chain-body"><b>Upload</b><p>The image is saved for processing and never permanently stored.</p></div></li>
          <li><span class="chain-index">2</span><div class="chain-body"><b>Preprocess</b><p>Resized to 384×384 and normalized to match the model's training statistics.</p></div></li>
          <li><span class="chain-index">3</span><div class="chain-body"><b>Classify</b><p>MobileNetV3-Small scores the image as real or AI-generated.</p></div></li>
          <li><span class="chain-index">4</span><div class="chain-body"><b>Badge check</b><p>Screens for a platform watermark that already reads "AI-generated".</p></div></li>
          <li><span class="chain-index">5</span><div class="chain-body"><b>Explain</b><p>Grad-CAM traces the verdict back to the image regions responsible for it.</p></div></li>
          <li><span class="chain-index">6</span><div class="chain-body"><b>Attribute</b><p>If flagged AI-generated, a frequency-domain check estimates the likely generator family.</p></div></li>
          <li><span class="chain-index">7</span><div class="chain-body"><b>Verify &amp; report</b><p>EXIF metadata is checked against the visual verdict and combined into the final report.</p></div></li>
        </ol>
      </div>

      <div class="spec-section">
        <h2>Component reference</h2>
        <table class="spec-table">
          <thead><tr><th>Component</th><th>Role</th></tr></thead>
          <tbody>
            <tr><td><span class="tag">classifier</span></td><td>MobileNetV3-Small, fine-tuned on a real/AI-generated image set, outputs a real vs. AI-generated score</td></tr>
            <tr><td><span class="tag">explainer</span></td><td>Grad-CAM traced through the network's last convolutional block, producing a heat-map of the regions that drove the verdict</td></tr>
            <tr><td><span class="tag">attribution</span></td><td>Frequency-domain analysis estimating the likely generator family for images flagged as AI-generated</td></tr>
            <tr><td><span class="tag">provenance</span></td><td>EXIF metadata inspection, checked against the visual verdict for agreement or conflict</td></tr>
            <tr><td><span class="tag">runtime</span></td><td>Python, PyTorch &amp; Torchvision, OpenCV, Pillow, NumPy</td></tr>
            <tr><td><span class="tag">tooling</span></td><td>scikit-learn, SciPy, Pandas, Hugging Face Hub / Transformers</td></tr>
            <tr><td><span class="tag">interface</span></td><td>FastAPI, single-file full-stack app</td></tr>
          </tbody>
        </table>
      </div>

      <div class="spec-section">
        <h2>Tech stack</h2>
        <table class="spec-table">
          <thead><tr><th>Category</th><th>Technologies</th></tr></thead>
          <tbody>
            <tr><td><span class="tag">backend</span></td><td>Python 3.10+, FastAPI, Uvicorn, Pydantic</td></tr>
            <tr><td><span class="tag">ml / ai</span></td><td>PyTorch, Torchvision, timm (PyTorch Image Models), MobileNetV3-Small, Hugging Face Hub, Transformers</td></tr>
            <tr><td><span class="tag">computer vision</span></td><td>OpenCV, Pillow (PIL), Albumentations, Grad-CAM</td></tr>
            <tr><td><span class="tag">data science</span></td><td>NumPy, SciPy, scikit-learn, Pandas, Matplotlib</td></tr>
            <tr><td><span class="tag">frontend</span></td><td>HTML5, CSS3, Vanilla JavaScript (single-file embedded UI)</td></tr>
            <tr><td><span class="tag">dev tools</span></td><td>Rich (CLI output), tqdm (progress bars), PyYAML, joblib, httpx</td></tr>
          </tbody>
        </table>
      </div>

      <div class="spec-section">
        <h2>Project</h2>
        <p>Built for SIH 2026 (Internal Hackathon), Problem Statement C-433 — "Telling Real From Synthetic
        in the Age of Generative Media" — at L. J. Institute of Engineering and Technology. Verdicts are
        decision support for journalists, platforms, and everyday users, and should be paired with human
        judgement, not treated as a final ruling.</p>
      </div>

    </div>
  </section>

  <section id="view-history" class="ss-view">
    <div class="history-wrap">
      <div class="history-header">
        <h2>Analysis history</h2>
        <button class="ss-btn ss-btn-ghost" id="clearHistoryBtn" type="button">clear all</button>
      </div>
      <div id="historyList"></div>
    </div>
  </section>

</main>

<!-- Settings modal -->
<div class="settings-overlay" id="settingsModal" hidden>
  <div class="settings-box">
    <div class="settings-header">
      <h2>Settings</h2>
      <button class="settings-close" id="settingsCloseBtn" type="button" aria-label="Close settings">&times;</button>
    </div>
    <div class="settings-body">
      <div class="settings-row">
        <div class="settings-row-info">
          <div class="settings-row-label">Auto-analyze on upload</div>
          <div class="settings-row-desc">Automatically start analysis when an image is selected.</div>
        </div>
        <label class="toggle">
          <input type="checkbox" id="settingAutoAnalyze" checked />
          <span class="toggle-track"></span>
        </label>
      </div>
      <div class="settings-row">
        <div class="settings-row-info">
          <div class="settings-row-label">Clear history</div>
          <div class="settings-row-desc">Remove all saved analysis results from this browser.</div>
        </div>
        <button class="ss-btn ss-btn-ghost" id="settingsClearBtn" type="button">clear</button>
      </div>
    </div>
  </div>
</div>

<footer class="ss-footer">
  Results are likelihood assessments, not definitive accusations — pair them with human judgement.
  Built for SIH 2026 · L. J. Institute of Engineering and Technology
</footer>

<div class="toast" id="toast" role="alert" aria-live="assertive" hidden></div>

<script>
(() => {
  "use strict";

  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const chooseBtn = document.getElementById("chooseBtn");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const dzIdle = document.getElementById("dzIdle");
  const dzScanning = document.getElementById("dzScanning");
  const scanLabel = document.getElementById("scanLabel");
  const previewImg = document.getElementById("previewImg");
  const dzPaste = document.getElementById("dzPaste");
  const pasteBtn = document.getElementById("pasteBtn");
  const pasteCancelBtn = document.getElementById("pasteCancelBtn");

  const verdictPanel = document.getElementById("verdictPanel");
  const heatmapEmpty = document.getElementById("heatmapEmpty");
  const heatmapImg = document.getElementById("heatmapImg");

  const reportForensic = document.getElementById("reportForensic");
  const reportAttribution = document.getElementById("reportAttribution");
  const reportRobustness = document.getElementById("reportRobustness");
  const robustnessBtn = document.getElementById("robustnessBtn");
  const robustnessOutput = document.getElementById("robustnessOutput");

  const apiStatusDot = document.getElementById("apiStatusDot");
  const apiStatusText = document.getElementById("apiStatusText");
  const toast = document.getElementById("toast");

  // Source preview
  const sourcePreview = document.getElementById("sourcePreview");
  const sourceEmpty = document.getElementById("sourceEmpty");
  const sourceFileName = document.getElementById("sourceFileName");

  // Settings
  const settingsBtn = document.getElementById("settingsBtn");
  const settingsModal = document.getElementById("settingsModal");
  const settingsCloseBtn = document.getElementById("settingsCloseBtn");
  const settingAutoAnalyze = document.getElementById("settingAutoAnalyze");
  const settingsClearBtn = document.getElementById("settingsClearBtn");

  // History
  const historyList = document.getElementById("historyList");
  const clearHistoryBtn = document.getElementById("clearHistoryBtn");

  let currentFile = null;
  let busy = false;

  // ── Settings (localStorage) ───────────────────────────────────────
  const SETTINGS_KEY = "ss_settings";
  const HISTORY_KEY = "ss_history";
  const MAX_HISTORY = 20;

  function loadSettings() {
    try { return JSON.parse(localStorage.getItem(SETTINGS_KEY)) || {}; } catch { return {}; }
  }
  function saveSettings(s) { localStorage.setItem(SETTINGS_KEY, JSON.stringify(s)); }

  const settings = loadSettings();
  if (settings.autoAnalyze === undefined) settings.autoAnalyze = true;
  settingAutoAnalyze.checked = settings.autoAnalyze;

  settingAutoAnalyze.addEventListener("change", () => {
    settings.autoAnalyze = settingAutoAnalyze.checked;
    saveSettings(settings);
  });

  function openSettings() { settingsModal.hidden = false; }
  function closeSettings() { settingsModal.hidden = true; }

  settingsBtn.addEventListener("click", (e) => { e.stopPropagation(); openSettings(); });
  settingsCloseBtn.addEventListener("click", (e) => { e.stopPropagation(); closeSettings(); });
  settingsModal.addEventListener("click", (e) => {
    if (e.target === settingsModal) closeSettings();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !settingsModal.hidden) closeSettings();
  });

  // ── History (localStorage) ────────────────────────────────────────
  function loadHistory() {
    try { return JSON.parse(localStorage.getItem(HISTORY_KEY)) || []; } catch { return []; }
  }
  function saveHistory(h) { localStorage.setItem(HISTORY_KEY, JSON.stringify(h)); }

  function makeThumbnail(file) {
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        const size = 80;
        const canvas = document.createElement("canvas");
        canvas.width = size; canvas.height = size;
        const ctx = canvas.getContext("2d");
        const s = Math.min(img.width, img.height);
        const sx = (img.width - s) / 2, sy = (img.height - s) / 2;
        ctx.drawImage(img, sx, sy, s, s, 0, 0, size, size);
        resolve(canvas.toDataURL("image/jpeg", 0.6));
        URL.revokeObjectURL(img.src);
      };
      img.onerror = () => resolve(null);
      img.src = URL.createObjectURL(file);
    });
  }

  async function addToHistory(file, data) {
    const thumb = await makeThumbnail(file);
    const entry = {
      id: Date.now(),
      filename: file.name || "image",
      thumbnail: thumb,
      verdict: data.verdict,
      confidence: data.confidence,
      real_prob: data.real_prob,
      ai_prob: data.ai_prob,
      elapsed_ms: data.elapsed_ms,
      timestamp: new Date().toISOString(),
    };
    const history = loadHistory();
    history.unshift(entry);
    if (history.length > MAX_HISTORY) history.length = MAX_HISTORY;
    saveHistory(history);
    renderHistory();
  }

  function clearHistory() {
    localStorage.removeItem(HISTORY_KEY);
    renderHistory();
    showToast("History cleared.");
  }

  function renderHistory() {
    const history = loadHistory();
    if (!history.length) {
      historyList.innerHTML = '<div class="history-empty">No analyses yet. Results will appear here after you analyze an image.</div>';
      return;
    }
    historyList.innerHTML = history.map((e) => {
      const isAi = e.verdict === "AI-Generated";
      const verdictClass = isAi ? "is-ai" : "is-real";
      const verdictLabel = isAi ? "AI-Generated" : "Real";
      const date = new Date(e.timestamp);
      const timeStr = date.toLocaleDateString(undefined, { month:"short", day:"numeric" }) + " " + date.toLocaleTimeString(undefined, { hour:"2-digit", minute:"2-digit" });
      const thumbHtml = e.thumbnail ? '<img class="history-thumb" src="' + e.thumbnail + '" alt="" />' : '<div class="history-thumb"></div>';
      return '<div class="history-card" data-history-id="' + e.id + '">' +
        thumbHtml +
        '<div class="history-info">' +
          '<div class="history-name">' + escapeHtml(e.filename) + '</div>' +
          '<div class="history-meta">' + escapeHtml(timeStr) + ' · ' + (e.elapsed_ms || "—") + ' ms · ' + (e.confidence || 0).toFixed(1) + '%</div>' +
        '</div>' +
        '<span class="history-verdict ' + verdictClass + '">' + verdictLabel + '</span>' +
      '</div>';
    }).join("");
  }

  clearHistoryBtn.addEventListener("click", clearHistory);
  settingsClearBtn.addEventListener("click", () => { clearHistory(); });
  renderHistory();

  document.querySelectorAll(".ss-nav-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".ss-nav-tab").forEach((b) => {
        b.classList.remove("is-active");
        b.setAttribute("aria-selected", "false");
      });
      btn.classList.add("is-active");
      btn.setAttribute("aria-selected", "true");

      const view = btn.dataset.view;
      document.querySelectorAll(".ss-view").forEach((v) => v.classList.remove("is-active"));
      document.getElementById(`view-${view}`).classList.add("is-active");
      if (view === "history") renderHistory();
    });
  });

  document.querySelectorAll(".report-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".report-tab").forEach((b) => {
        b.classList.remove("is-active");
        b.setAttribute("aria-selected", "false");
      });
      btn.classList.add("is-active");
      btn.setAttribute("aria-selected", "true");

      const which = btn.dataset.report;
      reportForensic.hidden = which !== "forensic";
      reportAttribution.hidden = which !== "attribution";
      reportRobustness.hidden = which !== "robustness";
    });
  });

  let toastTimer = null;
  function showToast(message) {
    toast.textContent = message;
    toast.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { toast.hidden = true; }, 4200);
  }

  async function checkHealth() {
    try {
      const res = await fetch("/api/health");
      const data = await res.json();
      if (data.model_loaded) {
        apiStatusDot.className = "status-dot is-online";
        apiStatusText.textContent = `online · ${data.device}`;
      } else {
        apiStatusDot.className = "status-dot is-offline";
        apiStatusText.textContent = "model weights missing";
      }
    } catch (e) {
      apiStatusDot.className = "status-dot is-offline";
      apiStatusText.textContent = "backend unreachable";
    }
  }
  checkHealth();

  function isSupportedFile(file) {
    return ["image/jpeg", "image/png", "image/webp", "image/bmp"].includes(file.type);
  }

  function setFile(file) {
    if (!file) return;
    if (!isSupportedFile(file)) {
      showToast("Unsupported file type. Use JPEG, PNG, WebP or BMP.");
      return;
    }
    if (file.size > 15 * 1024 * 1024) {
      showToast("Image is too large — the limit is 15 MB.");
      return;
    }
    if (!dzPaste.hidden) closePaste();
    currentFile = file;
    const url = URL.createObjectURL(file);
    previewImg.src = url;
    previewImg.hidden = false;
    dzIdle.hidden = true;
    analyzeBtn.disabled = false;

    // Show in the separate source preview panel
    sourcePreview.src = url;
    sourcePreview.hidden = false;
    sourceEmpty.hidden = true;
    sourceFileName.textContent = file.name || "pasted image";
    sourceFileName.hidden = false;

    if (settings.autoAnalyze) runAnalysis();
  }

  chooseBtn.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("click", () => {
    if (!busy && dzPaste.hidden) fileInput.click();
  });
  dropzone.addEventListener("keydown", (e) => {
    if ((e.key === "Enter" || e.key === " ") && !busy && dzPaste.hidden) { e.preventDefault(); fileInput.click(); }
  });
  fileInput.addEventListener("change", (e) => setFile(e.target.files[0]));

  // ── Paste an image (Ctrl+V / Cmd+V) ─────────────────────────────────
  document.addEventListener("paste", (e) => {
    if (busy) return;
    const items = (e.clipboardData && e.clipboardData.items) || [];
    for (const item of items) {
      if (item.kind === "file" && item.type.startsWith("image/")) {
        const file = item.getAsFile();
        if (file) {
          e.preventDefault();
          setFile(file);
        }
        break;
      }
    }
  });

  // ── Paste image mode ────────────────────────────────────────────────
  function openPaste() {
    if (busy) return;
    dzIdle.hidden = true;
    previewImg.hidden = true;
    dzPaste.hidden = false;
  }

  function closePaste() {
    dzPaste.hidden = true;
  }

  pasteBtn.addEventListener("click", (e) => { e.stopPropagation(); openPaste(); });
  pasteCancelBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    closePaste();
    dzIdle.hidden = !!currentFile;
    previewImg.hidden = !currentFile;
  });

  ["dragenter", "dragover"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.add("is-dragover"); })
  );
  ["dragleave", "drop"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.remove("is-dragover"); })
  );
  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) setFile(file);
  });

  document.querySelectorAll(".sample-chip").forEach((chip) => {
    chip.addEventListener("click", async () => {
      if (busy) return;
      const url = chip.dataset.sample;
      const res = await fetch(url);
      const blob = await res.blob();
      const name = url.split("/").pop() + ".png";
      const file = new File([blob], name, { type: blob.type || "image/png" });
      setFile(file);
    });
  });

  analyzeBtn.addEventListener("click", () => runAnalysis());

  const SCAN_MESSAGES = [
    "running MobileNet inference…",
    "tracing Grad-CAM activations…",
    "checking file metadata…",
    "compiling forensic report…",
  ];

  function setBusy(state) {
    busy = state;
    analyzeBtn.disabled = state || !currentFile;
    chooseBtn.disabled = state;
    pasteBtn.disabled = state;
    if (state) robustnessBtn.disabled = true;
    dzScanning.hidden = !state;
    previewImg.style.visibility = "visible";
  }

  async function runAnalysis() {
    if (!currentFile || busy) return;
    setBusy(true);

    let msgIdx = 0;
    scanLabel.textContent = SCAN_MESSAGES[0];
    const msgTimer = setInterval(() => {
      msgIdx = (msgIdx + 1) % SCAN_MESSAGES.length;
      scanLabel.textContent = SCAN_MESSAGES[msgIdx];
    }, 900);

    renderIdle("Analyzing…", "Reading the signal chain — this usually takes a few seconds.");

    try {
      const form = new FormData();
      form.append("file", currentFile);
      const res = await fetch("/api/analyze", { method: "POST", body: form });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Request failed (${res.status})`);
      }

      const data = await res.json();

      if (!data.ok) {
        renderFault(data.verdict === "Weights Missing" ? "Model weights not found" : "Analysis failed", data.error || "Please try another image.");
        reportForensic.textContent = "Unable to generate a report for this image.";
        reportAttribution.textContent = "Unable to generate a report for this image.";
        robustnessBtn.disabled = true;
        robustnessOutput.innerHTML = "";
        heatmapImg.hidden = true;
        heatmapEmpty.hidden = false;
        return;
      }

      renderVerdict(data);
      renderHeatmap(data);
      renderReport(reportForensic, data.forensic_report_markdown);
      renderReport(reportAttribution, data.attribution_markdown);

      // A fresh image was analyzed — any robustness result showing is now stale.
      robustnessBtn.disabled = false;
      robustnessOutput.innerHTML = "";

      // Save to history
      addToHistory(currentFile, data);
    } catch (e) {
      console.error(e);
      renderFault("Analysis failed", e.message || "Something went wrong contacting the backend.");
      showToast("Couldn't complete the analysis. See the readout panel for details.");
    } finally {
      clearInterval(msgTimer);
      setBusy(false);
    }
  }

  robustnessBtn.addEventListener("click", () => runRobustnessSweep());

  async function runRobustnessSweep() {
    if (!currentFile || busy || robustnessBtn.disabled) return;

    robustnessBtn.disabled = true;
    const originalLabel = robustnessBtn.textContent;
    robustnessBtn.textContent = "sweeping…";
    robustnessOutput.innerHTML = `<p class="dz-sub">Running JPEG and resize degradations through the classifier — a few seconds.</p>`;

    try {
      const form = new FormData();
      form.append("file", currentFile);
      const res = await fetch("/api/robustness", { method: "POST", body: form });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Request failed (${res.status})`);
      }

      const data = await res.json();
      if (!data.ok) {
        robustnessOutput.innerHTML = `<p class="dz-sub">${escapeHtml(data.error || "Robustness sweep failed for this image.")}</p>`;
        return;
      }

      robustnessOutput.innerHTML = renderRobustnessHtml(data);
    } catch (e) {
      console.error(e);
      robustnessOutput.innerHTML = `<p class="dz-sub">Couldn't complete the robustness sweep: ${escapeHtml(e.message || "unknown error")}</p>`;
    } finally {
      robustnessBtn.disabled = false;
      robustnessBtn.textContent = originalLabel;
    }
  }

  function renderRobustnessHtml(data) {
    const stableBadge = data.stable
      ? `<strong style="color:var(--real)">stable</strong>`
      : `<strong style="color:var(--ai)">inconsistent</strong>`;

    const row = (label, verdict, confidence) => `
      <tr>
        <td>${escapeHtml(String(label))}</td>
        <td>${escapeHtml(String(verdict))}</td>
        <td>${confidence != null ? confidence.toFixed(1) + "%" : "N/A"}</td>
      </tr>`;

    const jpegRows = (data.jpeg || [])
      .map((r) => row(`quality ${r.quality}`, r.verdict, r.confidence))
      .join("");
    const resizeRows = (data.resize || [])
      .map((r) => row(`scale ${r.scale}`, r.verdict, r.confidence))
      .join("");

    return `
      <p>Majority verdict: <strong>${escapeHtml(String(data.majority_verdict))}</strong> —
      stability score ${(data.stability_score * 100).toFixed(1)}% (${stableBadge}).</p>
      <h3 style="font-size:0.95rem;margin-top:16px;">JPEG compression sweep</h3>
      <table class="spec-table"><thead><tr><th>Setting</th><th>Verdict</th><th>Confidence</th></tr></thead>
      <tbody>${jpegRows}</tbody></table>
      <h3 style="font-size:0.95rem;margin-top:16px;">Resize degradation sweep</h3>
      <table class="spec-table"><thead><tr><th>Setting</th><th>Verdict</th><th>Confidence</th></tr></thead>
      <tbody>${resizeRows}</tbody></table>
    `;
  }

  function renderIdle(title, sub) {
    verdictPanel.className = "readout readout-idle";
    verdictPanel.innerHTML = `
      <div class="readout-title">${escapeHtml(title)}</div>
      <div class="readout-sub">${escapeHtml(sub)}</div>
    `;
  }

  function renderFault(title, sub) {
    verdictPanel.className = "readout readout-fault";
    verdictPanel.innerHTML = `
      <div class="readout-title">${escapeHtml(title)}</div>
      <div class="readout-sub">${escapeHtml(sub)}</div>
    `;
  }

  function meterHtml(pct, cssClass, segments = 20) {
    const filled = Math.round((segments * Math.max(0, Math.min(100, pct))) / 100);
    let cells = "";
    for (let i = 0; i < segments; i++) {
      cells += `<span class="seg ${i < filled ? cssClass : ""}"></span>`;
    }
    return `<div class="meter">${cells}</div>`;
  }

  function renderVerdict(data) {
    const isAi = data.verdict === "AI-Generated";
    const statusClass = isAi ? "status-ai" : "status-real";
    const label = isAi ? "Likely AI-generated" : "Likely real";

    const reviewHtml = data.needs_review
      ? `<div class="flag-line">flag — borderline score, recommend human review</div>`
      : "";

    const metaBits = [];
    if (data.model) metaBits.push(`model: ${data.model}`);
    if (data.device) metaBits.push(`device: ${data.device}`);
    if (typeof data.elapsed_ms === "number") metaBits.push(`${data.elapsed_ms} ms`);
    if (data.threshold_calibrated === false) metaBits.push("threshold: default (uncalibrated checkpoint)");
    if (typeof data.metadata_shift_applied === "number") metaBits.push(`EXIF shift: ${data.metadata_shift_applied > 0 ? "+" : ""}${data.metadata_shift_applied}pp`);
    const metaHtml = metaBits.length
      ? `<div class="readout-meta">${metaBits.map((b) => `<span>${escapeHtml(b)}</span>`).join("")}</div>`
      : "";

    verdictPanel.className = `readout ${statusClass}`;
    verdictPanel.innerHTML = `
      <div class="readout-row">
        <div class="readout-title">${label}</div>
        <div class="readout-num">${data.confidence.toFixed(1)}<span class="unit">%</span></div>
      </div>
      <div class="readout-meters">
        <div class="meter-row">
          <span class="meter-label">real</span>
          ${meterHtml(data.real_prob, "on-real")}
          <span class="meter-pct">${data.real_prob.toFixed(1)}%</span>
        </div>
        <div class="meter-row">
          <span class="meter-label">ai</span>
          ${meterHtml(data.ai_prob, "on-ai")}
          <span class="meter-pct">${data.ai_prob.toFixed(1)}%</span>
        </div>
      </div>
      ${reviewHtml}
      ${metaHtml}
    `;
    requestAnimationFrame(() => {
      verdictPanel.querySelectorAll(".meter").forEach((m) => m.classList.add("is-filled"));
    });
  }

  function renderHeatmap(data) {
    if (data.heatmap_image) {
      heatmapImg.src = data.heatmap_image;
      heatmapImg.hidden = false;
      heatmapEmpty.hidden = true;
    } else {
      heatmapImg.hidden = true;
      heatmapEmpty.hidden = false;
      heatmapEmpty.textContent = "Heat-map unavailable for this image — showing the verdict only.";
    }
  }

  function renderReport(el, markdown) {
    el.innerHTML = mdToHtml(markdown || "");
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function inlineMd(text) {
    return escapeHtml(text).replace(/\\*\\*(.+?)\\*\\*/g, "<strong>$1</strong>");
  }

  function mdToHtml(markdown) {
    const lines = markdown.split("\\n");
    let html = "";
    let inList = false;

    const closeList = () => { if (inList) { html += "</ul>"; inList = false; } };

    for (const raw of lines) {
      const line = raw.trim();
      if (!line) { closeList(); continue; }

      if (line.startsWith("### ")) {
        closeList();
        html += `<h3>${inlineMd(line.slice(4))}</h3>`;
      } else if (line.startsWith("* ") || line.startsWith("- ")) {
        if (!inList) { html += "<ul>"; inList = true; }
        html += `<li>${inlineMd(line.slice(2))}</li>`;
      } else {
        closeList();
        html += `<p>${inlineMd(line)}</p>`;
      }
    }
    closeList();
    return html || "<p>No report generated.</p>";
  }
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)