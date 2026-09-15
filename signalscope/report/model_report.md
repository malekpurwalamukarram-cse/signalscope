# SignalScope — Model Report (PS-2)

## Task Description
Binary classification of images into **Real** (authentic photographs) vs. **AI-Generated** (synthetic images from GANs, Diffusion Models, VAEs, etc.).

The key challenge is **unseen generator generalization**: the model must correctly classify images produced by generators absent from the training set, while maintaining robustness to post-processing degradations (JPEG compression, resizing, noise).

---

## Dataset Split
| Split | Description | Notes |
|---|---|---|
| Train | Real photos + AI images from known generators | Known GAN/Diffusion sources |
| Validation (Seen) | Held-out portion of training generators | Same generator distribution |
| Test (Unseen) | Images from **novel generators absent from training** | Primary evaluation target |

> **Suggested datasets**: CIFAKE, GenImage, ArtiFact, or custom-collected DALL-E / Stable Diffusion / Midjourney samples.

---

## Model Architecture
| Component | Details |
|---|---|
| **Primary detector (deployed)** | MobileNetV3-Small, fine-tuned on the CIFAKE real/synthetic image dataset. This is the model actually loaded by `signalscope/model/inference.py` and served by `app.py`. |
| **Forensic evidence** | Rule-grounded textual cue report + FFT-based generator attribution (Bonus B) |
| **Explainability** | Grad-CAM, hooked onto the last block of `MobileNetV3.features` (see `signalscope/src/forensic_explainer.py`) |
| **Input size** | 384 × 384 |
| **Parameters** | ~2.5M trainable classifier-head parameters on top of the MobileNetV3-Small backbone |
| **Device** | CUDA when available, CPU fallback |

> **Note on `architecture.py`**: `signalscope/model/architecture.py` also defines a heavier
> experimental dual-stream network (EfficientNet-B1 spatial stream + a learned FFT frequency
> stream, `SignalScopeModel`). That architecture is **not** what's deployed — the production
> path uses the single-stream MobileNetV3-Small described above via `train_cifake.py` /
> `inference.py`. `architecture.py` is kept for reference / future experimentation; if it's
> not going to be trained further, it should be labelled `experimental/` to avoid confusing
> reviewers about what's actually running.

### Unseen-Generator Robustness Augmentations
| Augmentation | Purpose |
|---|---|
| JPEG compression (Q=55–100) | Simulate post-processing by end users |
| Downscale + upsample (50–90%) | Simulate resize/repost artifacts |
| GaussianBlur (3–5px) | Simulate social media re-encoding |
| GaussNoise (σ=5–30) | Generalization noise |
| ColorJitter | Reduce colour-channel fingerprint overfitting |

---

## Training Protocol
| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW (weight_decay=1e-4) |
| Learning rate | 5e-4 |
| Scheduler | CosineAnnealingLR |
| Epochs | 5 |
| Batch size | 64 |
| Label smoothing | 0.05 |
| Mixed precision | FP16 via `torch.amp` on CUDA |

`train_cifake.py` calibrates the AI-decision threshold on a held-out validation split
(`choose_ai_threshold()`, subject to a target real-photo false-positive rate) and stores
it in the checkpoint under `ai_decision_threshold`. `inference.py` reads that value at
load time and uses it as the decision boundary; metadata (EXIF/provenance) is fused in
as a secondary signal via a bounded confidence shift, so a strong model signal (e.g. 90%
AI) is not overridden by metadata alone, but a genuinely borderline case can be nudged
by camera EXIF or AI-generator software tags.

The active checkpoint is saved to `signalscope/model/weights/cifake_384_finetuned/signalscope_cifake_best.pth`.

> ⚠️ **Calibration status of the currently shipped checkpoint**: the `.pth` file bundled
> in this submission does **not** contain an `ai_decision_threshold` key — it appears to
> predate the calibration step in `train_cifake.py`. `inference.py` detects this and falls
> back to a hardcoded default (0.55), and both the API response and the CLI (`--details`)
> now flag this explicitly (`threshold_calibrated: false`) rather than silently presenting
> the fallback as if it were calibrated. **Re-run `train_cifake.py` before final judging**
> so the shipped checkpoint contains a real calibrated threshold end-to-end.

---

## Evaluation Metrics

The table below previously reported placeholder-style figures. The values here are read
directly from the shipped checkpoint's stored metadata (`epoch`, `accuracy`, `macro_f1`,
`roc_auc`), not re-derived from a fresh test run — treat them as what the checkpoint
*claims* about itself, and re-validate against the organizer's held-out split before
presenting them as the final benchmark.

| Metric | Value (per checkpoint metadata) |
|---|---|
| **Accuracy** | **97.3%** |
| **Macro-F1** | **0.973** |
| **ROC-AUC** | **0.997** |

> No confusion matrix or per-class breakdown is currently stored in the checkpoint. If a
> confusion matrix is wanted for the report, add it to the `ckpt = {...}` dict saved in
> `train_cifake.py` (alongside `ai_decision_threshold`) so it travels with the weights,
> or regenerate it by re-running evaluation against the CIFAKE test split.

---

## Bonus Modules

All four bonus modules below are wired into the running app (`app.py`), not just
implemented in isolation:

### Bonus A — Explainable AI
- **Grad-CAM** (`signalscope/src/forensic_explainer.py`): spatial heatmap localising the
  regions that most influenced the verdict, returned by `POST /api/analyze` as
  `heatmap_image` and rendered in the "Anomaly map" panel.
- **Rule-grounded cue report**: human-readable forensic text generated for every
  prediction, shown in the "forensic & provenance" report tab.

### Bonus B — Generator Attribution
- Heuristic spectral classifier (`signalscope/src/attribution.py`) attributing
  AI-generated images to a GAN family (StyleGAN/ProGAN) vs. a Diffusion family (Stable
  Diffusion/DALL-E), based on FFT periodicity and high-frequency energy ratio. Runs
  automatically for any image classified AI-Generated; shown in the "generator
  attribution" report tab.

### Bonus C — Robustness Analysis
- Automated sweep (`signalscope/src/robustness.py`) over 6 JPEG quality levels and 4
  resize scales, reporting a verdict stability score and majority-vote verdict.
- Exposed as its own endpoint, `POST /api/robustness`, and a "run robustness sweep"
  button in the "robustness" report tab — kept separate from the main
  `POST /api/analyze` call since it costs ~10 extra forward passes per run and isn't
  needed for every single upload.

### Bonus D — Provenance & Metadata
- EXIF camera-hardware tags and AI-generator software/PNG-text fingerprints
  (`signalscope/src/provenance.py`), checked against the visual verdict.
- This now **actually feeds the decision**, not just the report: `app.py` computes the
  metadata first and passes it into `predict_with_details(image, metadata=...)`, which
  applies a bounded confidence shift for borderline cases (e.g. genuine camera EXIF
  nudges a near-threshold score back toward "Real"). When a shift is applied, it's
  reported back as `metadata_shift_applied` and shown in the verdict panel.

---

## Known Limitations & Failure Analysis
1. **Extremely high-quality AI images**: Recent FLUX/Stable Diffusion XL outputs approaching photorealism may fool the model.
2. **Heavily processed real photos**: HDR, artistic filters, or extreme edits may be misclassified as AI-Generated.
3. **Text/screenshots**: Model is trained on photographic content; text/document images may produce uncertain results.
4. **Very low resolution**: Images below 64×64 pixels lack sufficient spectral/spatial information.

---

## CLI Usage
```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Run inference
python signalscope/model/predict.py --image photo.jpg
python signalscope/model/predict.py --image photo.jpg --details
python signalscope/model/predict.py --image photo.jpg --json
```

`--details` / `--json` now also read the image's EXIF/provenance metadata and feed it
into the verdict (previously this metadata was collected but silently discarded due to
a key mismatch), and will report `threshold_calibrated: false` if the loaded checkpoint
has no stored calibration threshold.

## Python API
```python
from signalscope.model.inference import predict
label = predict("path/to/image.jpg")  # returns "Real" or "AI-Generated"

# With provenance fused into the decision:
from signalscope.model.inference import predict_with_details
from signalscope.src.provenance import inspect_metadata

metadata = inspect_metadata("path/to/image.jpg")
result = predict_with_details("path/to/image.jpg", metadata=metadata)
```

## Web App
```bash
python app.py
# open http://localhost:7860
```
- `POST /api/analyze` — full pipeline: classification (with metadata fusion) + Grad-CAM
  + forensic report + generator attribution.
- `POST /api/robustness` — Bonus C stability sweep, run on demand from the UI.
- `GET /api/health` — model-load status.