# 🔍 SignalScope — Telling Real From Synthetic in the Age of Generative Media

**SIH 2026 (Internal Hackathon)** · Problem Statement **C-433 / PS-2** · L. J. Institute
of Engineering and Technology

![Task](https://img.shields.io/badge/task-real%20vs%20AI--generated-blueviolet)
![Backbone](https://img.shields.io/badge/backbone-MobileNetV3--Small-orange)
![Dataset](https://img.shields.io/badge/dataset-CIFAKE-informational)
![Status](https://img.shields.io/badge/calibration-pending-yellow)

SignalScope takes an uploaded image and tells you whether it's a **real photograph**
or **AI-generated** — and, unlike a black-box verdict, *shows its work*: a Grad-CAM
anomaly heat-map, a rule-grounded forensic report, EXIF/provenance signals, a
generator-family attribution for AI images, and an on-demand robustness check against
JPEG/resize degradation. Every verdict is framed as a calibrated likelihood ("likely
AI-generated"), never an accusation — and the system makes no claims about real,
identifiable people.

---

## ✨ Modules Built

| Module | What it does | Status |
|---|---|---|
| 🧠 **Core Classifier** | Real-vs-AI-Generated verdict + confidence score | ✅ MobileNetV3-Small fine-tuned on CIFAKE |
| 🔥 **Bonus A — Faithful Explanation** | Grad-CAM heat-map + human-readable, rule-grounded cue report | ✅ Built |
| 🧬 **Bonus B — Generator Attribution** | Names the likely family: GAN vs. Diffusion | ✅ FFT / spectral heuristic |
| 🛡️ **Bonus C — Robustness to Degradation** | JPEG-quality + resize sweep, stability score | ✅ On-demand `/api/robustness` |
| 📷 **Bonus D — Provenance & Metadata** | EXIF + AI-generator tags, fused into the verdict | ✅ Built |
| 🗣️ **Bonus E — Multimodal (image + text)** | Caption/claim consistency check | ❌ Not attempted |
| ⚡ **Bonus F — Real-Time / Deployable** | Drag-and-drop web app, instant analysis | ✅ Single-process FastAPI UI |
| 🎯 **Bonus G — Active Defence Analysis** | Adversarial / failure-mode study | ❌ Not attempted |

---

## 🏗️ Architecture At a Glance

```
Image (+ optional EXIF)
        │
        ▼
 📷 Provenance check  ──────────┐
        │                       │
        ▼                       │
 🧠 MobileNetV3-Small (384×384) │  (bounded confidence fusion
        │                       │   for borderline cases only)
        ▼                       │
 ✅ Metadata-fused verdict  ◄───┘
        │
        ├──► 🔥 Grad-CAM explainer        (Bonus A)
        ├──► 🧬 Generator attribution     (Bonus B, AI-flagged only)
        └──► 🛡️ Robustness sweep, on demand (Bonus C)
        │
        ▼
 🖥️  Responsible UI — "likely AI-generated", never "certain"
```

| Path | What it is |
|---|---|
| `app.py` | Single-file FastAPI app — backend + embedded frontend. `python app.py` runs the whole thing. |
| `signalscope/model/` | Model architecture, training scripts, inference, and weights. |
| `signalscope/src/` | Bonus modules: Grad-CAM explainer, generator attribution, EXIF provenance, robustness sweep. |
| `signalscope/report/model_report.md` | Full model card: architecture, metrics, bonus write-ups, known limitations. |
| `test_samples/` | Sample real/AI images for quick "try it" testing. |
| `data/cifake/` | Cached CIFAKE dataset parquet files — training only, not needed to run the app. |

---

## 🚀 Setup and Run Instructions

> ⏱️ A judge should be able to reproduce a prediction in well under 10 minutes.

### 1️⃣ Clone & install

```bash
git clone https://github.com/malekpurwalamukarram-cse/signalscope.git
cd signalscope

python -m venv .venv
.venv\Scripts\activate        # Mac/linux: source .venv/bin/activate

pip install -r requirements.txt  # CPU
# or, on a CUDA machine:
pip install -r requirements-gpu.txt
```

### 2️⃣ Run the web app (core + all bonus modules)

```bash
python app.py
# or: uvicorn app:app --host 0.0.0.0 --port 7860
```

Open **http://localhost:7860**, upload an image (or use one of the built-in samples),
and get a verdict, Grad-CAM heat-map, forensic/provenance report, generator
attribution, and an optional robustness sweep.

| Endpoint | Purpose |
|---|---|
| `POST /api/analyze` | Full pipeline — classification (with metadata fusion) + Grad-CAM + forensic report + generator attribution |
| `POST /api/robustness` | Bonus C stability sweep, run on demand from the UI |
| `GET /api/health` | Model-load status |

### 3️⃣ Or run the CLI (required `predict` interface)

```bash
python signalscope/model/predict.py --image path/to/photo.jpg
python signalscope/model/predict.py --image path/to/photo.jpg --details
python signalscope/model/predict.py --image path/to/photo.jpg --json
```

### 4️⃣ Or call it from Python

```python
from signalscope.model.inference import predict
label = predict("path/to/image.jpg")  # "Real" or "AI-Generated"

# With provenance fused into the decision:
from signalscope.model.inference import predict_with_details
from signalscope.src.provenance import inspect_metadata

metadata = inspect_metadata("path/to/image.jpg")
result = predict_with_details("path/to/image.jpg", metadata=metadata)
```

### 📦 Model weights

`signalscope/model/weights/*.pth` are excluded from git (see `.gitignore`) since
they're large binaries. Attach them to a **GitHub Release** on this repo and link it
here before submission:

> **Weights download:** https://github.com/malekpurwalamukarram-cse/signalscope/releases/tag/v1.0

### 🔁 Retraining (optional)

```bash
python signalscope/model/train_cifake.py
```

Downloads CIFAKE via Hugging Face Hub, fine-tunes MobileNetV3-Small at 384×384, and
saves the best checkpoint (with a validation-calibrated decision threshold) to
`signalscope/model/weights/cifake_384_finetuned/`.

---

## 🗃️ Datasets Used

| Split | Dataset | Size | Source / Licence | Role |
|---|---|---|---|---|
| Train / Validation | **CIFAKE** (real photos + Stable-Diffusion-generated images) | ~120K images | Hugging Face Hub, CIFAKE (CC / research-use, derived from CIFAR-10 + SD1.4) | Trained and validated on this — cached locally under `data/cifake/` |
| Held-out test | Organizers' unseen-generator test set | Provided at judging | LJIET / SIH-2026 organizers | **Not trained on.** Evaluated only via the `predict` interface above, per the data rules |

No other public dataset (e.g. GenImage, ArtiFact) has been added to training in this
submission. No images of real, identifiable people were sourced or used.

---

## 📊 Reported Metrics

> The figures below are read from the shipped checkpoint's own stored metadata
> (`accuracy`, `macro_f1`, `roc_auc` saved by `train_cifake.py`), evaluated on the
> CIFAKE validation split — **not** the organizers' held-out set, since that set is
> only available at judging time.

| Metric | Value | Split |
|---|---|---|
| 🎯 Accuracy | **97.3%** | CIFAKE validation |
| 📐 Macro-F1 | **0.973** | CIFAKE validation |
| 📈 ROC-AUC | **0.997** | CIFAKE validation |
| 📈 ROC-AUC — unseen-generator split | ⏳ pending organizer evaluation | Held-out test |
| 🔲 Confusion matrix | ⏳ not yet generated | — |

### ⚠️ Known gaps to close before final judging

Tracked honestly rather than glossed over:

- The bundled checkpoint (`signalscope_cifake_best.pth`) predates the threshold
  calibration step in `train_cifake.py` and has no stored `ai_decision_threshold`.
  `inference.py` detects this and falls back to a default threshold (0.55), flagging
  `threshold_calibrated: false` in both the API and CLI output. **Re-run
  `train_cifake.py`** to ship a fully calibrated checkpoint.
- No confusion matrix is currently persisted with the checkpoint; add it to the
  `ckpt = {...}` dict in `train_cifake.py`, or regenerate it against the CIFAKE test
  split, before submission.
- The metrics above are self-reported on the CIFAKE validation split, not an
  independent held-out set — the organizers' unseen-generator AUC (the primary
  ranking metric per the problem statement) has not yet been measured.

---

## 🧪 Training & Robustness Recipe

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 5e-4, cosine schedule |
| Epochs | 5 |
| Batch size | 64 |
| Label smoothing | 0.05 |
| Precision | Mixed (FP16) |

**Augmentations for unseen-generator / degraded-image robustness:** JPEG compression
(Q 55–100), downscale + upsample (50–90%), Gaussian blur, Gaussian noise, colour
jitter.

> `signalscope/model/architecture.py` also contains an experimental dual-stream
> network (EfficientNet-B1 + a learned FFT stream) that is **not** what's deployed —
> it's kept for future work. The production path is the single-stream
> MobileNetV3-Small described above.

---

## 🧯 Known Limitations

| # | Limitation |
|---|---|
| 1 | Extremely high-quality outputs (recent FLUX / SDXL-class images) approaching photorealism may fool the model. |
| 2 | Heavily processed real photos (HDR, artistic filters, extreme edits) can be misclassified as AI-generated. |
| 3 | Trained on photographic content — text-heavy images/screenshots may produce uncertain results. |
| 4 | Very low-resolution images (below ~64×64) lack sufficient spectral/spatial information. |
| 5 | Metrics above are on CIFAKE validation, not the organizers' unseen-generator held-out set — true generalisation is not yet confirmed. |

📄 Full model card: [`signalscope/report/model_report.md`](signalscope/report/model_report.md)

---

## 🎥 Demo Video & Deployed App

| | |
|---|---|
| **Demo video (3–5 min)** | https://youtu.be/eCPbT5pSFp8 |
| **Deployed app** | Not deployed — run locally per the Setup instructions above |

---

## 📁 Repository Structure

```
├── app.py                              # FastAPI app — backend + embedded frontend (bonus F)
├── requirements.txt / requirements-gpu.txt
├── signalscope/
│   ├── model/
│   │   ├── architecture.py             # experimental dual-stream net (not deployed)
│   │   ├── train_cifake.py             # training entry point
│   │   ├── inference.py                # predict / predict_with_details
│   │   ├── predict.py                  # required CLI predict interface
│   │   └── weights/                    # checkpoints (git-ignored — see Setup)
│   ├── src/
│   │   ├── forensic_explainer.py       # Bonus A — Grad-CAM + cue report
│   │   ├── attribution.py              # Bonus B — generator attribution
│   │   ├── robustness.py               # Bonus C — degradation sweep
│   │   └── provenance.py               # Bonus D — EXIF/metadata fusion
│   └── report/model_report.md          # one-page model report
├── test_samples/                       # sample real/AI images for quick testing
└── data/cifake/                        # cached CIFAKE parquet files (training only)
```

## ✍️ Originality Declaration

Built on top of the open-source **CIFAKE** dataset (Hugging Face Hub) and the
**MobileNetV3-Small** backbone (`torchvision`/`timm`, pretrained on ImageNet). No
third-party real-vs-fake notebook was copied; explainability, attribution, robustness,
and provenance modules were written for this submission. AI coding assistance was
used during development, per the challenge rules.

## 🛡️ Scope & Ethics

This project detects AI-generated *imagery in general* (scenes, objects, art, product
shots) — **not** face-swap deepfakes of real, identifiable people, and **not**
political claims or real-world events. All verdicts are presented as likelihood
assessments, never accusations.
