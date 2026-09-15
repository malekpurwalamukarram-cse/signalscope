"""
SignalScope - Model Architecture (PS-2)
Dual-stream spatial + frequency-domain classifier for real vs. AI-generated image detection.

Stream 1: Spatial CNN (EfficientNet-B1) — high-level semantic and texture features.
Stream 2: Frequency DCT/FFT analysis — low-level spectral artifacts from generative models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


CLASS_NAMES  = ["AI-Generated", "Real"]
NUM_CLASSES  = 2
LABEL_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


# ── Frequency stream: DCT-based spectral analysis ────────────────────────────
class FrequencyStream(nn.Module):
    """
    Extracts spectral artifacts from the FFT magnitude spectrum.
    Diffusion and GAN models leave characteristic frequency-domain fingerprints.
    """

    def __init__(self, out_features: int = 256):
        super().__init__()
        # Process frequency spectrum as an image: 3-channel (R, G, B FFT magnitude)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(4),
        )
        self.fc = nn.Linear(128 * 4 * 4, out_features)

    def forward(self, freq_tensor: torch.Tensor) -> torch.Tensor:
        x = self.conv(freq_tensor)
        x = x.flatten(1)
        return self.fc(x)


def compute_fft_spectrum(x: torch.Tensor) -> torch.Tensor:
    """
    Compute log-magnitude FFT spectrum per channel.
    Input: [B, 3, H, W] float tensor (normalised).
    Output: [B, 3, H, W] float tensor — shifted log-magnitude spectrum.
    """
    # Work per channel
    x_fft = torch.fft.fft2(x)
    x_fft = torch.fft.fftshift(x_fft, dim=(-2, -1))
    magnitude = torch.abs(x_fft)
    log_mag = torch.log(magnitude + 1e-8)
    # Normalize per sample
    mn = log_mag.flatten(2).min(dim=2)[0][..., None, None]
    mx = log_mag.flatten(2).max(dim=2)[0][..., None, None]
    log_mag = (log_mag - mn) / (mx - mn + 1e-8)
    return log_mag


# ── Spatial stream: EfficientNet-B1 ──────────────────────────────────────────
class SpatialStream(nn.Module):
    """EfficientNet-B1 backbone for high-level texture/semantic feature extraction."""

    def __init__(self, out_features: int = 512, pretrained: bool = True):
        super().__init__()
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required. Install: pip install timm")
        self.backbone = timm.create_model(
            "efficientnet_b1", pretrained=pretrained, num_classes=0
        )
        in_features = self.backbone.num_features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.GELU(),
            nn.Dropout(0.3),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat_map = self.backbone.forward_features(x)   # [B, C, H, W]
        pooled   = self.pool(feat_map).flatten(1)       # [B, C]
        proj     = self.proj(pooled)                    # [B, out_features]
        return proj, feat_map


# ── Fusion classifier ─────────────────────────────────────────────────────────
class SignalScopeModel(nn.Module):
    """
    Dual-stream Real vs. AI-Generated image detector.
    Spatial + Frequency streams fused via learned attention weighting.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True,
                 dropout: float = 0.4):
        super().__init__()
        self.spatial_stream    = SpatialStream(out_features=512, pretrained=pretrained)
        self.frequency_stream  = FrequencyStream(out_features=256)

        # Fusion
        fused_dim = 512 + 256  # 768
        self.fusion = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        # Stream attention (how much to weight each stream)
        self.stream_attention = nn.Sequential(
            nn.Linear(fused_dim, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial features
        spatial_feat, feat_map = self.spatial_stream(x)

        # Frequency features (compute FFT inside forward)
        freq_spectrum = compute_fft_spectrum(x)
        freq_feat = self.frequency_stream(freq_spectrum)

        # Concatenate and fuse
        fused  = torch.cat([spatial_feat, freq_feat], dim=1)
        logits = self.fusion(fused)
        return logits

    def get_feature_maps(self, x: torch.Tensor):
        """Return (logits, spatial_feat_map) for Grad-CAM."""
        spatial_feat, feat_map = self.spatial_stream(x)
        freq_spectrum = compute_fft_spectrum(x)
        freq_feat = self.frequency_stream(freq_spectrum)
        fused  = torch.cat([spatial_feat, freq_feat], dim=1)
        logits = self.fusion(fused)
        return logits, feat_map


def build_model(num_classes: int = NUM_CLASSES,
                pretrained: bool = True,
                checkpoint: str | None = None,
                device: str = "auto") -> SignalScopeModel:
    """Build and optionally load weights into SignalScopeModel."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SignalScopeModel(num_classes=num_classes, pretrained=pretrained)
    model = model.to(device)

    if checkpoint is not None:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        state_dict = state.get("model_state_dict", state)
        model.load_state_dict(state_dict, strict=False)
        print(f"[SignalScope] Loaded weights from {checkpoint}")

    return model


if __name__ == "__main__":
    model = build_model(pretrained=False)
    dummy = torch.zeros(2, 3, 224, 224)
    out   = model(dummy)
    print(f"Output shape: {out.shape}  — {NUM_CLASSES} classes: {CLASS_NAMES}")

