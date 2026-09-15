# signalscope/model/train.py
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

class DirectForensicDataset(Dataset):
    """Direct Memory Loader: Recursively discovers CIFAKE image directories."""
    def __init__(self, base_dir, transform=None):
        self.samples = []
        self.transform = transform
        valid_exts = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff')

        print(f"🔍 Scanning directory path: {base_dir}")
        
        # 1. Self-correcting path crawler to find AI vs Real groups
        for root, dirs, files in os.walk(base_dir):
            for f in files:
                if f.lower().endswith(valid_exts):
                    file_path = os.path.join(root, f)
                    path_lower = root.lower()
                    
                    # Target CIFAKE subfolder names ('FAKE' / 'REAL') natively
                    if "fake" in path_lower or "ai" in path_lower:
                        self.samples.append((file_path, 0)) # Class 0: AI
                    elif "real" in path_lower:
                        self.samples.append((file_path, 1)) # Class 1: Real

        if len(self.samples) == 0:
            raise FileNotFoundError(f"❌ Loader Error: No valid image assets discovered inside '{base_dir}'.")
            
        ai_count = sum(1 for _, l in self.samples if l == 0)
        real_count = sum(1 for _, l in self.samples if l == 1)
        print(f"🎯 Successfully bound {len(self.samples)} CIFAKE images! (AI: {ai_count} | Real: {real_count})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception:
            return torch.zeros(3, 384, 384), label

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Launching Optimized RTX 3050 Accelerator on: {device}")

    # Root location where your CIFAKE 105MB dataset unzipped
    base_data_dir = "data/real_world"

    # Production transforms configured to rescale CIFAKE up to your 384x384 inference target
    train_transform = T.Compose([
        T.Resize((384, 384)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        full_dataset = DirectForensicDataset(base_data_dir, transform=train_transform)
    except Exception as e:
        print(str(e))
        sys.exit(1)

    # Perform a 90/10 operational split in system memory
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, _ = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # Saturated batch processing for your GPU VRAM limits
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    
    for param in model.parameters():
        param.requires_grad = False

    if hasattr(model, 'features') and len(model.features) > 0:
        for param in model.features[-1].parameters():
            param.requires_grad = True

    num_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-2)
    
    scaler = torch.amp.GradScaler('cuda')

    epochs = 3
    print("🏁 Starting GPU Core Training Optimization...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=True)
        
        for images, labels in loop:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * images.size(0)
            loop.set_postfix(loss=loss.item())
            
        print(f"📈 Epoch {epoch+1} Complete | Loss: {running_loss/len(train_dataset):.4f}")

    weight_path = "signalscope/model/weights/signalscope_production_384px.pth"
    os.makedirs(os.path.dirname(weight_path), exist_ok=True)
    torch.save(model, weight_path)
    print(f"🎉 Training complete! Calibrated weights secured at: {weight_path}")

if __name__ == "__main__":
    train_model()
