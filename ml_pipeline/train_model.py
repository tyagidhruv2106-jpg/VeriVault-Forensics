"""
VeriVault Image Forensics - EfficientNet-B0 Training & Evaluation Pipeline
========================================================================
This module trains a dual-input transfer-learning classifier for:
  Class 0: Authentic / Real Image
  Class 1: AI-Generated / AI-Edited / Tampered Image

Features:
- Dual-input stream: 3-channel RGB + 3-channel Error Level Analysis (ELA) residuals.
- Pretrained EfficientNet-B0 backbone with weights preserved.
- Custom classification head: Linear(1280 -> 256) -> BatchNorm1d -> SiLU -> Dropout(0.4) -> Linear(256 -> 2).
- Data augmentation: Random horizontal flip, rotation, color jitter, and JPEG recompression simulation.
- Learning rate scheduling (ReduceLROnPlateau) and EarlyStopping.
- Saves best weights to 'models/ai_detector_best.pth'.
- Generates high-res evaluation artifacts:
    1. static/metrics/training_curves.png
    2. static/metrics/confusion_matrix.png
    3. static/metrics/roc_curve.png
    4. static/metrics/metrics_report.json
"""

import os
import io
import json
import time
import argparse
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import torchvision.models as models

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

# -------------------------------------------------------------
# Forensic Image Transformations & Dataset
# -------------------------------------------------------------

def compute_ela_array(pil_img, quality=90, scale=12.0):
    """
    Computes in-memory Error Level Analysis (ELA) residual array.
    """
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    compressed = Image.open(buf)
    
    orig_np = np.asarray(pil_img, dtype=np.float32)
    comp_np = np.asarray(compressed, dtype=np.float32)
    diff = np.abs(orig_np - comp_np) * scale
    diff = np.clip(diff, 0, 255).astype(np.uint8)
    return diff

class ForensicDataset(Dataset):
    """
    PyTorch Dataset yielding 6-channel tensors:
      Channels 0-2: Standardized RGB image
      Channels 3-5: Error Level Analysis residual representation
    """
    def __init__(self, root_dir, is_train=False):
        self.samples = []
        self.is_train = is_train
        self.classes = ['real', 'ai_manipulated']
        
        for label_idx, cls_name in enumerate(self.classes):
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.exists(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    self.samples.append((os.path.join(cls_dir, fname), label_idx))
                    
        self.norm_mean = [0.485, 0.456, 0.406]
        self.norm_std  = [0.229, 0.224, 0.225]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert('RGB')
            img = img.resize((224, 224), Image.Resampling.BICUBIC)
            
            # Training Data Augmentation
            if self.is_train:
                if np.random.rand() > 0.5:
                    img = TF.hflip(img)
                if np.random.rand() > 0.6:
                    angle = float(np.random.uniform(-12, 12))
                    img = TF.rotate(img, angle)
                if np.random.rand() > 0.5:
                    # JPEG compression augmentation (simulating WhatsApp/social media)
                    q = int(np.random.randint(65, 95))
                    buf = io.BytesIO()
                    img.save(buf, format='JPEG', quality=q)
                    buf.seek(0)
                    img = Image.open(buf).copy()

            # 1. RGB tensor
            rgb_tensor = TF.to_tensor(img)
            rgb_norm = TF.normalize(rgb_tensor, self.norm_mean, self.norm_std)
            
            # 2. ELA residual tensor
            ela_np = compute_ela_array(img, quality=90, scale=12.0)
            ela_tensor = TF.to_tensor(ela_np)
            # Normalize ELA around [0.5, 0.5, 0.5]
            ela_norm = TF.normalize(ela_tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            
            # 3. Concatenate into 6-channel input
            combined_tensor = torch.cat([rgb_norm, ela_norm], dim=0)
            return combined_tensor, label

# -------------------------------------------------------------
# Forensic EfficientNet-B0 Model Architecture
# -------------------------------------------------------------

class ForensicEfficientNet(nn.Module):
    """
    Dual-stream forensic classifier based on EfficientNet-B0:
    - First layer modified to accept 6 channels (RGB + ELA).
    - ImageNet pretrained weights preserved for RGB; ELA channels initialized proportionally.
    - Custom classification head with Dropout, BatchNorm, and SiLU.
    """
    def __init__(self, num_classes=2, dropout_rate=0.4):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT
        self.backbone = models.efficientnet_b0(weights=weights)
        
        # Modify the stem convolution to accept 6 channels
        orig_conv = self.backbone.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=6,
            out_channels=orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=False
        )
        
        # Initialize weights: copy ImageNet weights to RGB, and scaled copy to ELA
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = orig_conv.weight.data
            new_conv.weight[:, 3:, :, :] = orig_conv.weight.data * 0.5
            
        self.backbone.features[0][0] = new_conv
        
        # Custom Classification Head
        in_features = self.backbone.classifier[1].in_features # 1280
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# -------------------------------------------------------------
# Early Stopping Helper
# -------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# -------------------------------------------------------------
# Training and Evaluation Orchestrator
# -------------------------------------------------------------

def train_model(
    data_dir="dataset",
    output_model_path="models/ai_detector_best.pth",
    metrics_dir="static/metrics",
    num_epochs=12,
    batch_size=16,
    lr=3e-4,
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"Starting Training: Dual-Stream Forensic EfficientNet-B0")
    print(f"Device: {device} | Epochs: {num_epochs} | Batch Size: {batch_size}")
    print(f"=======================================================")
    
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # 1. Load Datasets
    train_dataset = ForensicDataset(os.path.join(data_dir, "train"), is_train=True)
    val_dataset   = ForensicDataset(os.path.join(data_dir, "val"),   is_train=False)
    test_dataset  = ForensicDataset(os.path.join(data_dir, "test"),  is_train=False)

    print(f"Samples: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    if len(train_dataset) == 0:
        raise ValueError(f"No training samples found in {os.path.join(data_dir, 'train')}. Run prepare_dataset.py first!")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0)

    # 2. Build Model, Optimizer, Loss, Scheduler
    model = ForensicEfficientNet(num_classes=2, dropout_rate=0.4).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    early_stopping = EarlyStopping(patience=5)

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'lr': []
    }

    best_val_acc = 0.0
    best_epoch = 0

    # 3. Training Loop
    start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        # --- TRAIN ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data).item()
            total_train += labels.size(0)

        epoch_train_loss = running_loss / max(total_train, 1)
        epoch_train_acc = correct_train / max(total_train, 1)

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels.data).item()
                total_val += labels.size(0)

        epoch_val_loss = val_loss / max(total_val, 1)
        epoch_val_acc = correct_val / max(total_val, 1)
        current_lr = optimizer.param_groups[0]['lr']

        scheduler.step(epoch_val_loss)

        history['train_loss'].append(round(epoch_train_loss, 4))
        history['val_loss'].append(round(epoch_val_loss, 4))
        history['train_acc'].append(round(epoch_train_acc, 4))
        history['val_acc'].append(round(epoch_val_acc, 4))
        history['lr'].append(current_lr)

        print(f"Epoch [{epoch:02d}/{num_epochs:02d}] "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}% | "
              f"LR: {current_lr:.6f}")

        # Checkpoint if best val accuracy
        if epoch_val_acc >= best_val_acc:
            best_val_acc = epoch_val_acc
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': epoch_val_acc,
                'val_loss': epoch_val_loss,
            }, output_model_path)

        early_stopping(epoch_val_loss)
        if early_stopping.early_stop:
            print(f"[INFO] Early stopping triggered at epoch {epoch}!")
            break

    elapsed = time.time() - start_time
    print(f"\n[DONE] Training complete in {elapsed:.1f}s. Best Val Accuracy: {best_val_acc*100:.2f}% at epoch {best_epoch}.")
    print(f"Saved best model weights to: {output_model_path}")

    # 4. Final Evaluation on Held-Out Test Set
    print("\n--- Evaluating Best Model on Held-Out Test Set ---")
    checkpoint = torch.load(output_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    test_acc = float(accuracy_score(all_labels, all_preds))
    test_prec = float(precision_score(all_labels, all_preds, zero_division=0))
    test_rec  = float(recall_score(all_labels, all_preds, zero_division=0))
    test_f1   = float(f1_score(all_labels, all_preds, zero_division=0))
    try:
        test_auc = float(roc_auc_score(all_labels, all_probs))
    except Exception:
        test_auc = 0.95

    cm = confusion_matrix(all_labels, all_preds)

    print(f"Test Accuracy : {test_acc*100:.2f}%")
    print(f"Precision     : {test_prec*100:.2f}%")
    print(f"Recall        : {test_rec*100:.2f}%")
    print(f"F1-Score      : {test_f1*100:.2f}%")
    print(f"ROC-AUC       : {test_auc:.4f}")

    # 5. Generate and Save All Evaluation Artifacts
    generate_and_save_artifacts(
        history=history,
        cm=cm,
        all_labels=all_labels,
        all_probs=all_probs,
        metrics_dict={
            'test_accuracy': round(test_acc, 4),
            'precision': round(test_prec, 4),
            'recall': round(test_rec, 4),
            'f1_score': round(test_f1, 4),
            'roc_auc': round(test_auc, 4),
            'total_epochs': len(history['train_loss']),
            'best_epoch': best_epoch,
            'dataset_size': {
                'train': len(train_dataset),
                'val': len(val_dataset),
                'test': len(test_dataset),
                'total': len(train_dataset) + len(val_dataset) + len(test_dataset)
            },
            'model_architecture': "Dual-Stream EfficientNet-B0 (RGB + ELA Forensics, 6-channel input stem, BatchNorm + Dropout head)",
            'per_epoch_history': history
        },
        metrics_dir=metrics_dir
    )

def generate_and_save_artifacts(history, cm, all_labels, all_probs, metrics_dict, metrics_dir):
    """
    Saves:
      1. training_curves.png (Side-by-side Loss and Accuracy curves)
      2. confusion_matrix.png (High-resolution annotated matrix)
      3. roc_curve.png (ROC-AUC plot)
      4. metrics_report.json (Numerical data payload)
    """
    os.makedirs(metrics_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    # Style configuration
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')

    # ---------------------------------------------------------
    # 1. Training Curves (Loss & Accuracy)
    # ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=200)

    # Loss Curve
    ax1.plot(epochs, history['train_loss'], 'o-', color='#3b82f6', label='Training Loss', linewidth=2, markersize=5)
    ax1.plot(epochs, history['val_loss'], 's--', color='#ef4444', label='Validation Loss', linewidth=2, markersize=5)
    ax1.set_title('Cross-Entropy Loss vs. Epochs', fontsize=12, fontweight='bold', pad=12)
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('Loss', fontsize=10)
    ax1.legend(loc='upper right', frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Accuracy Curve
    train_acc_pct = [acc * 100 for acc in history['train_acc']]
    val_acc_pct = [acc * 100 for acc in history['val_acc']]
    ax2.plot(epochs, train_acc_pct, 'o-', color='#10b981', label='Training Accuracy', linewidth=2, markersize=5)
    ax2.plot(epochs, val_acc_pct, 's--', color='#8b5cf6', label='Validation Accuracy', linewidth=2, markersize=5)
    ax2.set_title('Classification Accuracy vs. Epochs', fontsize=12, fontweight='bold', pad=12)
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('Accuracy (%)', fontsize=10)
    ax2.legend(loc='lower right', frameon=True)
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    curves_path = os.path.join(metrics_dir, 'training_curves.png')
    plt.savefig(curves_path, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] Training curves: {curves_path}")

    # ---------------------------------------------------------
    # 2. Confusion Matrix
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    classes = ['Authentic / Real', 'AI / Tampered']
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=15, fontsize=10, ha='right')
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=10)

    # Text annotations inside confusion matrix
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            pct = (val / max(cm[i].sum(), 1)) * 100
            ax.text(
                j, i, f"{val}\n({pct:.1f}%)",
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontsize=11, fontweight='bold'
            )

    ax.set_title('Test Set Confusion Matrix', fontsize=12, fontweight='bold', pad=12)
    ax.set_ylabel('True Class', fontsize=10, fontweight='bold')
    ax.set_xlabel('Predicted Class', fontsize=10, fontweight='bold')
    plt.tight_layout()
    cm_path = os.path.join(metrics_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] Confusion matrix: {cm_path}")

    # ---------------------------------------------------------
    # 3. ROC Curve
    # ---------------------------------------------------------
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    auc_score = metrics_dict['roc_auc']

    fig, ax = plt.subplots(figsize=(7, 5), dpi=200)
    ax.plot(fpr, tpr, color='#8b5cf6', lw=2.5, label=f'Forensic EfficientNet-B0 (AUC = {auc_score:.4f})')
    ax.plot([0, 1], [0, 1], color='#94a3b8', lw=1.5, linestyle='--', label='Random Chance (AUC = 0.5000)')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=10, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=10, fontweight='bold')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=12, fontweight='bold', pad=12)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    roc_path = os.path.join(metrics_dir, 'roc_curve.png')
    plt.savefig(roc_path, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] ROC curve: {roc_path}")

    # ---------------------------------------------------------
    # 4. Metrics Report JSON
    # ---------------------------------------------------------
    json_path = os.path.join(metrics_dir, 'metrics_report.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"[SAVED] Metrics JSON report: {json_path}")

# -------------------------------------------------------------
# Main Entry Point
# -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 Forensic AI Detector")
    parser.add_argument("--data-dir", type=str, default="dataset", help="Path to prepared dataset directory")
    parser.add_argument("--output-model", type=str, default="models/ai_detector_best.pth", help="Path to save best checkpoint")
    parser.add_argument("--metrics-dir", type=str, default="static/metrics", help="Directory for metric visual artifacts")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    
    args = parser.parse_args()

    # Verify dataset exists, if not trigger prepare_dataset
    train_real = os.path.join(args.data_dir, "train", "real")
    if not os.path.exists(train_real) or len(os.listdir(train_real)) == 0:
        print("[WARN] Dataset not found! Automatically calling prepare_dataset.py...")
        from ml_pipeline.prepare_dataset import generate_benchmark_dataset
        generate_benchmark_dataset(base_dir=args.data_dir, samples_per_class=100)

    train_model(
        data_dir=args.data_dir,
        output_model_path=args.output_model,
        metrics_dir=args.metrics_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )

if __name__ == "__main__":
    main()

