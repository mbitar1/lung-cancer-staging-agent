import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# --- Import shared data pipeline from teammate's script ---
# This will:
#  - read lung_petct_dx.csv
#  - read statistics-clinical-20201221.xlsx
#  - merge them
#  - compute label_bin (early/advanced)
#  - build feature_cols, df_merged
#  - create train_df, val_df, test_df
#  - define LungCTDataset, N_SLICES, IMG_SIZE, device, classes
from lung_ct_classifier_final import (
    df_merged,
    feature_cols,
    N_SLICES,
    IMG_SIZE,
    device,
    classes,
    LungCTDataset,
    train_df,
    val_df,
    test_df,
)

# Number of tabular features (Age_z, Weight_z, Sex_bin, Smoking_bin)
num_tab_features = len(feature_cols)

print("UNet classifier will use the SAME data splits as the CNN.")
print(f"Train/Val/Test sizes: {len(train_df)}, {len(val_df)}, {len(test_df)}")
print("Tabular features:", feature_cols)
print("Device:", device)

# ---------- DataLoaders (same splits, same dataset class) ----------

BATCH_SIZE = 16

train_ds = LungCTDataset(train_df, feature_cols, n_slices=N_SLICES,
                         train=True, augment=True)
val_ds   = LungCTDataset(val_df,   feature_cols, n_slices=N_SLICES,
                         train=False, augment=False)
test_ds  = LungCTDataset(test_df,  feature_cols, n_slices=N_SLICES,
                         train=False, augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ---------- UNet building blocks ----------

class DoubleConv(nn.Module):
    """
    Two Conv2d + BatchNorm + ReLU blocks (standard UNet building block).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetEncoder2D(nn.Module):
    """
    A small 2D UNet (encoder-decoder with skips).
    Instead of outputting a segmentation map, we will use
    its final feature map as an image embedding.
    """

    def __init__(self, in_channels, base_channels=32):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(in_channels,       base_channels)       # C = 32
        self.enc2 = DoubleConv(base_channels,     base_channels * 2)   # 64
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)   # 128
        self.enc4 = DoubleConv(base_channels * 4, base_channels * 8)   # 256

        # Max pooling
        self.pool = nn.MaxPool2d(2)

        # Decoder
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4,
                                      kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2,
                                      kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels,
                                      kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)

        # Final conv to keep base_channels channels (we don't reduce to 1)
        self.out_conv = nn.Conv2d(base_channels, base_channels, kernel_size=1)

        self.base_channels = base_channels

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)          # (B, 32, H, W)
        p1 = self.pool(e1)         # (B, 32, H/2, W/2)

        e2 = self.enc2(p1)         # (B, 64, H/2, W/2)
        p2 = self.pool(e2)         # (B, 64, H/4, W/4)

        e3 = self.enc3(p2)         # (B, 128, H/4, W/4)
        p3 = self.pool(e3)         # (B, 128, H/8, W/8)

        e4 = self.enc4(p3)         # (B, 256, H/8, W/8)

        # Decoder with skip connections
        u3 = self.up3(e4)          # (B, 128, H/4, W/4)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))  # (B, 128, H/4, W/4)

        u2 = self.up2(d3)          # (B, 64, H/2, W/2)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))  # (B, 64, H/2, W/2)

        u1 = self.up1(d2)          # (B, 32, H, W)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))  # (B, 32, H, W)

        out = self.out_conv(d1)    # (B, 32, H, W)

        # This is our "feature map" for the image encoder
        return out  # shape: (B, base_channels, H, W)


# ---------- Multi-modal UNet-based classifier ----------

class LungCT_UNet_MultiModal(nn.Module):
    """
    Image branch: UNetEncoder2D (N_SLICES as channels).
    Tabular branch: MLP on clinical features.
    Final: concatenated features -> MLP -> 2-class logits.
    """
    def __init__(self, num_tab_features, n_slices=N_SLICES, base_channels=32):
        super().__init__()

        # UNet image encoder
        self.unet = UNetEncoder2D(in_channels=n_slices,
                                  base_channels=base_channels)

        # After UNet, feature map is (B, base_channels, H, W).
        # We will do global average pooling: (B, C, H, W) -> (B, C)
        self.fc_img = nn.Linear(base_channels, 256)
        self.dropout_img = nn.Dropout(p=0.3)

        # Tabular feature branch
        self.fc_tab = nn.Linear(num_tab_features, 64)

        # Combined branch
        self.fc_comb = nn.Linear(256 + 64, 128)
        self.dropout_comb = nn.Dropout(p=0.4)

        # Output: 2 classes (early / advanced)
        self.fc_out = nn.Linear(128, 2)

    def forward(self, x_img, x_tab):
        # x_img shape: (B, N_SLICES, IMG_SIZE, IMG_SIZE)

        feat_map = self.unet(x_img)           # (B, base_channels, H, W)

        # Global average pooling over H, W
        feat_vec = feat_map.mean(dim=(2, 3))  # (B, base_channels)

        # Image branch fully-connected
        x = F.relu(self.fc_img(feat_vec))
        x = self.dropout_img(x)

        # Tabular branch
        t = F.relu(self.fc_tab(x_tab))

        # Combine
        z = torch.cat([x, t], dim=1)
        z = F.relu(self.fc_comb(z))
        z = self.dropout_comb(z)

        logits = self.fc_out(z)               # (B, 2)
        return logits


# ---------- Instantiate model, loss, optimizer, scheduler ----------

torch.manual_seed(42)

model = LungCT_UNet_MultiModal(
    num_tab_features=num_tab_features,
    n_slices=N_SLICES,
    base_channels=32
).to(device)

print("\nUNet-based multimodal model:")
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=1e-3,
                             weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.5, patience=3
)


# ---------- Evaluation utility ----------

@torch.no_grad()
def evaluate(loader):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0

    for x_img, x_tab, y in loader:
        x_img, x_tab, y = x_img.to(device), x_tab.to(device), y.to(device)

        logits = model(x_img, x_tab)
        loss = criterion(logits, y)

        total_loss += loss.item() * x_img.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total += x_img.size(0)

    return total_loss / total, total_correct / total


# ---------- Training loop ----------

def main():
    epochs = 30 

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0

        for x_img, x_tab, y in train_loader:
            x_img, x_tab, y = x_img.to(device), x_tab.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x_img, x_tab)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_img.size(0)
            preds = logits.argmax(dim=1)
            running_correct += (preds == y).sum().item()
            running_total += x_img.size(0)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        val_loss, val_acc = evaluate(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:2d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    # --- Plots: loss ---
    plt.figure(figsize=(6, 6))
    epochs_axis = np.arange(1, epochs + 1)
    plt.plot(epochs_axis, train_losses, label="Training Loss")
    plt.plot(epochs_axis, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"UNet: Train vs Val Loss ({N_SLICES}-slice, multimodal)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Plots: accuracy ---
    plt.figure(figsize=(6, 6))
    plt.plot(epochs_axis, train_accs, label="Training Accuracy")
    plt.plot(epochs_axis, val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)
    plt.title(f"UNet: Train vs Val Accuracy ({N_SLICES}-slice, multimodal)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Final test evaluation ---
    test_loss, test_acc = evaluate(test_loader)
    print(f"\n[UNet] Final Test Loss and Accuracy | loss={test_loss:.4f} acc={test_acc:.4f}")

    # Simple bar plot for test accuracy
    plt.figure(figsize=(4, 5))
    plt.bar(["UNet Test Accuracy"], [test_acc])
    plt.ylim(0.0, 1.0)
    plt.ylabel("Accuracy")
    plt.title("UNet Final Test Accuracy")
    plt.tight_layout()
    plt.show()


# ---------- Prediction + confusion matrix ----------

@torch.no_grad()
def get_preds_and_labels(loader, model, device):
    model.eval()
    all_preds, all_labels = [], []

    for x_img, x_tab, y in loader:
        x_img, x_tab, y = x_img.to(device), x_tab.to(device), y.to(device)

        logits = model(x_img, x_tab)
        preds = logits.argmax(dim=1)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.cpu().numpy())

    return np.concatenate(all_preds), np.concatenate(all_labels)


def show_confusion_matrix(preds, labels, class_names, title):
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    plt.figure(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", ax=plt.gca(), xticks_rotation=45)
    plt.title(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

    test_preds, test_labels = get_preds_and_labels(test_loader, model, device)
    test_acc = (test_preds == test_labels).mean()
    print(f"[UNet] Test accuracy (recomputed): {test_acc:.4f}")

    show_confusion_matrix(
        test_preds,
        test_labels,
        classes,
        f"Early vs Advanced Confusion Matrix (UNet multimodal, {N_SLICES}-slice)",
    )
