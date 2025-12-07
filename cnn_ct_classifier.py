import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import nibabel as nib
import pandas as pd
from pathlib import Path

import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


# ---------- Helper: robust NIfTI loader ----------

def load_nifti_volume_as_float(ct_path: str) -> np.ndarray:

    """
    Load a NIfTI volume and return a float32 numpy array.

    - If the data is structured RGB (dtype with fields 'R','G','B'),
      convert to grayscale.
    - Otherwise, cast directly to float32.
    """
    nii = nib.load(ct_path)
    arr = np.asanyarray(nii.dataobj)

    # Structured dtype (e.g. RGB)
    if arr.dtype.fields is not None:
        if all(ch in arr.dtype.fields for ch in ("R", "G", "B")):
            r = arr["R"].astype(np.float32)
            g = arr["G"].astype(np.float32)
            b = arr["B"].astype(np.float32)
            vol = (r + g + b) / 3.0  # simple grayscale
        else:
            raise TypeError(f"Unsupported structured dtype for {ct_path}: {arr.dtype}")
    else:
        vol = arr.astype(np.float32)

    return vol

# ----- Repro / device -----
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image resolution fed to CNN
IMG_SIZE = 64  # bigger than 32 to keep more info

# Number of axial slices to stack as channels
N_SLICES = 5   # <-- can change to 5 or 11 to experiment later

# ----- Paths -----

BASE_DIR = Path(__file__).resolve().parent

CSV_PATH = BASE_DIR / "lung_petct_dx.csv"
CLINICAL_XLSX_PATH = BASE_DIR / "Data" / "statistics-clinical-20201221.xlsx"


# ---------- Load and merge CSV + clinical Excel ----------

# ----- DATA CHECKER: CT CSV -----
df_ct_raw = pd.read_csv(CSV_PATH)
print(f"Rows in raw CSV: {len(df_ct_raw)}")

assert "subject_id" in df_ct_raw.columns, "CSV must contain 'subject_id'"
assert "ct_path" in df_ct_raw.columns, "CSV must contain 'ct_path'"

df_ct = df_ct_raw.copy()

# Extract case ID like A0001, B0001, etc.
df_ct["case_id"] = df_ct["subject_id"].str.extract(r"([A-Z]\d+)", expand=False)

print(f"Rows with non-null case_id after regex: {df_ct['case_id'].notna().sum()}")

print(f"Rows with NULL case_id after regex: {df_ct['case_id'].isna().sum()}")


# Debug CT-path completeness BEFORE filtering
num_rows_total = len(df_ct)
num_ct_missing = df_ct["ct_path"].isna().sum()
num_ct_empty   = (df_ct["ct_path"] == "").sum()
print(f"Rows total (after adding case_id): {num_rows_total}")
print(f"Rows with ct_path NaN: {num_ct_missing}")
print(f"Rows with ct_path empty string: {num_ct_empty}")

# Drop rows with missing or empty CT paths
df_ct = df_ct[df_ct["ct_path"].notna() & (df_ct["ct_path"] != "")]
df_ct = df_ct.reset_index(drop=True)
print(f"Rows remaining after ct_path filter: {len(df_ct)}")
print(f"Unique case_id in CT CSV (after ct_path filter): {df_ct['case_id'].nunique()}")

# Clinical data
df_clin = pd.read_excel(CLINICAL_XLSX_PATH)

df_clin.columns = [str(c).strip() for c in df_clin.columns]

print("Raw Excel columns:", list(df_clin.columns))  # DEBUG

# Helper to normalize names (lowercase, remove spaces/dashes/underscores)
def _norm(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())

# Helper that finds a column by a few candidate names and renames it
def find_and_rename(df, candidates, new_name):
    norm_map = {_norm(c): c for c in df.columns}
    for cand in candidates:
        key = _norm(cand)
        if key in norm_map:
            old = norm_map[key]
            if old != new_name:
                df.rename(columns={old: new_name}, inplace=True)
            print(f"Mapped Excel column '{old}' -> '{new_name}'")
            return
    raise KeyError(
        f"Could not find column for '{new_name}'. "
        f"Tried candidates: {candidates}. "
        f"Available columns: {list(df.columns)}"
    )

# Map specific columns
find_and_rename(df_clin, ["NewPatientID"], "case_id")
find_and_rename(df_clin, ["T-Stage"], "T_Stage")
find_and_rename(df_clin, ["N-Stage"], "N_Stage")
find_and_rename(df_clin, ["M-Stage", "Ｍ-Stage"], "M_Stage")
find_and_rename(df_clin, ["Histopathological grading"], "Histopathological_grading")

required_cols = ["case_id", "T_Stage", "N_Stage", "M_Stage", "Histopathological_grading"]
missing = [c for c in required_cols if c not in df_clin.columns]
assert not missing, f"Missing columns in clinical Excel (after rename): {missing}"

print(f"Clinical Excel rows: {len(df_clin)}")
print(f"Unique case_id in clinical Excel: {df_clin['case_id'].nunique()}")


# ----- DATA CHECKER: Merge characteristics -----
df_merge_debug = pd.merge(
    df_ct[["case_id"]],
    df_clin[["case_id"]],
    on="case_id",
    how="outer",
    indicator=True
)

print("\nMerge indicator counts (how many case_ids in each group):")
print(df_merge_debug["_merge"].value_counts())

print("Example case_ids only in CT CSV (left_only):",
      df_merge_debug.loc[df_merge_debug["_merge"] == "left_only", "case_id"].head().tolist())

print("Example case_ids only in Clinical Excel (right_only):",
      df_merge_debug.loc[df_merge_debug["_merge"] == "right_only", "case_id"].head().tolist())


# Inner join: keep only patients that have both CT and clinical rows
df_merged = pd.merge(df_ct, df_clin, on="case_id", how="inner")

print(f"\nMerged rows (inner join): {len(df_merged)}")
print("  T_Stage:", df_merged["T_Stage"].unique()[:5])

print("  N_Stage:", df_merged["N_Stage"].unique()[:5])

print("  M_Stage:", df_merged["M_Stage"].unique()[:5])

print("  Histopathological:", df_merged["Histopathological_grading"].unique()[:5])


# ---------- Drop unusable NIfTI files ----------

good_indices = []
bad_paths = []

for idx, row in df_merged.iterrows():
    ct_path = row["ct_path"]
    try:
        _ = load_nifti_volume_as_float(ct_path)

    except Exception as e:
        print(f"[skip] Problem reading {ct_path}: {e}")
        bad_paths.append(ct_path)
        continue
    good_indices.append(idx)

df_merged = df_merged.iloc[good_indices].reset_index(drop=True)

print(f"\nAfter dropping bad NIfTIs: kept {len(df_merged)} rows, skipped {len(bad_paths)}")

# ---------- Stage -> binary label (early=0 / advanced=1) ----------

def parse_stage_numeric(val):
    """
    Extract the leading numeric digit from entries like '2b', '1c', 3, '1a', etc.
    Returns 0 if no digit is found or val is NaN.
    """
    if pd.isna(val):
        return 0
    s = str(val).strip()
    for ch in s:
        if ch.isdigit():
            return int(ch)
    return 0

def stage_row_to_binary(row):
    """
    Simple heuristic:
      advanced (1) if:
        - M >= 1 OR
        - N >= 2 OR
        - T >= 3
      else early (0)
    """
    Tn = parse_stage_numeric(row["T_Stage"])
    Nn = parse_stage_numeric(row["N_Stage"])
    Mn = parse_stage_numeric(row["M_Stage"])


    if Mn >= 1 or Nn >= 2 or Tn >= 3:
        return 1  # advanced
    else:
        return 0  # early

df_merged["label_bin"] = df_merged.apply(stage_row_to_binary, axis=1).astype(int)

print("\nLabel distribution (0=early,1=advanced):")
print(df_merged["label_bin"].value_counts())


# ---------- Build tabular features for multimodal model ----------

# Sex -> binary (M=1, F=0, anything else -> 0)
def map_sex(s):
    s = str(s).strip().upper()
    if s in ("M", "MALE"):
        return 1.0
    elif s in ("F", "FEMALE"):
        return 0.0
    else:
        return 0.0

df_merged["Sex_bin"] = df_merged["Sex"].apply(map_sex).astype(np.float32)


# Smoking history -> binary (already 0/1 in Excel, but clean it up)
if "Smoking History" in df_merged.columns:
    df_merged["Smoking_bin"] = (
        pd.to_numeric(df_merged["Smoking History"], errors="coerce")

        .fillna(0)
        .astype(np.float32)
    )
else:
    df_merged["Smoking_bin"] = 0.0

# Age & weight numeric
df_merged["Age"] = pd.to_numeric(df_merged["Age"], errors="coerce")

df_merged["weight (kg)"] = pd.to_numeric(df_merged["weight (kg)"], errors="coerce")


# Fill NaNs with column means
for col in ["Age", "weight (kg)"]:
    mean_val = df_merged[col].mean()
    df_merged[col] = df_merged[col].fillna(mean_val)


# Standardize (z-score)
df_merged["Age_z"] = (df_merged["Age"] - df_merged["Age"].mean()) / df_merged["Age"].std()
df_merged["Weight_z"] = (df_merged["weight (kg)"] - df_merged["weight (kg)"].mean()) / df_merged["weight (kg)"].std()

# List of tabular feature columns we’ll feed into the MLP
feature_cols = ["Age_z", "Weight_z", "Sex_bin", "Smoking_bin"]
num_tab_features = len(feature_cols)
print("\nUsing tabular features:", feature_cols)

# Class names for reporting
classes = ["early", "advanced"]

# ---------- Dataset with stacked slices (contrast stretch) ----------

class LungCTDataset(Dataset):
    def __init__(self, df, feature_cols, n_slices=N_SLICES,
                 train: bool = False, augment: bool = False):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.n_slices = n_slices
        self.train = train          # keep track of train/val/test
        self.augment = augment      # whether to apply data augmentation

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ct_path = row["ct_path"]
        label_int = int(row["label_bin"])

        # --- load volume robustly as float32 ---
        vol = load_nifti_volume_as_float(ct_path)

        # --- stack multiple slices around the middle as channels ---
        if vol.ndim == 3:
            H, W, D = vol.shape
            mid = D // 2
            half = self.n_slices // 2
            slice_idxs = [min(max(mid + o, 0), D - 1) for o in range(-half, half + 1)]
            slices = [vol[:, :, z] for z in slice_idxs]
        elif vol.ndim == 4:
            H, W, D, C = vol.shape
            mid = D // 2
            half = self.n_slices // 2
            slice_idxs = [min(max(mid + o, 0), D - 1) for o in range(-half, half + 1)]
            # use first channel of 4D volume
            slices = [vol[:, :, z, 0] for z in slice_idxs]
        else:
            raise ValueError(f"Unexpected volume shape {vol.shape} for {ct_path}")

        # (N_SLICES, H, W)
        img_np = np.stack(slices, axis=0).astype(np.float32)

        # --- to tensor, resize to IMG_SIZE x IMG_SIZE ---
        img = torch.from_numpy(img_np)          # (C, H, W)
        img = img.unsqueeze(0)                  # (1, C, H, W)
        img = F.interpolate(
            img, size=(IMG_SIZE, IMG_SIZE),
            mode="bilinear", align_corners=False
        )
        img = img.squeeze(0)                    # (C, IMG_SIZE, IMG_SIZE)

        # --- standardize per image (over all channels + pixels) ---
        mean = img.mean()
        std = img.std()
        if std > 0:
            img = (img - mean) / std

        # ---------- contrast ----------
        img_mean = img.mean()
        high = img > img_mean
        low = img <= img_mean
        img = img.clone()
        img[high] = img[high] + 0.5 * (img[high] - img_mean)
        img[low] = img[low] - 0.5 * (img_mean - img[low])
        img = torch.clamp(img, -3.0, 3.0)

        # ---------- data augmentation (train only) ----------
        if self.augment and self.train:
            # random rotation by multiples of 90°
            k = torch.randint(0, 4, (1,)).item()
            img = torch.rot90(img, k, dims=(1, 2))

            # random horizontal flip
            if torch.rand(1) < 0.5:
                img = torch.flip(img, dims=(2,))

        # tabular features
        feats_np = row[self.feature_cols].values.astype(np.float32)
        feats = torch.from_numpy(feats_np)      # shape: (num_tab_features,)

        # label tensor
        label = torch.tensor(label_int, dtype=torch.long)

        return img, feats, label
    
# ---------- Stratified 60/20/20 split on df_merged ----------

labels_np = df_merged["label_bin"].values
indices = np.arange(len(df_merged))

train_idx, temp_idx, y_train, y_temp = train_test_split(
    indices,
    labels_np,
    test_size=0.4,
    random_state=42,
    stratify=labels_np,
)

val_idx, test_idx, y_val, y_test = train_test_split(
    temp_idx,
    labels_np[temp_idx],
    test_size=0.5,
    random_state=42,
    stratify=labels_np[temp_idx],
)

train_df = df_merged.iloc[train_idx].reset_index(drop=True)
val_df   = df_merged.iloc[val_idx].reset_index(drop=True)
test_df  = df_merged.iloc[test_idx].reset_index(drop=True)

print("\nSplit sizes (stratified 60/20/20):")
print("  Train:", len(train_df))
print("  Val:  ", len(val_df))
print("  Test: ", len(test_df))

print("\nLabel distribution per split:")
print("Train:\n", train_df["label_bin"].value_counts())
print("Val:\n", val_df["label_bin"].value_counts())
print("Test:\n", test_df["label_bin"].value_counts())

# Create datasets
train_ds = LungCTDataset(train_df, feature_cols, n_slices=N_SLICES, train=True,  augment=True)
val_ds   = LungCTDataset(val_df,   feature_cols, n_slices=N_SLICES, train=False, augment=False)
test_ds  = LungCTDataset(test_df,  feature_cols, n_slices=N_SLICES, train=False, augment=False)

# Data loaders
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=16, shuffle=False, num_workers=0)

# ---------- Multi-modal CNN (image + tabular) with stacked slices ----------

class LungCT_MultiModal(nn.Module):
    def __init__(self, num_tab_features, n_slices=N_SLICES):
        super().__init__()

        # image branch: 4 conv blocks (deeper)
        # NOTE: in_channels = n_slices (stacked axial slices)
        self.conv1 = nn.Conv2d(n_slices, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(256)

        # after 4x maxpool (2x2) on 64x64 → 4x4
        flat_dim = 256 * (IMG_SIZE // 16) * (IMG_SIZE // 16)

        self.fc_img = nn.Linear(flat_dim, 256)
        self.dropout_img = nn.Dropout(p=0.3)

        # tabular branch
        self.fc_tab = nn.Linear(num_tab_features, 64)

        # combined branch
        self.fc_comb = nn.Linear(256 + 64, 128)
        self.dropout_comb = nn.Dropout(p=0.4)
        self.fc_out = nn.Linear(128, 2)  # early / advanced

    def forward(self, x_img, x_tab):
        # image path
        x = F.relu(self.bn1(self.conv1(x_img)))
        x = F.max_pool2d(x, 2)  # /2

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)  # /4

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)  # /8

        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)  # /16

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_img(x))
        x = self.dropout_img(x)

        # tabular path
        t = F.relu(self.fc_tab(x_tab))

        # combine
        z = torch.cat([x, t], dim=1)
        z = F.relu(self.fc_comb(z))
        z = self.dropout_comb(z)
        logits = self.fc_out(z)
        return logits

model = LungCT_MultiModal(num_tab_features=num_tab_features, n_slices=N_SLICES).to(device)
print(model)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# LR scheduler
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

# ---------- Train ----------

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

            running_loss   += loss.item() * x_img.size(0)
            preds = logits.argmax(dim=1)
            running_correct += (preds == y).sum().item()
            running_total   += x_img.size(0)

        train_loss = running_loss / running_total
        train_acc  = running_correct / running_total

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

    # Plot training vs validation loss
    plt.figure(figsize=(6, 6))
    epochs_axis = np.arange(1, epochs + 1)
    plt.plot(epochs_axis, train_losses, label="Training Loss")
    plt.plot(epochs_axis, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training vs Validation Loss (Early vs Advanced, Multi-modal, {N_SLICES}-slice)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot training vs validation accuracy
    plt.figure(figsize=(6, 6))
    plt.plot(epochs_axis, train_accs, label="Training Accuracy")
    plt.plot(epochs_axis, val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)
    plt.title(f"Training vs Validation Accuracy (Early vs Advanced, Multi-modal, {N_SLICES}-slice)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Final test
    test_loss, test_acc = evaluate(test_loader)
    print(f"Final Test Loss and Accuracy | loss={test_loss:.4f} acc={test_acc:.4f}")

    # Simple bar plot for final test accuracy
    plt.figure(figsize=(4, 5))

    plt.bar(["Test Accuracy"], [test_acc])
    plt.ylim(0.0, 1.0)
    plt.ylabel("Accuracy")
    plt.title("Final Test Accuracy")
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
    print(f"Test accuracy (recomputed): {test_acc:.4f}")

    show_confusion_matrix(
        test_preds, test_labels, classes,
        f"Early vs Advanced Confusion Matrix (Multi-modal, {N_SLICES}-slice)"
    )