import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Paths

BASE_DIR = r"C:\Users\patha\Desktop\ML Diabetic Retinopathy"

csv_path    = os.path.join(BASE_DIR, "messidor2", "messidor_data.csv")
img_folder  = os.path.join(BASE_DIR, "messidor2", "IMAGES")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)
print("CSV path    :", csv_path)
print("Images path :", img_folder)
print("Results dir :", RESULTS_DIR)

# 2. Transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # match ResNet50 input
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet stats
        std=[0.229, 0.224, 0.225]
    )
])

# 3. Load CSV and cleanign the data

df = pd.read_csv(csv_path)

# we drop rows without labels bole toh preprocessing step :)
df = df.dropna(subset=["adjudicated_dr_grade"])
df["adjudicated_dr_grade"] = df["adjudicated_dr_grade"].astype(int)

print("Total rows in CSV after dropping NaNs:", len(df))


# 4. Dataset class with 3-class mapping
# basically hamko allow krta h images aur csv ki files ko reaad nd fir manipulate krne
# ke liye jisse ham 5 classes ko 3 class label krte baadme
class MessidorDataset(Dataset):
    """
    We map to 3 classes:
        0 -> No DR (0)
        1 -> Mild/Moderate (1, 2)
        2 -> Severe/Proliferative (>=3)
    """
    def __init__(self, dataframe, root_dir, transform=None):
        self.data = dataframe.copy()
        self.root_dir = root_dir
        self.transform = transform

        # check .png extension and if nhi toh add krdo
        self.data["image_id"] = self.data["image_id"].astype(str).apply(
            lambda x: x if x.endswith(".png") else x + ".png"
        )

        # sirf un files ko rkho (image) jinke data csv m exist krta
        valid_files = set(os.listdir(root_dir))
        before = len(self.data)
        self.data = self.data[self.data["image_id"].isin(valid_files)].reset_index(drop=True)
        after = len(self.data)

        print(f"Filtered dataset: {before} -> {after} images (existing files only)")

    def __len__(self):
        return len(self.data)

#idhar hori h mapping shapping

    def _map_label_5_to_3(self, raw_label: int) -> int:
        if raw_label == 0:
            return 0
        elif raw_label in [1, 2]:
            return 1
        else:
            return 2

#har image ka path laakr deta h
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_name = row["image_id"]
        img_path = os.path.join(self.root_dir, img_name)
# image ko rgb convert and if nhi ho exact image path aur error throw krta
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {e}")
# alsi chnage idhr hora h labels m(func ) calling
        raw_label = int(row["adjudicated_dr_grade"])
        label = self._map_label_5_to_3(raw_label)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


# 5. Train/Val split wahi usual 80-20

labels_raw = df["adjudicated_dr_grade"].values

train_df, val_df = train_test_split(df,test_size=0.2,random_state=42,stratify=labels_raw)

print("Train samples:", len(train_df))
print("Val samples:", len(val_df))
#do dataset instances bn rhe test aur train ke
train_dataset = MessidorDataset(train_df, img_folder, transform=transform)
val_dataset   = MessidorDataset(val_df,   img_folder, transform=transform)
#32 samples per mini batch
batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                          num_workers=0, pin_memory=True)

# 6. Model: ResNet50 with 3-class head(basically hydrogen bomb vs coughing baby lol)

print("Building model...")

try:
    weights = models.ResNet50_Weights.IMAGENET1K_V1
    backbone = models.resnet50(weights=weights)
except Exception:
    backbone = models.resnet50(pretrained=True)

num_ftrs = backbone.fc.in_features
backbone.fc = nn.Linear(num_ftrs, 3)  # 3 classes

model = backbone.to(device)

# 7. Loss, optimizer, scheduler

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                 factor=0.5, patience=3, verbose=True)

# 8. Training loop

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
#sends images to GPU
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss=running_loss/total
    epoch_acc=correct/total

    return epoch_loss,epoch_acc


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss/total
    epoch_acc = correct/total

    return epoch_loss,epoch_acc

#main trainig idhr
num_epochs=20 
best_val_acc=0.0
best_model_path=os.path.join(RESULTS_DIR, "best_model_3class.pth")

train_losses, val_losses=[], []
train_accs, val_accs=[], []

for epoch in range(1, num_epochs + 1):
    train_loss,train_acc=train_one_epoch(model,train_loader,optimizer,criterion,device)
    val_loss,val_acc     = eval_one_epoch(model, val_loader, criterion, device)

#store these for plotting grph
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
#current valdation loss k according lr adjust krta
    scheduler.step(val_loss)

    print(f"Epoch [{epoch}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"--> New best model saved with Val Acc: {best_val_acc:.4f}")

print("Training complete. Best Val Acc:", best_val_acc)
print("Best model saved at:", best_model_path)

# 9. Plot training curves (inko save bhi kra h)

epochs_range = range(1, num_epochs + 1)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label="Train Loss")
plt.plot(epochs_range, val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accs, label="Train Acc")
plt.plot(epochs_range, val_accs, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy")
plt.legend()

plt.tight_layout() #over na ho isiliye
plt.savefig(os.path.join(RESULTS_DIR,"training_curves.png"), bbox_inches="tight")
plt.close()

# 10. Reload best model for final eval

best_model = backbone
best_model.load_state_dict(torch.load(best_model_path, map_location=device))
best_model.to(device) #device is GPU
best_model.eval()

# 11. Collect predictions & probs on val set

def collect_predictions(model, loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)               # [B, 3]
            probs = F.softmax(outputs, dim=1)     # [B, 3]

            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy()) # GPU to CPU then convert to Numpy
            all_probs.extend(probs.cpu().numpy()) #.exted to add to list

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


all_labels, all_preds, all_probs = collect_predictions(best_model, val_loader, device)

class_names = ["No DR", "Mild/Moderate DR", "Severe DR"]
num_classes = 3

# 12. Confusion Matrix, Report, QWK

cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:\n", cm)

print("\nClassification Report:\n",
      classification_report(all_labels, all_preds, target_names=class_names))

qwk = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
print(f"\nQuadratic Weighted Kappa (QWK): {qwk:.4f}")

plt.figure(figsize=(7, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Val Set, 3-class)")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), bbox_inches="tight")
plt.close()

# 13. Histogram of predictions

plt.figure(figsize=(7, 5))
plt.hist(
    all_preds,
    bins=np.arange(-0.5, num_classes + 0.5, 1),
    edgecolor="black",
    alpha=0.7
)
plt.xticks(range(num_classes), class_names, rotation=15)
plt.xlabel("Predicted Class")
plt.ylabel("Count")
plt.title("Distribution of Predictions (Val Set)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "prediction_histogram.png"), bbox_inches="tight")
plt.close()

# 14. Precision-Recall Curves

all_labels_bin=label_binarize(all_labels, classes=list(range(num_classes)))

precision={}
recall={}
avg_precision={}

for i in range(num_classes):
    precision[i], recall[i], _ = precision_recall_curve(all_labels_bin[:, i], all_probs[:, i])
    avg_precision[i] = average_precision_score(all_labels_bin[:, i], all_probs[:, i])

precision["micro"], recall["micro"], _ = precision_recall_curve(
    all_labels_bin.ravel(), all_probs.ravel()
)
avg_precision["micro"]=average_precision_score(all_labels_bin, all_probs, average="micro")

plt.figure(figsize=(8, 6))
for i in range(num_classes):
    plt.plot(
        recall[i],
        precision[i],
        label=f"{class_names[i]} (AP = {avg_precision[i]:.2f})"
    )

plt.plot(
    recall["micro"],
    precision["micro"],
    label=f"Micro-average (AP = {avg_precision['micro']:.2f})",
    linestyle="--",
    linewidth=2,
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve (Val Set, 3-class)")
plt.legend(loc="lower left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "precision_recall_curves.png"), bbox_inches="tight")
plt.close()

# 15. ROC Curves

fpr={}
tpr={}
roc_auc={}

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(all_labels_bin[:, i], all_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(all_labels_bin.ravel(), all_probs.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure(figsize=(8, 6))
for i in range(num_classes):
    plt.plot(
        fpr[i],
        tpr[i],
        label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})"
    )

plt.plot(
    fpr["micro"],
    tpr["micro"],
    label=f"Micro-average (AUC = {roc_auc['micro']:.2f})",
    linestyle="--",
    linewidth=2,
)

plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Val Set, 3-class)")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "roc_curves.png"), bbox_inches="tight")
plt.close()

print("\nAll plots saved in:", RESULTS_DIR)
