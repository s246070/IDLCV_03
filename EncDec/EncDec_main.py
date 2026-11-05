import os, json, csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from time import time
from IDLCV_03.lib.model.EncDecModel import EncDec
from IDLCV_03.lib.dataset.dataset import PH2
from IDLCV_03.lib.metrics import SegmentationMetrics
import matplotlib
matplotlib.use("Agg")       
import matplotlib.pyplot as plt

seed = 42
size = 128
batch_size = 6
learning_rate = 0.001
epochs = 20

transform = transforms.Compose([transforms.Resize((size, size)),
                                transforms.ToTensor()])
dataset = PH2(transform=transform)

train_set, val_set, test_set = torch.utils.data.random_split(
    dataset, [0.7, 0.1, 0.2], generator=torch.Generator().manual_seed(seed))

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=3)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=3)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=3)

print(f"Loaded {len(train_set)} training images")
print(f"Loaded {len(val_set)} validation images")
print(f"Loaded {len(test_set)} test images")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = EncDec().to(device)
opt = optim.Adam(model.parameters(), learning_rate)
loss_fn = nn.BCEWithLogitsLoss()
metrics = SegmentationMetrics(threshold=0.5, from_logits=True)

results_dir = "IDLCV_03/EncDec/results"

@torch.no_grad()
def evaluate(loader):
    model.eval()
    metrics.reset()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        total_loss += loss_fn(y_pred, y).item() / len(loader)
        metrics.update(y_pred, y)
    return total_loss, metrics.compute()


history = []
best_dice = -1

for epoch in range(epochs):
    tic = time()
    model.train()
    train_loss = 0
    for X_batch, y_true in train_loader:
        X_batch, y_true = X_batch.to(device), y_true.to(device)

        opt.zero_grad()
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        opt.step()

        train_loss += loss.item() / len(train_loader)

    val_loss, val_m = evaluate(val_loader)

    history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss,
                    **{f"val_{k}": v for k, v in val_m.items()}})
    print(f"* Epoch {epoch+1}/{epochs} - train: {train_loss:.4f} | val: {val_loss:.4f} | "
          f"val Dice: {val_m['Dice']:.4f} | val IoU: {val_m['IoU']:.4f}")

    if val_m["Dice"] > best_dice:
        best_dice = val_m["Dice"]
        torch.save(model.state_dict(), f"{results_dir}/best_model.pth")

with open(f"{results_dir}/history.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=history[0].keys())
    w.writeheader()
    w.writerows(history)

model.load_state_dict(torch.load(f"{results_dir}/best_model.pth", map_location=device))
test_loss, test_m = evaluate(test_loader)
test_m["loss"] = test_loss
with open(f"{results_dir}/test_metrics.json", "w") as f:
    json.dump(test_m, f, indent=2)


@torch.no_grad()
def save_predictions(loader, path, n=8):
    model.eval()
    imgs, gts, preds = [], [], []
    for X, y in loader:
        p = (torch.sigmoid(model(X.to(device))) >= 0.5).cpu()   # logit -> bináris maszk
        imgs.append(X); gts.append(y >= 0.5); preds.append(p)
        if sum(len(i) for i in imgs) >= n:
            break
    imgs = torch.cat(imgs)[:n]
    gts = torch.cat(gts)[:n].reshape(n, size, size)
    preds = torch.cat(preds)[:n].reshape(n, size, size)

    fig, ax = plt.subplots(n, 4, figsize=(10, 2.5 * n))
    for i in range(n):
        g, p = gts[i], preds[i]
        tp, fp, fn = (g & p).sum().item(), (~g & p).sum().item(), (g & ~p).sum().item()
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0

        err = torch.zeros(size, size, 3)
        err[g & p] = torch.tensor([0., 1., 0.])
        err[~g & p] = torch.tensor([1., 0., 0.])
        err[g & ~p] = torch.tensor([0., 0., 1.])

        ax[i, 0].imshow(imgs[i].permute(1, 2, 0)); ax[i, 0].set_title("Image")
        ax[i, 1].imshow(g, cmap="gray");           ax[i, 1].set_title("Ground truth")
        ax[i, 2].imshow(p, cmap="gray");           ax[i, 2].set_title(f"Prediction (Dice {dice:.3f})")
        ax[i, 3].imshow(err);                      ax[i, 3].set_title("TP / FP / FN")
        for a in ax[i]:
            a.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)


save_predictions(test_loader, f"{results_dir}/test_predictions.png", n=8)