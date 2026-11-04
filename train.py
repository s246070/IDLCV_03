import os
import numpy as np
import glob
import PIL.Image as Image

# pip install torchsummary
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torchsummary import summary
import torch.optim as optim
from time import time
from lib.model.EncDecModel import EncDec
from lib.model.UNetModel import UNet
from IDLCV_03.lib.dataset.dataset import DRIVE, PH2
from sklearn.model_selection import train_test_split

seed = 42
# Dataset
size = 128
train_transform = transforms.Compose([transforms.Resize((size, size)),
                                    transforms.ToTensor()])
test_transform = transforms.Compose([transforms.Resize((size, size)),
                                    transforms.ToTensor()])

batch_size = 6
labeledset = DRIVE(train=True, transform=train_transform)
unlabeledset = DRIVE(train=False, transform=test_transform)

train_subset_l, val_subset_l, test_subset_l = torch.utils.data.random_split(labeledset, [0.7, 0.1, 0.2], generator=torch.Generator().manual_seed(seed))
train_subset, val_subset, test_subset = torch.utils.data.random_split(unlabeledset, [0.7, 0.1, 0.2], generator=torch.Generator().manual_seed(seed))

# X_train_l, X_temp_l, y_train_l, y_temp_l = train_test_split(X_labeled, y_labeled, test_size=0.3, random_state=seed)
# X_val_l, X_test_l, y_val_l, y_test_l = train_test_split(X_labeled, y_labeled, test_size=2/3, random_state=seed)

# X_train, X_temp, y_train, y_temp = train_test_split(X_unlabeled, y_unlabeled, test_size=0.3, random_state=seed)
# X_val, X_test, y_val, y_test = train_test_split(X_unlabeled, y_unlabeled, test_size=2/3, random_state=seed)

train_set = train_subset_l.__add__(train_subset)
val_set = val_subset_l.__add__(val_subset)
test_set = test_subset_l.__add__(test_subset)


train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=3)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=3)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=3)

# dataset = PH2(transform=train_transform)
# data_loadet = DataLoader(dataset, batch_size=batch_size, num_workers=3)

print(f"Loaded {len(train_set)} training images")
print(f"Loaded {len(val_loader)} validation images")
print(f"Loaded {len(test_loader)} test images")

# Training setup
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#model = EncDec().to(device)
model = UNet().to(device)
learning_rate = 0.001
opt = optim.Adam(model.parameters(), learning_rate)

#loss_fn = BCELoss()
#loss_fn = DiceLoss() # TODO
# loss_fn = FocalLoss() # TODO
loss_fn = nn.BCEWithLogitsLoss()
epochs = 20

# Training loop
X_test, Y_test = next(iter(test_loader))
model.train()  # train mode
for epoch in range(epochs):
    tic = time()
    print(f'* Epoch {epoch+1}/{epochs}')

    avg_loss = 0
    for X_batch, y_true in train_loader:
        X_batch = X_batch.to(device)
        y_true = y_true.to(device)

        # set parameter gradients to zero
        opt.zero_grad()

        # forward
        y_pred = model(X_batch)
        # IMPORTANT NOTE: Check whether y_pred is normalized or unnormalized
        # and whether it makes sense to apply sigmoid or softmax.
        loss = loss_fn(y_pred, y_true)  # forward-pass
        loss.backward()  # backward-pass
        opt.step()  # update weights

        # calculate metrics to show the user
        avg_loss += loss / len(train_loader)
    # IMPORTANT NOTE: It is a good practice to check performance on a
    # validation set after each epoch.
    #model.eval()  # testing mode
    #Y_hat = F.sigmoid(model(X_test.to(device))).detach().cpu()
    print(f' - loss: {avg_loss}')

# Save the model
torch.save(model, './model.pth')
print("Training has finished!")
