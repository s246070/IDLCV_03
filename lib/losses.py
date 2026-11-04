from torch import nn
import torch

class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        loss = torch.mean(y_pred - y_true*y_pred + torch.log(1 + torch.exp(-y_pred)))
        return loss

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true): 
        y_pred = torch.sigmoid(y_pred) 
        numerator = torch.mean(2*y_pred*y_true +1)
        denominator = torch.mean(y_pred + y_true) + 1
        loss = 1 - numerator/denominator
        return loss

class FocalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, gamma =2.0):
        loss = -torch.mean((1-torch.sigmoid(y_pred))**gamma * y_true * torch.log(torch.sigmoid(y_pred)) + (1-y_true) * torch.log(1-torch.sigmoid(y_pred)))
        return loss

class BCELoss_TotalVariation(nn.Module):
    def __init__(self, regularizer="contiguity", reg_weight=0.1):
        super().__init__()

        valid_regularizers = {"centered", "sparsity", "contiguity", None}
        if regularizer not in valid_regularizers:
            raise ValueError(
                f"regularizer must be one of {valid_regularizers}, got {regularizer}"
            )

        self.regularizer = regularizer
        self.reg_weight = reg_weight

    def forward(self, y_pred, y_true):
        
        loss = torch.mean(y_pred - y_true*y_pred + torch.log(1 + torch.exp(-y_pred)))

        # No regularization
        if self.regularizer is None:
            return loss

        probs = torch.sigmoid(y_pred)
        H, W = probs.shape[-2:]

        if self.regularizer == "centered":
            regularization = 1 - probs[:, 0, H // 2, W // 2].mean()

        elif self.regularizer == "sparsity":
            regularization = probs.mean()

        elif self.regularizer == "contiguity":
            vertical = torch.abs(probs[:, :, 1:, :] - probs[:, :, :-1, :])
            horizontal = torch.abs(probs[:, :, :, 1:] - probs[:, :, :, :-1])
            regularization = vertical.mean() + horizontal.mean()

        return loss + self.reg_weight * regularization