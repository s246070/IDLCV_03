import torch

class SegmentationMetrics:
    def __init__(self, threshold=0.5, from_logits=False):
        self.threshold = threshold
        self.from_logits = from_logits
        self.reset()

    def reset(self):
        # Reset confusion matrix components
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def update(self, y_pred, y_true, fov_mask=None):
        # Check that prediction and ground truth have matching shapes
        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"Shape mismatch: {y_pred.shape} vs {y_true.shape}"
            )

        # Convert logits to probabilities if needed
        if self.from_logits:
            y_pred = torch.sigmoid(y_pred)

        # Convert predictions and labels to binary tensors
        y_pred = (y_pred >= self.threshold)
        y_true = (y_true >= 0.5)

        # Only evaluate pixels inside the field of view if provided
        if fov_mask is not None:
            if fov_mask.shape != y_true.shape:
                raise ValueError(
                    f"FOV shape mismatch: {fov_mask.shape} vs {y_true.shape}"
                )

            valid = (fov_mask >= 0.5)
            y_pred = y_pred[valid]
            y_true = y_true[valid]

        # Confusion matrix components
        self.tp += (y_pred & y_true).sum().item()
        self.tn += (~y_pred & ~y_true).sum().item()
        self.fp += (y_pred & ~y_true).sum().item()
        self.fn += (~y_pred & y_true).sum().item()

    def compute(self):
        tp, tn = self.tp, self.tn
        fp, fn = self.fp, self.fn

        def safe_div(num, den):
            # Avoid division by zero
            return num / den if den > 0 else 0.0

        return {
            "Dice": safe_div(2 * tp, 2 * tp + fp + fn),
            "IoU": safe_div(tp, tp + fp + fn),
            "Accuracy": safe_div(tp + tn, tp + tn + fp + fn),
            "Sensitivity": safe_div(tp, tp + fn),
            "Specificity": safe_div(tn, tn + fp),
        }