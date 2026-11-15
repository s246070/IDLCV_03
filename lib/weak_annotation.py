from scipy.ndimage import distance_transform_edt
import numpy as np
import torch

class Weak_Annotation():
    def __init__(self, vessel_masks, seed=42, device='cpu'):
        # the vessel_masks is what is called Y in the notebook
        # e.g. X, Y = trainset[0]
        self.vessel_masks = vessel_masks
        self.seed = seed
        self.device = device

    def annotate(self, idx, num_positive_samples=10, num_negative_samples=10, sampling="distance"):

        if sampling not in ("distance", "uniform"):
            raise ValueError(f"Unknown sampling mode: {sampling}")

        generator = torch.Generator().manual_seed(self.seed + idx)

        mask = (self.vessel_masks[idx].detach().cpu().numpy().squeeze() > 0)

        positive_coords = torch.nonzero(torch.from_numpy(mask))
        n_pos = min(num_positive_samples, positive_coords.size(0))

        if n_pos > 0:
            if sampling == "distance":
                Y_dist = distance_transform_edt(mask)
                weights = torch.from_numpy(Y_dist[mask]).float() ** 0.5
                probs = weights / weights.sum()
                pos_idx = torch.multinomial(probs, n_pos, replacement=False, generator=generator)
            else: 
                pos_idx = torch.randperm(positive_coords.size(0), generator=generator)[:n_pos]
            pos_picked = positive_coords[pos_idx]
        else:
            pos_picked = torch.empty((0, 2), dtype=torch.long)

        negative_coords = torch.nonzero(torch.from_numpy(~mask))
        neg_idx = torch.randperm(negative_coords.size(0), generator=generator)[:num_negative_samples]
        neg_picked = negative_coords[neg_idx]

        candidates_matrix = torch.zeros(mask.shape, dtype=torch.int)
        candidates_matrix[pos_picked[:, 0], pos_picked[:, 1]] = 1
        candidates_matrix[neg_picked[:, 0], neg_picked[:, 1]] = -1

        return candidates_matrix.to(self.device)


if __name__ == "__main__":
    vessel_masks = torch.randint(0, 2, (5, 128, 128))
    weak_annotator = Weak_Annotation(vessel_masks)

    for i in range(len(vessel_masks)):
        candidates_matrix = weak_annotator.annotate(i)
        print(f"Number of positive samples: {(candidates_matrix == 1).sum().item()}")
        print(f"Number of negative samples: {(candidates_matrix == -1).sum().item()}")
