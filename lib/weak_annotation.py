
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

    def annotate(self, idx, num_positive_samples=10, num_negative_samples=10):

        # Reproducible random sampling for each image
        generator = torch.Generator().manual_seed(self.seed + idx)

        mask = (self.vessel_masks[idx].detach().to(self.device).numpy().squeeze() > 0)
        Y_dist = distance_transform_edt(mask)
        positive_cand = mask  # Keep all vessel pixels

        # convert positive_cand to a tensor with 0 and 1 values
        positive_cand_tensor = torch.from_numpy(positive_cand.astype(int))

        # pick 10 random positive candidates, weighted by distance
        positive_coords = torch.nonzero(positive_cand_tensor)

        if positive_coords.size(0) > 0:
            weights = torch.from_numpy(Y_dist[positive_cand]).float() ** 0.5
            probs = weights / weights.sum()

            random_indices = torch.multinomial(
                probs,
                min(num_positive_samples, positive_coords.size(0)),
                replacement=False,
                generator=generator
            )
        else:
            random_indices = torch.empty(0, dtype=torch.long)

        # pick random negative candidates
        negative_cand = ~mask
        negative_cand_tensor = torch.from_numpy(negative_cand.astype(int))
        negative_coords = torch.nonzero(negative_cand_tensor)

        random_negative_indices = torch.randperm(negative_coords.size(0),generator=generator)[:num_negative_samples]

        # put positive and negative samples in a single matrix
        candidates_matrix = torch.zeros(mask.shape, dtype=torch.int)

        for i in random_indices:
            coord = positive_coords[i]
            candidates_matrix[coord[0], coord[1]] = 1

        for i in random_negative_indices:
            coord = negative_coords[i]
            candidates_matrix[coord[0], coord[1]] = -1

        return candidates_matrix.to(self.device)


if __name__ == "__main__":
    vessel_masks = torch.randint(0, 2, (5, 128, 128))
    weak_annotator = Weak_Annotation(vessel_masks)

    for i in range(len(vessel_masks)):
        candidates_matrix = weak_annotator.annotate(i)
        print(f"Number of positive samples: {(candidates_matrix == 1).sum().item()}")
        print(f"Number of negative samples: {(candidates_matrix == -1).sum().item()}")
