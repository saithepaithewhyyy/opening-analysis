import torch
import torch.nn as nn
from torch.utils.data import Dataset

class OpeningDataset(Dataset):
    def __init__(self, bitboards_all, scalars_all, targets_all):
        # bitboards -> (n, 13, 64)
        # scalars -> (n, 7 or 8 idk i might change it)
        # targets -> (n, n_classes)
        self.bitboards = torch.from_numpy(bitboards_all)
        self.scalars = torch.from_numpy(scalars_all)
        self.targets = torch.from_numpy(targets_all)

    def __len__(self):
        return len(self.bitboards)

    def __getitem__(self, idx):
        return self.bitboards[idx], self.scalars[idx], self.targets[idx]
