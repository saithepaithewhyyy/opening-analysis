import torch
import torch.nn as nn
from torch.utils.data import Dataset

import load_data as ld

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
    
def make_save_data(folder_path=".."):
    _, bb, sc, targets, eco_classes = ld.load_data()
    valid = targets.sum(axis=1) > 0
    bb = bb[valid]
    sc = sc[valid]
    targets = targets[valid]

    dataset = OpeningDataset(bb, sc, targets)
    return dataset
