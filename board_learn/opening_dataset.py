import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from . import load_data as ld

DATASET_FILE = "dataset.pt"

class OpeningDataset(Dataset):
    def __init__(self, bitboards_all, scalars_all, targets_all, eco_classes):
        # bitboards -> (n, 13, 64)
        # scalars -> (n, 14)
        # targets -> (n, tuple) -> each tuple has the sparse classes
        self.bitboards = torch.from_numpy(bitboards_all)
        self.scalars = torch.from_numpy(scalars_all)
        self.targets = targets_all
        self.eco_classes = eco_classes

    def __len__(self):
        return len(self.bitboards)

    def __getitem__(self, idx):
        return self.bitboards[idx].float(), self.scalars[idx].float(), self.targets[idx]
    
def opening_collate(batch):
    bb, sc, targets = zip(*batch)
    return (
        torch.stack(bb),
        torch.stack(sc),
        targets
    )
    
def make_save_data(folder_path="../"):
    if DATASET_FILE in os.listdir(folder_path):
        dataset = torch.load(folder_path + DATASET_FILE, weights_only=False)
        return dataset
    
    _, bb, sc, targets, eco_classes = ld.load_data(ld.PARQUET_PATH)
    valid = np.array([
        len(idxs) > 0
        for idxs, probs in targets
    ])
    bb = bb[valid]
    sc = sc[valid]
    targets = [ targets[i] for i in np.where(valid)[0]]

    dataset = OpeningDataset(bb, sc, targets, eco_classes)
    torch.save(dataset, folder_path + DATASET_FILE, pickle_protocol=4)
    return dataset

if __name__ == "__main__":
    make_save_data()
