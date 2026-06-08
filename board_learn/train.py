import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import load_data as ld
import opening_model as om
import opening_dataset as od
from opening_dataset import OpeningDataset, opening_collate
from utils import sparse_kl_loss

CHECKPOINTS_DIR = "checkpoints"
TEST_SIZE = 0.1
BATCH_SIZE = 256
NUM_WORKERS = 4
LR = 1e-3
WEIGHT_DECAY = 1e-2
CKPT_COUNT = 10
    
def train():

    dataset = od.make_save_data()
    labels = np.array([
        idxs[np.argmax(probs)]
        for idxs, probs in dataset.targets
    ])
    counts = np.bincount(labels)
    valid_mask = counts[labels] >= 2
    valid_indices = np.where(valid_mask)[0]

    train_indices, test_indices = train_test_split(
        valid_indices,
        test_size=TEST_SIZE,
        stratify=labels[valid_indices],
        random_state=42
    )
    
    train_data = Subset(dataset, train_indices)
    test_data = Subset(dataset, test_indices)

    device = torch.device('cuda' if torch.cuda.is_available()  # for cuda gpus
                          else 'mps' if torch.backends.mps.is_available() # for metal gpus (apple)
                          else 'cpu') # you dont have anything :(
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=opening_collate)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=opening_collate)
   
    eco_classes = dataset.eco_classes

    model = om.OpeningModel(n_classes=len(eco_classes)).to(device)
    model = torch.compile(model)
    
    num_batches = len(train_loader)
    print(f"number of batches: {num_batches}")
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    
    criterion = sparse_kl_loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_batches)
    
    best_val = float('inf')
    total_loss = 0.0
    print(f"Training Starting on {device}...")
    for i, data in enumerate(tqdm(train_loader)):
        model.train()
        
        bb, sc, target = data
        bb = bb.to(device)
        sc = sc.to(device)
        
        optimizer.zero_grad()
                    
        out = model(bb, sc)
        loss = criterion(out, target)
        loss.backward() # type: ignore
        lp = out
        probs = lp.exp()
        entropy = -(probs * lp).sum(dim=1)
        
        grad_norm_unclipped = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        total_loss += loss.item()  # type: ignore 
        scheduler.step()
        
        ckpt_iter = len(train_loader) / CKPT_COUNT
        if i%ckpt_iter == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for bb_test, sc_test, target_test in test_loader:
                    bb_test = bb_test.to(device)
                    sc_test = sc_test.to(device)
                    out_test = model(bb_test, sc_test)
                    val_loss += criterion(out_test, target_test).item() # type: ignore
                    
            avg_train = total_loss / (i+1)
            avg_val = val_loss / len(test_loader)
                    
            checkpoint = {
                'epoch': i+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train,
                'val_loss': avg_val,
                'eco_classes': eco_classes,
            }
            
            ckpt_path = os.path.join(CHECKPOINTS_DIR, f"checkpoint_epoch{i+1:02d}.pt")
            torch.save(checkpoint, ckpt_path)
            
            if avg_val < best_val:
                best_val = avg_val
                ckpt_path = os.path.join(CHECKPOINTS_DIR, "final_model.pt")
                torch.save(checkpoint, ckpt_path)

        tqdm.write(f"Step {i+1:02d} | train_loss={total_loss/(i+1):.4f} | "
              f"gradient norm unclipped={grad_norm_unclipped:.4f} | "
              f"entropy_mean={entropy.mean().item():.4f}")
            

    model.eval()
    val_loss = 0
    with torch.no_grad():
        print("Final validation over test set")
        for i, data in enumerate(tqdm(test_loader)):
            bb_test, sc_test, target_test = data
            bb_test = bb_test.to(device)
            sc_test = sc_test.to(device)
            out_test = model(bb_test, sc_test)
            val_loss += criterion(out_test, target_test).item() # type: ignore
            
    avg_train = total_loss / (i+1)
    avg_val = val_loss / len(test_loader)
    print(f"val_loss={avg_val:.4f}")
    
    if avg_val < best_val:
        torch.save(checkpoint, os.path.join(CHECKPOINTS_DIR, "final_model.pt"))
        
    return
    
if __name__ == "__main__":
    train()
