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
from opening_dataset import OpeningDataset
    
def train():

    dataset = od.make_save_data()
    labels = dataset.targets_all.argmax(axis=1)
    counts = np.bincount(labels)
    valid_mask = counts[labels] >= 2
    valid_indices = np.where(valid_mask)[0]

    train_indices, test_indices = train_test_split(
        valid_indices,
        test_size=0.2,
        stratify=labels[valid_indices],
        random_state=42
    )
    
    train_data = Subset(dataset, train_indices)
    test_data = Subset(dataset, test_indices)
    
    train_loader = DataLoader(train_data, batch_size=4000, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=4000, shuffle=True, num_workers=4, pin_memory=True)
   
    eco_classes = dataset.eco_classes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = om.OpeningModel(n_classes=len(eco_classes)).to(device)
    model = torch.compile(model)
    
    num_epochs=20
    print(f"number of epochs: {num_epochs}")
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val = float('inf')
    
    total_loss = 0.0
    print(f"Training Starting on {device}...")
    # for epoch in tqdm(range(num_epochs)):
    for i, data in enumerate(tqdm(train_loader)):
        model.train()
        
        bb, sc, target = data
        # for bb, sc, target in train_loader:
        bb = bb.to(device)
        sc = sc.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
                    
        out = model(bb, sc)
        loss = criterion(out, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
            
        scheduler.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for bb_test, sc_test, target_test in test_loader:
                bb_test = bb_test.to(device)
                sc_test = sc_test.to(device)
                target_test = target_test.to(device)
                out_test = model(bb_test, sc_test)
                val_loss += criterion(out_test, target_test).item()
                
        avg_train = total_loss / len(train_loader)
        avg_val = val_loss / len(test_loader)
                
        checkpoint = {
            'epoch': i + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train,
            'val_loss': avg_val,
            'eco_classes': eco_classes,
        }
        
        # ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch+1:02d}.pt")
        # torch.save(checkpoint, ckpt_path)
        
        if avg_val < best_val:
            best_val = avg_val
            ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, ckpt_path)

        print(f"Epoch {i+1:02d} | train_loss={total_loss/len(train_loader):.4f} | "
              f"val_loss={val_loss/len(test_loader):.4f}")
            
    torch.save(checkpoint, os.path.join(checkpoint_dir, "final_model.pt"))
    return
    
if __name__ == "__main__":
    train()
