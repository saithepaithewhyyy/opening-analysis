import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.model_selection import train_test_split

import load_data as ld
import opening_model as om
import opening_dataset as od
    
def train():

    dataset = od.make_save_data()
    train_indices, test_indices = train_test_split(
        range(len(dataset)),
        test_size=0.2,
        stratify=targets.argmax(axis=1),
        random_state=42
    )
    
    train_data = Subset(dataset, train_indices)
    test_data = Subset(dataset, test_indices)
    
    train_loader = DataLoader(train_data, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = om.OpeningModel(n_classes=len(eco_classes)).to(device)
    
    num_epochs=20
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for bb, sc, target in train_loader:
            bb = bb.to(device)
            sc = sc.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
                        
            out = model(bb, sc)
            loss = criterion(torch.log(out + 1e-9), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            
        scheduler.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for bb, sc, target in test_loader:
                bb = bb.to(device)
                sc = sc.to(device)
                target = target.to(device)
                out = model(bb, sc)
                val_loss += criterion(out, target).item()
                
        avg_train = total_loss / len(train_loader)
        avg_val = val_loss / len(test_loader)
                
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train,
            'val_loss': avg_val,
            'eco_classes': eco_classes,
        }
        
        ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch+1:02d}.pt")
        torch.save(checkpoint, ckpt_path)
        
        if avg_val < best_val:
            best_val = avg_val
            ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, ckpt_path)

        print(f"Epoch {epoch+1:02d} | train_loss={total_loss/len(train_loader):.4f} | "
              f"val_loss={val_loss/len(test_loader):.4f}")
            
    torch.save(checkpoint, os.path.join(checkpoint_dir, "final_model.pt"))
    return
    
if __name__ == "__main__":
    train()
