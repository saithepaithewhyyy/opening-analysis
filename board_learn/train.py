import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.model_selection import train_test_split

import load_data as ld
import opening_model as om

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
    
def train():
    
    _, bb, sc, targets, eco_classes = ld.load_data()
    valid = targets.sum(axis=1) > 0
    bb = bb[valid]
    sc = sc[valid]
    targets = targets[valid]

    dataset = OpeningDataset(bb, sc, targets)
    
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
    
    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    scaler = torch.cuda.amp.GradScaler(enabled = (device.type=="cuda"))

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for bb, sc, target in train_loader:
            bb = bb.to(device)
            sc = sc.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled = (device.type=="cuda")):
                out = model(bb, sc)
                loss = criterion(out, target)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            
        scheduler.step()
        
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for bb, sc, target in test_loader:
                bb = bb.to(device)
                sc = sc.to(device)
                target = target.to(device)
                out = model(bb, sc)
                val_loss += criterion(out, target).item()

        print(f"Epoch {epoch+1:02d} | train_loss={total_loss/len(train_loader):.4f} | "
              f"val_loss={val_loss/len(test_loader):.4f}")
            
    
    return
    
if __name__ == "__main__":
    train()
