import torch.nn as nn
import torch.nn.functional as F

# Model Architechture:- 
# Dual Stream transformer approach
# 
# One stream attends to only the squares, while one attends to the pieces
# Finally another cross attention layer is applied to get a piece square relation
# 
# Square stream input transform - (batch, 64, 12) 
# Piece stream input transform (extra dim for overall occupied) - (batch, 13, 64)
# 
# Scalar features (per board) - {castling rights, en passant, side to move, target entropy, depth statistics}

class OpeningModel(nn.Module):
    def __init__(self, 
                n_classes,
                n_bb = 12, 
                n_sq = 64, 
                n_flat = 8, 
                d_model = 256, 
                nhead = 8, 
                n_layers = 6, 
                dim_ff = 256, 
                dropout = 0.1):
        
        super().__init__()
        
        
        

        
    def forward(self, x):
        return x
        