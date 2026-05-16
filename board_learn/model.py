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
                n_bb = 13, 
                n_sq = 64, 
                n_flat = 8, 
                d_model = 256, 
                nhead = 8, 
                num_layers = 6, 
                d_ff = 256,
                d_hidden = 1024,
                dropout = 0.1):
        
        super().__init__()
        
        self.scalar = nn.Linear(n_flat, d_model)
        
        self.sq_linear = nn.Linear(n_sq, d_model)
        self.row_embed = nn.Embedding(8, d_model // 2)
        self.col_embed = nn.Embedding(8, d_model // 2)
        
        self.bb_linear = nn.Linear(n_bb, d_model)
        self.bb_embed = nn.Embedding(n_bb, d_model)
        
        self.sq_transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_ff, batch_first=True, norm_first=False)
        self.bb_transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_ff, batch_first=True, norm_first=False)
        
        self.sq_transformer = nn.TransformerEncoder(self.sq_transformer_layer, num_layers=num_layers)
        self.bb_transformer = nn.TransformerEncoder(self.bb_transformer_layer, num_layers=num_layers)
        
        self.cross_attn_sq = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn_bb = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm_sq = nn.LayerNorm(d_model)
        self.norm_bb = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model*3, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, n_classes)
        )
        
    def forward(self, x):
        return x
        