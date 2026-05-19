import torch
import torch.nn as nn

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
                n_flat = 7, 
                d_model = 256, 
                nhead = 8, 
                num_layers = 6, 
                d_ff = 1024,
                d_hidden = 2048,
                dropout = 0.1):
        
        super().__init__()
        
        self.scalar = nn.Linear(n_flat, d_model)
        
        self.sq_linear = nn.Linear(n_bb-1, d_model)
        self.rank_embed = nn.Embedding(8, d_model // 2)
        self.file_embed = nn.Embedding(8, d_model // 2)
        
        self.bb_linear = nn.Linear(n_sq, d_model)
        self.bb_embed = nn.Embedding(n_bb, d_model)

        self.sq_transformer = nn.TransformerEncoder(                                                                             
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_ff, batch_first=True, norm_first=False),
            num_layers=num_layers                                                                                                
        )

        self.bb_transformer = nn.TransformerEncoder(                                                                             
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_ff, batch_first=True, norm_first=False),
            num_layers=num_layers                                                                                                
        )
        
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
        
        self.softmax = nn.Softmax(dim=-1)
        
    def rank_file_encode(self, device):
        rank = torch.arange(8, device=device).repeat_interleave(8)
        File = torch.arange(8, device=device).repeat(8)
        
        embed = torch.cat([self.rank_embed(rank), self.file_embed(File)], dim=-1)
        return embed
        
    def forward(self, bitboards, meta):
        # bitboards shape - (b, 13, 64)
        # meta shape - (b, n_flat)
        
        meta = self.scalar(meta)

        sq = bitboards[:, 1:, :].permute(0, 2, 1)
        sq = self.sq_linear(sq)
        sq += self.rank_file_encode(device=sq.device)
        sq = self.sq_transformer(sq)
        
        bb = bitboards[:, :, :]
        bb = self.bb_linear(bb)
        bb += self.bb_embed(torch.arange(13, device=bb.device))
        bb = self.bb_transformer(bb)
        
        sq = self.norm_sq(sq + self.cross_attn_sq(query=sq, key=bb, value=bb)[0])
        bb = self.norm_bb(bb + self.cross_attn_bb(query=bb, key=sq, value=sq)[0])
        
        sq_out = sq.mean(dim=1)
        bb_out = bb.mean(dim=1)
        
        op = torch.cat([sq_out, bb_out, meta], dim=-1)
        op = self.ff(op)
        
        return self.softmax(op) 
        
