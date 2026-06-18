import pandas as pd
import numpy as np

import torch

PIECE_INDEX = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
}

def sparse_kl_loss(log_q, sparse_targets):
    loss = 0.0
    for b, (idxs, probs) in enumerate(sparse_targets):

        idxs = torch.as_tensor(
            idxs,
            dtype=torch.long,
            device=log_q.device
        )

        probs = torch.as_tensor(
            probs,
            dtype=log_q.dtype,
            device=log_q.device
        )

        # KL Div Loss anyways. For sparse, we dont even care about the zero classes do we
        loss += (probs * (torch.log(probs)- log_q[b, idxs])).sum()

    return loss / len(sparse_targets)


def pos_to_bb(pos, form="fen"):
    bbs = np.zeros(13, dtype=np.uint64)
    squares = {}

    if form == 'fen':
        board_part = pos.split()[0]
        rank, file = 7, 0
        for ch in board_part:
            if ch == '/':
                rank -= 1; file = 0
            elif ch.isdigit():
                file += int(ch)
            else:
                squares[rank * 8 + file] = ch
                file += 1
    else:
        raise ValueError(f"only supports fen for now :<")

    for sq, piece in squares.items():
        if piece not in PIECE_INDEX:
            continue
        bit = np.uint64(1) << np.uint64(sq)
        bbs[0] |= bit
        bbs[PIECE_INDEX[piece] + 1] |= bit

    bb_bytes = bbs.view(np.uint8).reshape(13, 8)

    bb = np.unpackbits(
        bb_bytes,
        axis=1,
        bitorder="little"
    ).astype(np.uint8)

    return bb


def inference_features(pos, form="fen"):
    bb = pos_to_bb(pos, form)
    
    if form == "fen":
        parts = pos.split()

        turn = 0
        if parts[1] == "w":
            turn = 1

        castling = parts[2]
        wk = int("K" in castling)
        wq = int("Q" in castling)
        bk = int("k" in castling)
        bq = int("q" in castling)

        ep = parts[3]
        ep_file_oh = np.zeros(8, dtype=np.uint8)

        if ep != "-":
            file = ord(ep[0]) - ord("a")
            ep_file_oh[file] = 1

        scalars = np.concatenate([
            np.array([
                turn,
                int(ep!="-"),
                wk, wq, bk, bq
            ]),
            ep_file_oh
        ])
    else:
        raise ValueError(f"only supports fen for now :<")
        
    return bb, scalars

def plot_grads(piece_grad):
        return
