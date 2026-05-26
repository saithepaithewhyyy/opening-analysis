import pandas as pd
import numpy as np

PIECE_INDEX = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
}

def pos_to_bb(pos, type):
    bbs = np.zeros(13, dtype=np.uint64)
    squares = {}

    if type == 'fen':
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

    return bbs
    