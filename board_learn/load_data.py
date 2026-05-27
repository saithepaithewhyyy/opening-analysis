import struct
from pathlib import Path
import os

import pandas as pd
import numpy as np


BOARD_FORMAT = '<12QQI2BxxQ'
BOARD_SIZE = struct.calcsize(BOARD_FORMAT)


def parse_board(raw):
    vals = struct.unpack(BOARD_FORMAT, raw)
    return {
        'bb': [[vals[r * 6 + c] for c in range(6)] for r in range(2)],
        'occupied': vals[12],
        'turn': vals[13],
        'castling': vals[14],
        'ep_sq': vals[15],
        'zobrist': vals[16],
    }


def read_str(f):
    length = struct.unpack('<I', f.read(4))[0]
    return f.read(length).decode('utf-8')

def flatten_board(board):
    row = {
        'board_occupied': board['occupied'],
        'board_turn': board['turn'],
        'board_castling': board['castling'],
        'board_ep_sq': board['ep_sq'],
        'zobrist': board['zobrist'],
    }

    for side in range(2):
        for piece in range(6):
            row[f'board_bb_{side}_{piece}'] = board['bb'][side][piece]

    return row

def parse_data(path, output_dir='.'):
    path = Path(path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    roots_rows = []
    reach_index_rows = []
    board_zh_rows = []

    with open(path, 'rb') as f:
        nroots = struct.unpack('<Q', f.read(8))[0]

        for root_id in range(nroots):
            eco = read_str(f)
            name = read_str(f)
            board = parse_board(f.read(BOARD_SIZE))
            prior = struct.unpack('<d', f.read(8))[0]

            row = {
                'root_id': root_id,
                'eco': eco,
                'name': name,
                'prior': prior,
            }
            row.update(flatten_board(board))
            roots_rows.append(row)

        n = struct.unpack('<Q', f.read(8))[0]

        for reach_id in range(n):
            zh = struct.unpack('<Q', f.read(8))[0]
            ne = struct.unpack('<I', f.read(4))[0]

            for entry_id in range(ne):
                reach_index_rows.append({
                    'reach_id': reach_id,
                    'zobrist': zh,
                    'entry_id': entry_id,
                    'eco': read_str(f),
                    'prob': struct.unpack('<f', f.read(4))[0],
                    'path_length': struct.unpack('<B', f.read(1))[0],
                })

        nboard = struct.unpack('<Q', f.read(8))[0]

        for board_id in range(nboard):
            board = parse_board(f.read(BOARD_SIZE))

            row = {
                'board_id': board_id,
            }
            row.update(flatten_board(board))
            board_zh_rows.append(row)

    roots_df = pd.DataFrame(roots_rows)
    reach_index_df = pd.DataFrame(reach_index_rows)
    board_zh_df = pd.DataFrame(board_zh_rows)

    roots_df.to_parquet(output_dir / 'roots.parquet', index=False)
    reach_index_df.to_parquet(output_dir / 'reach_index.parquet', index=False)
    board_zh_df.to_parquet(output_dir / 'board_zh.parquet', index=False)

    return roots_df, reach_index_df, board_zh_df

def load_data(folder_path='.'):
    required = ['roots.parquet', 'reach_index.parquet', 'board_zh.parquet']
    if not all(f in os.listdir(folder_path) for f in required):
        parse_data(Path(folder_path) / "index.bin", output_dir=folder_path)

    roots_df = pd.read_parquet(folder_path + '/roots.parquet')
    reach_index_df = pd.read_parquet(folder_path + '/reach_index.parquet')
    board_zh_df = pd.read_parquet(folder_path + '/board_zh.parquet')

    eco_classes = sorted(roots_df['eco'].unique())
    eco_to_idx = {eco: i for i, eco in enumerate(eco_classes)}
    n_classes = len(eco_classes)

    target_map = {}
    for zh, group in reach_index_df.groupby('zobrist'):
        vec = np.zeros(n_classes, dtype=np.float32)
        for _, row in group.iterrows():
            idx = eco_to_idx.get(row['eco'])
            if idx is not None:
                vec[idx] = row['prob']
        target_map[zh] = vec

    bb_cols = ['board_occupied'] + [f'board_bb_{side}_{piece}' for side in range(2) for piece in range(6)]
    
    bb_uint64 = np.stack([
        board_zh_df[col].to_numpy(dtype=np.uint64) for col in bb_cols
    ], axis=1)
    bb_bytes = bb_uint64.view(np.uint8).reshape(len(board_zh_df), 13, 8)
    bitboards_all = np.unpackbits(bb_bytes, axis=2, bitorder='little').astype(np.float32)

    castling = board_zh_df['board_castling'].to_numpy(dtype=np.uint32)
    castling_bytes = castling.view(np.uint8).reshape(len(castling), 4)
    castling_bits = np.unpackbits(castling_bytes, axis=1, bitorder='little')[:, :4].astype(np.float32)
    
    ep_sq = board_zh_df['board_ep_sq'].to_numpy(dtype=np.uint16)
    ep_present = (ep_sq < 64).astype(np.float32)
    ep_file_oh= np.zeros((len(ep_sq), 8), dtype=np.float32)
    valid = ep_sq < 64
    ep_file_oh[valid, ep_sq[valid] % 8] = 1.0

    turn = board_zh_df['board_turn'].to_numpy(dtype=np.float32)

    scalars_all = np.concatenate([turn[:, None], ep_present[:, None], castling_bits, ep_file_oh], axis=1)

    zobrists = board_zh_df['zobrist'].values
    targets_all = np.stack([
        target_map.get(int(zh), np.zeros(n_classes, dtype=np.float32))
        for zh in zobrists
    ])

    return zobrists, bitboards_all, scalars_all, targets_all, eco_classes

if __name__ == "__main__":
    zobrists, bitboards_all, scalars_all, targets_all, eco_classes = load_data(folder_path="..")
