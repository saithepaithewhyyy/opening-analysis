import struct
from pathlib import Path

import pandas as pd
import numpy as np
import os


BOARD_FORMAT = '<12QQI2BQ'
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

def load_data(path, output_dir='.', save_format='csv'):
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

    save_format = save_format.lower()
    roots_df.to_csv(output_dir / 'roots.csv', index=False)
    reach_index_df.to_csv(output_dir / 'reach_index.csv', index=False)
    board_zh_df.to_csv(output_dir / 'board_zh.csv', index=False)

    return roots_df, reach_index_df, board_zh_df

def clean_data(folder_path='.'):
    required = ['roots.csv', 'reach_index.csv', 'board_zh.csv']
    if not all(f in os.listdir(folder_path) for f in required):
        load_data('index.bin', output_dir=folder_path)

    roots_df = pd.read_csv(folder_path + '/roots.csv')
    reach_index_df = pd.read_csv(folder_path + '/reach_index.csv')
    board_zh_df = pd.read_csv(folder_path + '/board_zh.csv')

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
    bb_uint64 = board_zh_df[bb_cols].values.astype(np.uint64)
    bb_bytes = bb_uint64.view(np.uint8).reshape(len(board_zh_df), 13, 8)
    bitboards_all = np.unpackbits(bb_bytes, axis=2, bitorder='little').astype(np.float32)

    scalars_all = board_zh_df[['board_castling', 'board_ep_sq', 'board_turn']].values.astype(np.float32)

    zobrists = board_zh_df['zobrist'].astype(np.int64).values
    targets_all = np.stack([
        target_map.get(int(zh), np.zeros(n_classes, dtype=np.float32))
        for zh in zobrists
    ])

    records = list(zip(zobrists.tolist(), bitboards_all, scalars_all, targets_all))
    return records, eco_classes
