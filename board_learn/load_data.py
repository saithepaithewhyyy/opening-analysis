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

def clean_data(path, folder_path='.'):
    if ['roots.csv', 'reach_index.csv', 'board_zh.csv'] not in os.listdir(folder_path):
        load_data('index.bin')
        
    roots_df = pd.read_csv(folder_path + '/roots.csv')
    reach_index_df = pd.read_csv(folder_path + '/reach_index.csv')
    board_zh_df = pd.read_csv(folder_path + '/board_zh.csv')
    
    eco_cols = roots_df['eco']
    board_zh_df[eco_cols] = 0.0
  
    for _, row in board_zh_df.iterrows():
        prob_values_df = reach_index_df[row['zobrist']]
        for _, row2 in prob_values_df.iterrows():
            row[row2['eco']] = row2['prob']
            
    
