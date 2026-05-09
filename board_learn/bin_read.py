import struct

BOARD_FORMAT = '<12QQI2BQ'
BOARD_SIZE = struct.calcsize(BOARD_FORMAT)

def parse_board(raw):
    vals = struct.unpack(BOARD_FORMAT, raw)
    return {
        'bb': [[vals[r*6 + c] for c in range(6)] for r in range(2)],
        'occupied': vals[12],
        'turn': vals[13],
        'castling': vals[14],
        'ep_sq': vals[15],
        'zobrist': vals[16],
    }

def read_str(f):
    length = struct.unpack('<I', f.read(4))[0]
    return f.read(length).decode('utf-8')

def load_index(path):
    with open(path, 'rb') as f:

        nroots = struct.unpack('<Q', f.read(8))[0]
        roots = []
        for _ in range(nroots):
            roots.append({
                'eco': read_str(f),
                'name': read_str(f),
                'board': parse_board(f.read(BOARD_SIZE)),
                'prior': struct.unpack('<d', f.read(8))[0],
            })

        n = struct.unpack('<Q', f.read(8))[0]
        reach_index = {}
        for _ in range(n):
            zh = struct.unpack('<Q', f.read(8))[0]
            ne = struct.unpack('<I', f.read(4))[0]
            reach_index[zh] = [{
                'eco': read_str(f),
                'prob': struct.unpack('<f', f.read(4))[0],
                'path_length': struct.unpack('<B', f.read(1))[0],
            } for _ in range(ne)]

        nboard = struct.unpack('<Q', f.read(8))[0]
        board_zh = [parse_board(f.read(BOARD_SIZE)) for _ in range(nboard)]

    return roots, reach_index, board_zh