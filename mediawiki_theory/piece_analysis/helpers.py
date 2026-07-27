import chess
from typing import List

################## CONSTANTS ##################

CENTER_SQUARES = [chess.D4, chess.E4, chess.D5, chess.E5]
EXTENDED_CENTER = [chess.C3, chess.D3, chess.E3, chess.F3,
                   chess.C4, chess.D4, chess.E4, chess.F4,
                   chess.C5, chess.D5, chess.E5, chess.F5,
                   chess.C6, chess.D6, chess.E6, chess.F6]

FILE_NAMES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

_KNIGHT_CENTER_BONUS = [
    -4, -2, -1, -1, -1, -1, -2, -4,
    -2,  0,  1,  1,  1,  1,  0, -2,
    -1,  1,  3,  4,  4,  3,  1, -1,
    -1,  2,  4,  5,  5,  4,  2, -1,
    -1,  2,  4,  5,  5,  4,  2, -1,
    -1,  1,  3,  4,  4,  3,  1, -1,
    -2,  0,  1,  1,  1,  1,  0, -2,
    -4, -2, -1, -1, -1, -1, -2, -4,
]

FIANCHETTO_SQUARES = {
    chess.WHITE: [chess.G2, chess.B2],
    chess.BLACK: [chess.G7, chess.B7],
}
FIANCHETTO_KING_SQUARES = {
    chess.WHITE: {chess.G2: chess.G1, chess.B2: chess.B1},
    chess.BLACK: {chess.G7: chess.G8, chess.B7: chess.B8},
}

LONG_DIAGONAL_A1H8 = [chess.square(i, i) for i in range(8)]
LONG_DIAGONAL_A8H1 = [chess.square(i, 7 - i) for i in range(8)]


################## PRECOMPUTED LOOKUP TABLES ##################

def _build_pawn_attacks() -> List[List[List[int]]]:
    table = [[], []]
    for sq in range(64):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        white_atks, black_atks = [], []
        for df in (-1, 1):
            nf = f + df
            if 0 <= nf < 8:
                if r + 1 < 8:
                    white_atks.append(chess.square(nf, r + 1))
                if r - 1 >= 0:
                    black_atks.append(chess.square(nf, r - 1))
        table[chess.WHITE].append(white_atks)
        table[chess.BLACK].append(black_atks)
    return table

def _build_knight_attacks() -> List[List[int]]:
    offsets = [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]
    return [
        [chess.square(f+df, r+dr)
         for df, dr in offsets if 0 <= f+df < 8 and 0 <= r+dr < 8]
        for sq in range(64)
        for f, r in [(chess.square_file(sq), chess.square_rank(sq))]
    ]

def _build_king_attacks() -> List[List[int]]:
    offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    return [
        [chess.square(f+df, r+dr)
         for df, dr in offsets if 0 <= f+df < 8 and 0 <= r+dr < 8]
        for sq in range(64)
        for f, r in [(chess.square_file(sq), chess.square_rank(sq))]
    ]

def _build_ray_attacks(deltas) -> List[List[int]]:
    result = []
    for sq in range(64):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        attacks = []
        for df, dr in deltas:
            nf, nr = f + df, r + dr
            while 0 <= nf < 8 and 0 <= nr < 8:
                attacks.append(chess.square(nf, nr))
                nf += df; nr += dr
        result.append(attacks)
    return result

def _build_all_rays() -> List[List[List[int]]]:
    directions = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    result = []
    for sq in range(64):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        rays = []
        for df, dr in directions:
            ray, cf, cr = [], f+df, r+dr
            while 0 <= cf < 8 and 0 <= cr < 8:
                ray.append(chess.square(cf, cr))
                cf += df; cr += dr
            if ray:
                rays.append(ray)
        result.append(rays)
    return result

def _build_diagonal_rays() -> List[List[List[int]]]:
    directions = [(-1,-1),(-1,1),(1,-1),(1,1)]
    result = []
    for sq in range(64):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        rays = []
        for df, dr in directions:
            ray, cf, cr = [], f+df, r+dr
            while 0 <= cf < 8 and 0 <= cr < 8:
                ray.append(chess.square(cf, cr))
                cf += df; cr += dr
            if ray:
                rays.append(ray)
        result.append(rays)
    return result

def _build_squares_between() -> List[List[List[int]]]:
    table = [[[] for _ in range(64)] for _ in range(64)]
    for sq1 in range(64):
        f1, r1 = chess.square_file(sq1), chess.square_rank(sq1)
        for sq2 in range(64):
            if sq1 == sq2:
                continue
            f2, r2 = chess.square_file(sq2), chess.square_rank(sq2)
            df = 0 if f1 == f2 else (1 if f2 > f1 else -1)
            dr = 0 if r1 == r2 else (1 if r2 > r1 else -1)
            if df != 0 and dr != 0 and abs(f2 - f1) != abs(r2 - r1):
                continue  # not aligned
            between, cf, cr = [], f1 + df, r1 + dr
            while (cf, cr) != (f2, r2):
                between.append(chess.square(cf, cr))
                cf += df; cr += dr
            table[sq1][sq2] = between
    return table

def _build_pawn_attack_bitboards() -> List[List[int]]:
    table = [[], []]
    for color in (chess.WHITE, chess.BLACK):
        for sq in range(64):
            bb = 0
            for atk in _PAWN_ATTACKS[color][sq]:
                bb |= 1 << atk
            table[color].append(bb)
    return table

def _build_front_spans() -> List[List[List[int]]]:
    table = [[], []]
    for sq in range(64):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        table[chess.WHITE].append([chess.square(f, rr) for rr in range(r + 1, 8)])
        table[chess.BLACK].append([chess.square(f, rr) for rr in range(0, r)])
    return table

def _build_passed_pawn_masks() -> List[List[int]]:
    table = [[], []]
    for sq in range(64):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        adj_files = [ff for ff in (f-1, f, f+1) if 0 <= ff < 8]
        white_mask = 0 
        black_mask = 0
        for ff in adj_files:
            for rr in range(r + 1, 8):
                white_mask |= 1 << chess.square(ff, rr)
            for rr in range(0, r):
                black_mask |= 1 << chess.square(ff, rr)
        table[chess.WHITE].append(white_mask)
        table[chess.BLACK].append(black_mask)
    return table


_PAWN_ATTACKS: List[List[List[int]]] = _build_pawn_attacks()
_KNIGHT_ATTACKS: List[List[int]] = _build_knight_attacks()
_KING_ATTACKS: List[List[int]] = _build_king_attacks()
_BISHOP_ATTACKS: List[List[int]] = _build_ray_attacks([(-1,-1),(-1,1),(1,-1),(1,1)])
_ROOK_ATTACKS: List[List[int]] = _build_ray_attacks([(-1,0),(1,0),(0,-1),(0,1)])
_QUEEN_ATTACKS: List[List[int]] = _build_ray_attacks([(-1,-1),(-1,1),(1,-1),(1,1),(-1,0),(1,0),(0,-1),(0,1)])
_ALL_RAYS: List[List[List[int]]]  = _build_all_rays()
_DIAGONAL_RAYS: List[List[List[int]]]  = _build_diagonal_rays()
_SQUARES_BETWEEN: List[List[List[int]]]  = _build_squares_between()
_PAWN_ATTACK_BB: List[List[int]] = _build_pawn_attack_bitboards()
_FRONT_SPANS: List[List[List[int]]] = _build_front_spans()
_PASSED_PAWN_MASKS: List[List[int]] = _build_passed_pawn_masks()

_IS_DARK = [(chess.square_file(sq) + chess.square_rank(sq)) % 2 == 0 for sq in range(64)]



def pawn_attacks(square: int, color: bool) -> List[int]:
    return _PAWN_ATTACKS[color][square]

def knight_attacks(square: int) -> List[int]:
    return _KNIGHT_ATTACKS[square]

def bishop_attacks(square: int) -> List[int]:
    return _BISHOP_ATTACKS[square]

def rook_attacks(square: int) -> List[int]:
    return _ROOK_ATTACKS[square]

def queen_attacks(square: int) -> List[int]:
    return _QUEEN_ATTACKS[square]

def king_attacks(square: int) -> List[int]:
    return _KING_ATTACKS[square]

def attack(square: int, king: bool) -> List[int]:
    base = (_PAWN_ATTACKS[chess.WHITE][square] + _PAWN_ATTACKS[chess.BLACK][square] +
            _KNIGHT_ATTACKS[square] + _BISHOP_ATTACKS[square] +
            _ROOK_ATTACKS[square] + _QUEEN_ATTACKS[square])
    return base + _KING_ATTACKS[square] if king else base

def front_span(square: int, color: bool) -> List[int]:
    return _FRONT_SPANS[color][square]

def adjacent_files(file: int) -> List[int]:
    return [f for f in (file - 1, file + 1) if 0 <= f < 8]

def is_dark(sq: int) -> bool:
    return _IS_DARK[sq]

def bishop_color(sq: int) -> str:
    return "dark" if _IS_DARK[sq] else "light"

def compute_phase(board: chess.Board) -> float:
    non_pawn_bb = (board.occupied_co[chess.WHITE] | board.occupied_co[chess.BLACK]) \
                  & ~board.pawns & ~board.kings
    return min(1.0, bin(non_pawn_bb).count('1') / 16.0)

def pawn_attack_bitboard(board: chess.Board, color: bool) -> int:
    bb = 0
    for sq in board.pieces(chess.PAWN, color):
        bb |= _PAWN_ATTACK_BB[color][sq]
    return bb

def get_all_rays(sq: int) -> List[List[int]]:
    return _ALL_RAYS[sq]

def squares_between(sq1: int, sq2: int) -> List[int]:
    return _SQUARES_BETWEEN[sq1][sq2]

def file_pawns(board: chess.Board, file: int, color: bool) -> List[int]:
    return [sq for sq in board.pieces(chess.PAWN, color) if chess.square_file(sq) == file]

def is_open_file(board: chess.Board, file: int) -> bool:
    return not file_pawns(board, file, chess.WHITE) and \
           not file_pawns(board, file, chess.BLACK)

def is_semi_open_file(board: chess.Board, file: int, color: bool) -> bool:
    return not file_pawns(board, file, color) and \
           bool(file_pawns(board, file, not color))

def rook_file_ray(board: chess.Board, sq: int, direction: int) -> List[int]:
    file = chess.square_file(sq)
    rank = chess.square_rank(sq)
    squares = []
    r = rank + direction
    while 0 <= r < 8:
        target = chess.square(file, r)
        squares.append(target)
        if board.piece_at(target):
            break
        r += direction
    return squares

def rook_rank_ray(board: chess.Board, sq: int, direction: int) -> List[int]:
    file = chess.square_file(sq)
    rank = chess.square_rank(sq)
    squares = []
    f = file + direction
    while 0 <= f < 8:
        target = chess.square(f, rank)
        squares.append(target)
        if board.piece_at(target):
            break
        f += direction
    return squares

def _find_passed_pawns(board: chess.Board, color: bool) -> List[int]:
    enemy = not color
    enemy_pawn_bb = int(board.pieces(chess.PAWN, enemy))
    passed = []
    for sq in board.pieces(chess.PAWN, color):
        if not (_PASSED_PAWN_MASKS[color][sq] & enemy_pawn_bb):
            passed.append(sq)
    return passed

def is_outpost(board: chess.Board, sq: int, color: bool) -> bool:
    rank = chess.square_rank(sq)
    if color == chess.WHITE and rank < 4:
        return False
    if color == chess.BLACK and rank > 3:
        return False
    protected = any(sq in _PAWN_ATTACKS[color][p]
                    for p in board.pieces(chess.PAWN, color))
    if not protected:
        return False
    enemy_can_attack = any(sq in _PAWN_ATTACKS[not color][p]
                           for p in board.pieces(chess.PAWN, not color))
    return not enemy_can_attack

def get_diagonal_squares(sq: int) -> List[List[int]]:
    return _DIAGONAL_RAYS[sq]