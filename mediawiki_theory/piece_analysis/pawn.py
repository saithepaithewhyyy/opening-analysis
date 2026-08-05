import chess
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from . import helpers as hp
from . import pawn_structures as ps

def analyze_classical_structure(board: chess.Board, color: bool) -> Dict:
    pawns = list(board.pieces(chess.PAWN, color))
    enemy_pawns = list(board.pieces(chess.PAWN, not color))
    enemy = not color

    file_map: Dict[int, List[int]] = {f: [] for f in range(8)}
    for p in pawns:
        file_map[chess.square_file(p)].append(p)

    # DOUBLED AND TRIPLED PAWNS
    doubled = sum(len(v) - 1 for v in file_map.values() if len(v) > 1)
    tripled = sum(len(v) - 2 for v in file_map.values() if len(v) > 2)

    # ISOLATED PAWNS
    isolated = 0
    for f in range(8):
        if file_map[f]:
            has_adj = any(file_map.get(f + df) for df in [-1, 1])
            if not has_adj:
                isolated += len(file_map[f])

    # PAWN ISLANDS
    islands = 0
    in_island = False
    for f in range(8):
        if file_map[f] and not in_island:
            islands += 1
            in_island = True
        elif not file_map[f]:
            in_island = False

    # ADJACENT
    phalanx = 0
    for p in pawns:
        pr = chess.square_rank(p)
        pf = chess.square_file(p)
        if any(
            chess.square_rank(q) == pr and abs(chess.square_file(q) - pf) == 1
            for q in pawns
        ):
            phalanx += 1
    phalanx //= 2  # each pair counted once

    # PAWN CHAINS
    chains = 0
    for p in pawns:
        if any(p in hp.pawn_attacks(q, color) for q in pawns if q != p):
            chains += 1

    # HANGING PAWNS (NEED TO LOOK INTO HANGING PAWNS)
    hanging = 0
    c_pawns = file_map.get(2, [])
    d_pawns = file_map.get(3, [])
    if c_pawns and d_pawns:
        if not file_map.get(1) and not file_map.get(4):
            hanging = len(c_pawns) + len(d_pawns)

    # BACKWARD PAWNS
    backward = []
    for p in pawns:
        pf = chess.square_file(p)
        pr = chess.square_rank(p)
        fwd_rank = pr + 1 if color == chess.WHITE else pr - 1
        if not (0 <= fwd_rank < 8):
            continue
        fwd_sq = chess.square(pf, fwd_rank)

        supported = False
        for q in pawns:
            if q == p:
                continue
            qf = chess.square_file(q)
            qr = chess.square_rank(q)
            if abs(qf - pf) == 1:
                if color == chess.WHITE and qr <= pr:
                    supported = True
                if color == chess.BLACK and qr >= pr:
                    supported = True

        if not supported:
            if any(fwd_sq in hp.pawn_attacks(ep, enemy) for ep in enemy_pawns):
                backward.append(p)

    return {
        "doubled": doubled,
        "tripled": tripled,
        "isolated": isolated,
        "islands": islands,
        "phalanx": phalanx,
        "chains": chains,
        "hanging": hanging,
        "backward": backward,
        "file_map": file_map,
    }


def analyze_pawn_majorities(board: chess.Board, color: bool) -> Dict:
    enemy = not color
    our = list(board.pieces(chess.PAWN, color))
    their = list(board.pieces(chess.PAWN, enemy))

    our_q = sum(1 for p in our if chess.square_file(p) <= 3)
    our_k = sum(1 for p in our if chess.square_file(p) >= 4)
    their_q = sum(1 for p in their if chess.square_file(p) <= 3)
    their_k = sum(1 for p in their if chess.square_file(p) >= 4)

    return {
        "queenside_majority": our_q > their_q,
        "kingside_majority": our_k > their_k,
        "queenside_diff": our_q - their_q,
        "kingside_diff": our_k - their_k,
    }


# PAWN LEVERS: POTENTIAL BREAKS, WEAKNESS CREATION - IDENTIFIES LEVERS IN POSITION. 
# PAWN SACRIFICES?

def classify_pawn_levers(board: chess.Board, color: bool) -> List[Dict]:
    enemy = not color
    enemy_king = board.king(enemy)
    levers = []

    for sq in board.pieces(chess.PAWN, color):
        for atk in hp.pawn_attacks(sq, color):
            target = board.piece_at(atk)
            if target and target.piece_type == chess.PAWN and target.color == enemy:
                rank = chess.square_rank(sq)
                atk_file = chess.square_file(atk)

                # Good break: advanced pawn, opens center or files toward enemy king
                is_central = atk_file in [2, 3, 4, 5]
                is_advanced = (color == chess.WHITE and rank >= 4) or \
                              (color == chess.BLACK and rank <= 3)
                toward_king = enemy_king and abs(atk_file - chess.square_file(enemy_king)) <= 2

                lever_type = "good" if (is_advanced or is_central or toward_king) else "bad"
                lever_subtype = "advanced pawn" if is_advanced else "central pawn break" if is_central else "king pawn break"

                levers.append({
                    "from": sq,
                    "to": atk,
                    "type": f"{lever_type} lever with {lever_subtype}",
                    "opens_file": True,
                })

    return levers

# PAWN SHIELD
def analyze_pawn_shield(board: chess.Board, color: bool) -> Dict:
    king = board.king(color)

    kf = chess.square_file(king)
    kr = chess.square_rank(king)
    shield_pawns = 0
    open_files = []

    for df in [-1, 0, 1]:
        f = kf + df
        if not 0 <= f < 8:
            continue

        r1 = kr + 1 if color == chess.WHITE else kr - 1
        r2 = kr + 2 if color == chess.WHITE else kr - 2

        has_r1 = 0 <= r1 < 8 and any(
            chess.square_file(p) == f and chess.square_rank(p) == r1
            for p in board.pieces(chess.PAWN, color)
        )
        has_r2 = 0 <= r2 < 8 and any(
            chess.square_file(p) == f and chess.square_rank(p) == r2
            for p in board.pieces(chess.PAWN, color)
        )

        if has_r1:
            shield_pawns += 2
        elif has_r2:
            shield_pawns += 1
        else:
            open_files.append(hp.FILE_NAMES[f])

    return {"shield_count": shield_pawns, "open_files": open_files}


# WEAK SQUARES & OUTPOSTS
def compute_weak_squares(board: chess.Board, color: bool) -> Dict:
    enemy = not color
    our_attack_bb = hp.pawn_attack_bitboard(board, color)
    enemy_attack_bb = hp.pawn_attack_bitboard(board, enemy)

    weak_squares = []
    outpost_squares = []

    for sq in chess.SQUARES:
        rank = chess.square_rank(sq)

        in_relevant_zone = (
            (color == chess.WHITE and rank >= 3) or
            (color == chess.BLACK and rank <= 4)
        )
        if not in_relevant_zone:
            continue

        not_defended = not (our_attack_bb & (1 << sq))
        if not_defended:
            weak_squares.append(sq)

            # Outpost: also not attackable by enemy pawn
            not_enemy_attackable = not (enemy_attack_bb & (1 << sq))
            in_enemy_half = (color == chess.WHITE and rank >= 4) or \
                            (color == chess.BLACK and rank <= 3)

            if not_enemy_attackable and in_enemy_half:
                outpost_squares.append(sq)

    # Color complex weakness
    our_bishops = list(board.pieces(chess.BISHOP, color))
    if our_bishops:
        weak_dark = sum(1 for s in weak_squares if hp.is_dark(s))
        weak_light = sum(1 for s in weak_squares if not hp.is_dark(s))
    else:
        weak_dark = sum(1 for s in weak_squares if hp.is_dark(s))
        weak_light = sum(1 for s in weak_squares if not hp.is_dark(s))

    return {
        "weak_squares": weak_squares,
        "weak_count": len(weak_squares),
        "outpost_squares": outpost_squares,
        "outpost_count": len(outpost_squares),
        "weak_dark": weak_dark,
        "weak_light": weak_light,
    }

# PAWN PRESSURE
def compute_tension_grid(board: chess.Board) -> Dict:
    tension_map = [0] * 64
    tension_pairs = []

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.piece_type == chess.PAWN:
            for atk in hp.pawn_attacks(sq, piece.color):
                target = board.piece_at(atk)
                if target and target.piece_type == chess.PAWN and target.color != piece.color:
                    tension_map[sq] = 1
                    tension_map[atk] = 1
                    tension_pairs.append((sq, atk))

    return {
        "tension_squares": sum(tension_map),
        "tension_pairs": tension_pairs,
    }


def detect_minority_attack(board: chess.Board, color: bool) -> Dict:
    enemy = not color
    our = list(board.pieces(chess.PAWN, color))
    their = list(board.pieces(chess.PAWN, enemy))

    results = {}
    for side, files in [("queenside", range(4)), ("kingside", range(4, 8))]:
        our_count = sum(1 for p in our if chess.square_file(p) in files)
        their_count = sum(1 for p in their if chess.square_file(p) in files)
        results[f"{side}_minority_attack"] = our_count < their_count and our_count > 0
        results[f"{side}_our"] = our_count
        results[f"{side}_their"] = their_count

    return results


def pawn_structure_symmetry(board: chess.Board) -> Dict:
    white_files = sorted(chess.square_file(p) for p in board.pieces(chess.PAWN, chess.WHITE))
    black_files = sorted(chess.square_file(p) for p in board.pieces(chess.PAWN, chess.BLACK))

    count_diff = abs(len(white_files) - len(black_files))
    file_symmetry = sum(1 for f in white_files if f in black_files)

    return {
        "count_diff": count_diff,
        "file_symmetry": file_symmetry,
        "white_pawn_count": len(white_files),
        "black_pawn_count": len(black_files),
    }

def score_side(pawns, required, optional, forbidden):
    req_total = required.bit_count()
    req_hit = (pawns & required).bit_count()

    opt_total = optional.bit_count()
    opt_hit = (pawns & optional).bit_count()

    forbidden_present = (pawns & forbidden).bit_count()

    score = (
        0.7 * (req_hit / max(req_total, 1)) +
        0.3 * (opt_hit / max(opt_total, 1))
    )

    score -= 0.25 * forbidden_present

    return max(score, 0.0)

def comp_struct(board: chess.Board, structure):
    black_pawns = int(board.occupied_co[chess.BLACK] & board.pawns)
    white_pawns = int(board.occupied_co[chess.WHITE] & board.pawns)
    
    white_score = score_side(
        white_pawns,
        structure.white_required,
        structure.white_optional,
        structure.white_forbidden,
    )

    black_score = score_side(
        black_pawns,
        structure.black_required,
        structure.black_optional,
        structure.black_forbidden,
    )

    return (white_score + black_score) / 2

def full_pawn_evaluation(board: chess.Board) -> Dict:
    result = {}

    phase = hp.compute_phase(board)
    result["endgame_factor"] = 1.0 - phase

    # Shared features
    tension = compute_tension_grid(board)
    result["tension"] = tension

    symmetry = pawn_structure_symmetry(board)
    result["symmetry"] = symmetry
    
    best_structure = max(
        ps.PAWN_STRUCTURES.values(),
        key=lambda s: comp_struct(board, s)
    )   

    best_score = comp_struct(board, best_structure)
    
    check = (best_score > 0.5)
    result["pawn_structure"] = best_structure.name if check else ""
    result["pawn_structure_themes"] = best_structure.themes if check else ()
    result["pawn_structure_character"] = best_structure.character if check else ""
    result["pawn_structure_related_openings"] = best_structure.openings if check else ()
    result["pawn_structure_similarity"] = best_score

    for color, name in [(chess.WHITE, "white"), (chess.BLACK, "black")]:

        classical = analyze_classical_structure(board, color)
        majorities = analyze_pawn_majorities(board, color)
        levers = classify_pawn_levers(board, color)
        shield = analyze_pawn_shield(board, color)
        weak = compute_weak_squares(board, color)
        minority = detect_minority_attack(board, color)

        result[name] = {
            # Classical
            "doubled_pawns": classical["doubled"],
            "tripled_pawns": classical["tripled"],
            "isolated_pawns": classical["isolated"],
            "pawn_islands": classical["islands"],
            "phalanx_pawns": classical["phalanx"],
            "pawn_chains": classical["chains"],
            "pawn_chain_type": "",
            "hanging_pawns": classical["hanging"],
            "backward_pawn_count": len(classical["backward"]),
            "backward_squares": classical["backward"],

            # Majorities
            **majorities,

            # Levers
            "good_pawn_levers": sum(1 for l in levers if l["type"] == "good"),
            "bad_pawn_levers": sum(1 for l in levers if l["type"] == "bad"),
            "pawn_levers": levers,

            # Storm & Shield
            "pawn_shield_count": shield["shield_count"],
            "king_open_files": shield["open_files"],

            # Weak squares & outposts
            "weak_squares_count": weak["weak_count"],
            "outpost_squares_count": weak["outpost_count"],
            "weak_dark_squares": weak["weak_dark"],
            "weak_light_squares": weak["weak_light"],
            "outpost_squares": [chess.square_name(s) for s in weak["outpost_squares"][:10]],

            **minority,
        }

    return result
