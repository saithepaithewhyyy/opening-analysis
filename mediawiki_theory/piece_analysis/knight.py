import chess
from typing import Dict, List, Tuple, Optional
import piece_analysis.helpers as hp

def analyze_knight_mobility(board: chess.Board, color: bool) -> Dict:

    knights = list(board.pieces(chess.KNIGHT, color))
    data = []
    total_mobility = 0
    total_attacks = 0

    for sq in knights:
        attacks = hp.knight_attacks(sq)
        effective = [s for s in attacks
                     if not (board.piece_at(s) and board.piece_at(s).color == color)]
        enemy_attacked = [s for s in attacks
                          if board.piece_at(s) and board.piece_at(s).color != color]

        mob = len(effective)
        total_mobility += mob
        total_attacks += len(attacks)

        data.append({
            "square": sq,
            "name": chess.square_name(sq),
            "attack_count": len(attacks),
            "mobility": mob,
            "attacks_enemy": len(enemy_attacked),
            "enemy_attacked_squares": [chess.square_name(s) for s in enemy_attacked],
        })

    avg_mobility = total_mobility / len(knights) if knights else 0.0
    
    return {
        "count": len(knights),
        "per_knight": data,
        "total_mobility": total_mobility,
        "avg_mobility": avg_mobility,
    }


def analyze_knight_centralization(board: chess.Board, color: bool) -> Dict:
    knights = list(board.pieces(chess.KNIGHT, color))
    data = []
    total_bonus = 0
    rim_knights = 0

    for sq in knights:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        bonus = hp._KNIGHT_CENTER_BONUS[sq]
        on_rim = f in [0, 7] or r in [0, 7]
        in_center = sq in hp.CENTER_SQUARES
        in_extended = sq in hp.EXTENDED_CENTER

        total_bonus += bonus
        if on_rim:
            rim_knights += 1

        data.append({
            "square": sq,
            "name": chess.square_name(sq),
            "center_bonus": bonus,
            "on_rim": on_rim,
            "in_center": in_center,
            "in_extended_center": in_extended,
        })

    return {
        "total_centralization": total_bonus,
        "rim_knights": rim_knights,
        "per_knight": data,
    }


def analyze_knight_coordination(board: chess.Board, color: bool) -> Dict:
    knights = list(board.pieces(chess.KNIGHT, color))
    enemy_king = board.king(not color)

    mutual_defense = 0
    shared_attack_squares = 0
    king_zone_attacks = 0

    if len(knights) >= 2:
        kn_attacks = {sq: set(hp.knight_attacks(sq)) for sq in knights}

        for i, sq1 in enumerate(knights):
            for sq2 in knights[i+1:]:
                if sq2 in kn_attacks[sq1] or sq1 in kn_attacks[sq2]:
                    mutual_defense += 1

        all_attack_sets = list(kn_attacks.values())
        if len(all_attack_sets) >= 2:
            shared = all_attack_sets[0]
            for s in all_attack_sets[1:]:
                shared = shared & s
            shared_attack_squares = len(shared)

    if enemy_king:
        king_zone = set(hp.knight_attacks(enemy_king)) | {enemy_king}
        for sq in knights:
            if set(hp.knight_attacks(sq)) & king_zone:
                king_zone_attacks += 1

    return {
        "knight_count": len(knights),
        "mutual_defense_pairs": mutual_defense,
        "shared_attack_squares": shared_attack_squares,
        "king_zone_attackers": king_zone_attacks,
    }

def full_knight_evaluation(board: chess.Board) -> Dict:
    result = {}


    for color, name in [(chess.WHITE, "white"), (chess.BLACK, "black")]:
        mobility = analyze_knight_mobility(board, color)
        central = analyze_knight_centralization(board, color)
        coord = analyze_knight_coordination(board, color)

        result[name] = {
            "knight_count": mobility["count"],
            "total_mobility": mobility["total_mobility"],
            "avg_mobility": round(mobility["avg_mobility"], 2),
            "per_knight_mobility": [
                {"sq": d["name"], "mobility": d["mobility"],
                 "attacks_enemy": d["attacks_enemy"]}
                for d in mobility["per_knight"]
            ],
            "centralization_score": central["total_centralization"],
            "rim_knights": central["rim_knights"],
            "mutual_defense_pairs": coord["mutual_defense_pairs"],
            "shared_attack_squares": coord["shared_attack_squares"],
            "king_zone_attackers": coord["king_zone_attackers"],
        }

    return result


