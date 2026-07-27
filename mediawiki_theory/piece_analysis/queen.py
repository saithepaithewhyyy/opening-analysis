import chess
from typing import Dict, List, Tuple, Optional
import piece_analysis.helpers as hp

def analyze_queen_mobility(board: chess.Board, color: bool) -> Dict:
    queens = list(board.pieces(chess.QUEEN, color))
    data = []
    total_mobility = 0

    for sq in queens:
        attacks = list(board.attacks(sq))
        reachable = [s for s in attacks
                     if not (board.piece_at(s) and board.piece_at(s).color == color)]
        enemy_attacked = [s for s in reachable
                          if board.piece_at(s) and board.piece_at(s).color != color]

        f, r = chess.square_file(sq), chess.square_rank(sq)
        diagonal_mob   = sum(1 for s in reachable
                             if chess.square_file(s) != f and chess.square_rank(s) != r)
        orthogonal_mob = sum(1 for s in reachable
                             if chess.square_file(s) == f or chess.square_rank(s) == r)

        attacks_center = sum(1 for s in reachable if s in hp.CENTER_SQUARES)
        mob = len(reachable)
        total_mobility += mob
        
        data.append({
            "square": sq,
            "name": chess.square_name(sq),
            "mobility": mob,
            "diagonal_mobility": diagonal_mob,
            "orthogonal_mobility": orthogonal_mob,
            "attacks_center": attacks_center,
            "attacks_enemy": len(enemy_attacked),
        })

    return {
        "count": len(queens),
        "per_queen": data,
        "total_mobility": total_mobility,
    }


def analyze_queen_development(board: chess.Board, color: bool) -> Dict:
    phase = hp.compute_phase(board)
    queens = list(board.pieces(chess.QUEEN, color))
    home_rank = 0 if color == chess.WHITE else 7

    minors_on_home_rank = sum(
        1 for pt in [chess.KNIGHT, chess.BISHOP]
        for sq in board.pieces(pt, color)
        if chess.square_rank(sq) == home_rank
    )

    early_development_penalty = 0
    per_queen = []

    for sq in queens:
        rank = chess.square_rank(sq)
        advanced = rank != home_rank

        penalty = 0
        if advanced and phase > 0.7: 
            penalty = minors_on_home_rank * 4  
            
        early_development_penalty += penalty
        per_queen.append({
            "square": chess.square_name(sq),
            "on_home_rank": rank == home_rank,
            "penalty": penalty,
        })

    return {
        "minors_on_home_rank": minors_on_home_rank,
        "early_development_penalty": early_development_penalty,
        "per_queen": per_queen,
        "is_early_game": phase > 0.7,
    }


def analyze_queen_safety(board: chess.Board, color: bool) -> Dict:
    enemy = not color
    queens = list(board.pieces(chess.QUEEN, color))
    data = []
    total_exposure = 0

    for sq in queens:
        attackers = list(board.attackers(enemy, sq))
        defenders = list(board.attackers(color, sq))

        attacker_types = [board.piece_at(a).piece_type for a in attackers if board.piece_at(a)]
        minor_attacks = sum(1 for t in attacker_types if t in [chess.KNIGHT, chess.BISHOP])
        pawn_attacks  = sum(1 for t in attacker_types if t == chess.PAWN)

        escape = [s for s in board.attacks(sq)
                  if not (board.piece_at(s) and board.piece_at(s).color == color)
                  and not board.is_attacked_by(enemy, s)]

        is_trapped = len(escape) <= 2 and len(attackers) > 0
        exposure = len(attackers) * 3 + minor_attacks * 5 + pawn_attacks * 8 - len(defenders) * 2

        total_exposure += max(0, exposure)

        data.append({
            "square": chess.square_name(sq),
            "attackers": len(attackers),
            "defenders": len(defenders),
            "minor_attacks": minor_attacks,
            "pawn_attacks": pawn_attacks,
            "escape_squares": len(escape),
            "is_trapped": is_trapped,
            "exposure_score": max(0, exposure),
        })

    return {
        "per_queen": data,
        "total_exposure": total_exposure,
        "trapped_queens": sum(1 for d in data if d["is_trapped"]),
    }

def analyze_queen_activity(board: chess.Board, color: bool) -> Dict:
    queens = list(board.pieces(chess.QUEEN, color))
    data = []
    total_activity = 0.0

    for sq in queens:
        f, r = chess.square_file(sq), chess.square_rank(sq)

        center_dist = ((f - 3.5) ** 2 + (r - 3.5) ** 2) ** 0.5
        centrality = max(0, 5 - center_dist)

        advancement = r if color == chess.WHITE else 7 - r

        mob = len([s for s in board.attacks(sq)
                   if not (board.piece_at(s) and board.piece_at(s).color == color)])

        total_squares_attacked = len(list(board.attacks(sq)))

        activity = centrality * 2.0 + advancement * 0.5 + mob * 0.5
        total_activity += activity

        data.append({
            "square": chess.square_name(sq),
            "centrality": round(centrality, 2),
            "advancement": advancement,
            "mobility": mob,
            "board_coverage": total_squares_attacked,
            "activity_index": round(activity, 2),
        })

    return {
        "per_queen": data,
        "total_activity": round(total_activity, 2),
    }

def analyze_queen_pins(board: chess.Board, color: bool) -> Dict:
    enemy = not color
    queens = list(board.pieces(chess.QUEEN, color))
    enemy_king = board.king(enemy)

    pins = []
    skewers = []

    for qsq in queens:
        for ray in hp.get_all_rays(qsq):
            first_piece = None
            second_piece = None

            for s in ray:
                piece = board.piece_at(s)
                if piece:
                    if first_piece is None:
                        first_piece = (s, piece)
                    else:
                        second_piece = (s, piece)
                        break

            if first_piece and second_piece:
                fp_sq, fp = first_piece
                sp_sq, sp = second_piece

                if fp.color == enemy and sp.color == enemy:
                    if sp_sq == enemy_king:
                        pins.append({
                            "queen": chess.square_name(qsq),
                            "pinned": chess.square_name(fp_sq),
                            "pinned_to": "king",
                            "type": "absolute",
                        })
                    elif sp.piece_type in [chess.QUEEN, chess.ROOK]:
                        pins.append({
                            "queen": chess.square_name(qsq),
                            "pinned": chess.square_name(fp_sq),
                            "pinned_to": chess.square_name(sp_sq),
                            "type": "relative",
                        })

                if fp.color == enemy and sp.color == enemy:
                    if fp.piece_type in [chess.KING, chess.QUEEN, chess.ROOK]:
                        skewers.append({
                            "queen": chess.square_name(qsq),
                            "skewered": chess.square_name(fp_sq),
                            "behind": chess.square_name(sp_sq),
                        })

    return {
        "pins": pins,
        "pin_count": len(pins),
        "absolute_pins": sum(1 for p in pins if p["type"] == "absolute"),
        "relative_pins": sum(1 for p in pins if p["type"] == "relative"),
        "skewers": skewers,
        "skewer_count": len(skewers),
    }


def analyze_queen_overloading(board: chess.Board, color: bool) -> Dict:
    queens = list(board.pieces(chess.QUEEN, color))
    pieces = [sq for sq in chess.SQUARES
              if board.piece_at(sq) and board.piece_at(sq).color == color
              and board.piece_at(sq).piece_type != chess.KING]

    overload_data = []

    for qsq in queens:
        solely_defended = []
        total_defended = []

        for psq in pieces:
            if psq == qsq:
                continue
            defenders = list(board.attackers(color, psq))
            if qsq in defenders:
                total_defended.append(psq)
                if len(defenders) == 1:
                    solely_defended.append(psq)

        overload_data.append({
            "queen": chess.square_name(qsq),
            "total_defending": len(total_defended),
            "solely_defending": len(solely_defended),
            "solely_defending_squares": [chess.square_name(s) for s in solely_defended],
            "is_overloaded": len(solely_defended) >= 2,
        })

    return {
        "per_queen": overload_data,
        "overloaded_count": sum(1 for d in overload_data if d["is_overloaded"]),
    }

def full_queen_evaluation(board: chess.Board) -> Dict:
    result = {}
    
    for color, name in [(chess.WHITE, "white"), (chess.BLACK, "black")]:
        mobility = analyze_queen_mobility(board, color)
        development = analyze_queen_development(board, color)
        safety = analyze_queen_safety(board, color)
        activity = analyze_queen_activity(board, color)
        pins = analyze_queen_pins(board, color)
        overload = analyze_queen_overloading(board, color)

        result[name] = {
            "queen_count": mobility["count"],
            "total_mobility": mobility["total_mobility"],
            "per_queen_mobility": [
                {"sq": d["name"], "mobility": d["mobility"],
                 "diag": d["diagonal_mobility"], "orth": d["orthogonal_mobility"],
                 "center": d["attacks_center"]}
                for d in mobility["per_queen"]
            ],
            "minors_on_home": development["minors_on_home_rank"],
            "early_dev_penalty": development["early_development_penalty"],
            "total_exposure": safety["total_exposure"],
            "trapped_queens": safety["trapped_queens"],
            "per_queen_safety": [
                {"sq": d["square"], "attackers": d["attackers"],
                 "defenders": d["defenders"], "escape": d["escape_squares"],
                 "trapped": d["is_trapped"]}
                for d in safety["per_queen"]
            ],
            "total_activity": activity["total_activity"],
            "pin_count": pins["pin_count"],
            "absolute_pins": pins["absolute_pins"],
            "relative_pins": pins["relative_pins"],
            "skewer_count": pins["skewer_count"],
            "overloaded_count": overload["overloaded_count"],
            "overload_details": overload["per_queen"],
        }

    return result

