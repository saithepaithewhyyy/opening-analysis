import chess
from typing import Dict, List, Tuple, Optional
import piece_analysis.helpers as hp

def analyze_rook_mobility(board: chess.Board, color: bool) -> Dict:
    
    rooks = list(board.pieces(chess.ROOK, color))
    data = []
    total_mobility = 0

    for sq in rooks:
        attacks = list(board.attacks(sq))
        reachable = [s for s in attacks
                     if not (board.piece_at(s) and board.piece_at(s).color == color)]
        enemy_attacked = [s for s in reachable
                          if board.piece_at(s) and board.piece_at(s).color != color]

        # Rank vs file breakdown
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)
        file_mob = sum(1 for s in reachable if chess.square_file(s) == file)
        rank_mob = sum(1 for s in reachable if chess.square_rank(s) == rank)

        mob = len(reachable)
        total_mobility += mob

        data.append({
            "square": sq,
            "name": chess.square_name(sq),
            "mobility": mob,
            "file_mobility": file_mob,
            "rank_mobility": rank_mob,
            "attacks_enemy": len(enemy_attacked),
        })

    return {
        "count": len(rooks),
        "per_rook": data,
        "total_mobility": total_mobility,
        "avg_mobility": total_mobility / len(rooks) if rooks else 0.0,
    }

def analyze_file_control(board: chess.Board, color: bool) -> Dict:

    rooks = list(board.pieces(chess.ROOK, color))
    enemy_rooks = list(board.pieces(chess.ROOK, not color))

    per_rook = []
    open_count = 0
    semi_open_count = 0
    contested = []

    for sq in rooks:
        file = chess.square_file(sq)
        open_f = hp.is_open_file(board, file)
        semi_f = hp.is_semi_open_file(board, file, color)
        enemy_on_file = any(chess.square_file(er) == file for er in enemy_rooks)

        if open_f:
            open_count += 1
            status = "open"
        elif semi_f:
            semi_open_count += 1
            status = "semi_open"
        else:
            status = "closed"

        if enemy_on_file:
            contested.append(file)

        per_rook.append({
            "square": chess.square_name(sq),
            "file": hp.FILE_NAMES[file],
            "file_status": status,
            "contested": enemy_on_file,
        })

    # Available open/semi-open files without a rook
    open_files_available  = [hp.FILE_NAMES[f] for f in range(8) if hp.is_open_file(board, f) and
                              not any(chess.square_file(r) == f for r in rooks)]
    semi_files_available  = [hp.FILE_NAMES[f] for f in range(8) if hp.is_semi_open_file(board, f, color) and
                              not any(chess.square_file(r) == f for r in rooks)]

    return {
        "per_rook": per_rook,
        "open_file_count": open_count,
        "semi_open_file_count": semi_open_count,
        "contested_files": [hp.FILE_NAMES[f] for f in set(contested)],
        "available_open_files": open_files_available,
        "available_semi_open_files": semi_files_available,
    }

def analyze_connected_rooks(board: chess.Board, color: bool) -> Dict:

    rooks = list(board.pieces(chess.ROOK, color))

    if len(rooks) < 2:
        return {"connected": False, "connection_rank": None}

    for i, r1 in enumerate(rooks):
        for r2 in rooks[i+1:]:
            if chess.square_rank(r1) == chess.square_rank(r2):
                f1, f2 = sorted([chess.square_file(r1), chess.square_file(r2)])
                rank = chess.square_rank(r1)
                clear = not any(
                    board.piece_at(chess.square(f, rank))
                    for f in range(f1 + 1, f2)
                )
                if clear:
                    return {
                        "connected": True,
                        "r1": chess.square_name(r1),
                        "r2": chess.square_name(r2),
                        "connection_rank": rank + 1,
                    }

    return {"connected": False, "connection_rank": None}

def analyze_rook_activity(board: chess.Board, color: bool) -> Dict:

    rooks = list(board.pieces(chess.ROOK, color))
    data = []
    total_activity = 0.0

    for sq in rooks:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)

        advancement = r if color == chess.WHITE else 7 - r
        centrality = 4 - abs(f - 3.5)
        mob = len([s for s in board.attacks(sq)
                   if not (board.piece_at(s) and board.piece_at(s).color == color)])

        activity = advancement * 1.5 + centrality * 2.0 + mob * 1.0
        total_activity += activity

        data.append({
            "square": chess.square_name(sq),
            "advancement": advancement,
            "centrality": centrality,
            "mobility": mob,
            "activity_index": activity,
        })

    return {
        "per_rook": data,
        "total_activity": round(total_activity, 2),
        "avg_activity": round(total_activity / len(rooks), 2) if rooks else 0.0,
    }

def full_rook_evaluation(board: chess.Board) -> Dict:
    result = {}

    for color, name in [(chess.WHITE, "white"), (chess.BLACK, "black")]:
        mobility = analyze_rook_mobility(board, color)
        files = analyze_file_control(board, color)
        connected = analyze_connected_rooks(board, color)
        activity = analyze_rook_activity(board, color)


        result[name] = {
            "rook_count": mobility["count"],
            "total_mobility": mobility["total_mobility"],
            "avg_mobility": round(mobility["avg_mobility"], 2),
            "per_rook_mobility": [
                {"sq": d["name"], "mobility": d["mobility"],
                 "file_mob": d["file_mobility"], "rank_mob": d["rank_mobility"]}
                for d in mobility["per_rook"]
            ],
            "open_file_count": files["open_file_count"],
            "semi_open_file_count": files["semi_open_file_count"],
            "contested_files": files["contested_files"],
            "available_open_files": files["available_open_files"],
            "per_rook_file": files["per_rook"],
            "connected": connected["connected"],
            "total_activity": activity["total_activity"],
            "per_rook_activity": activity["per_rook"],
        }

    return result