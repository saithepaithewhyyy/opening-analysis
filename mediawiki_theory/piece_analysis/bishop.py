import chess
from typing import Dict, List, Tuple, Optional
import piece_analysis.helpers as hp


def _piece_cache(board: chess.Board) -> Dict[int, Optional[chess.Piece]]:
    """Build a square→piece lookup"""
    return {sq: board.piece_at(sq) for sq in chess.SQUARES}



def analyze_bishop_mobility(board: chess.Board, color: bool,
                             piece_cache: Optional[Dict] = None) -> Dict:
    if piece_cache is None:
        piece_cache = _piece_cache(board)

    bishops = list(board.pieces(chess.BISHOP, color))
    data = []
    total_mobility = 0

    for sq in bishops:
        # Compute attacks once; reuse the set for all downstream checks.
        attacks = list(board.attacks(sq))

        reachable = []
        enemy_attacked_count = 0
        center_attacks = 0

        for s in attacks:
            occupant = piece_cache.get(s)
            if occupant and occupant.color == color:
                continue
            reachable.append(s)
            if occupant:
                enemy_attacked_count += 1
            if s in hp.CENTER_SQUARES:
                center_attacks += 1

        max_ray = 0
        for ray in hp.get_diagonal_squares(sq):
            ray_len = 0
            for rs in ray:
                piece = piece_cache.get(rs)
                if piece:
                    if piece.color != color:
                        ray_len += 1
                    break
                ray_len += 1
            if ray_len > max_ray:
                max_ray = ray_len

        mob = len(reachable)
        total_mobility += mob

        data.append({
            "square": sq,
            "name": chess.square_name(sq),
            "color": hp.bishop_color(sq),
            "mobility": mob,
            "max_diagonal_reach": max_ray,
            "attacks_enemy_count": enemy_attacked_count,
            "attacks_center": center_attacks,
        })

    return {
        "count": len(bishops),
        "per_bishop": data,
        "total_mobility": total_mobility,
    }

def analyze_open_diagonals(board: chess.Board, color: bool,
                            piece_cache: Optional[Dict] = None) -> Dict:
    if piece_cache is None:
        piece_cache = _piece_cache(board)

    bishops = list(board.pieces(chess.BISHOP, color))
    results = []
    open_count = semi_open_count = 0

    for sq in bishops:
        best_ray_status = "blocked"
        for ray in hp.get_diagonal_squares(sq):
            own_pawns_in_ray = 0
            any_piece_in_ray = False

            for rs in ray:
                piece = piece_cache.get(rs)
                if piece:
                    any_piece_in_ray = True
                    if piece.piece_type == chess.PAWN and piece.color == color:
                        own_pawns_in_ray += 1

            if not any_piece_in_ray:
                best_ray_status = "open"
                break
            elif own_pawns_in_ray == 0:
                best_ray_status = "semi_open"
                # Don't break — a fully open ray is better, keep looking.

        if best_ray_status == "open":
            open_count += 1
        elif best_ray_status == "semi_open":
            semi_open_count += 1

        results.append({
            "square": chess.square_name(sq),
            "diagonal_status": best_ray_status,
        })

    return {
        "open_diagonals": open_count,
        "semi_open_diagonals": semi_open_count,
        "blocked_diagonals": len(bishops) - open_count - semi_open_count,
    }


def analyze_fianchetto(board: chess.Board, color: bool,
                        piece_cache: Optional[Dict] = None) -> Dict:
    if piece_cache is None:
        piece_cache = _piece_cache(board)

    fianchetto_sqs = hp.FIANCHETTO_SQUARES[color]
    king_sq = board.king(color)
    fianchettos = []

    for fsq in fianchetto_sqs:
        piece = piece_cache.get(fsq)
        if not (piece and piece.piece_type == chess.BISHOP and piece.color == color):
            continue

        expected_king  = hp.FIANCHETTO_KING_SQUARES[color].get(fsq)
        king_castled   = (king_sq == expected_king) if expected_king else False
        on_long_diag   = fsq in hp.LONG_DIAGONAL_A1H8 or fsq in hp.LONG_DIAGONAL_A8H1

        max_reach = 0
        for ray in hp.get_diagonal_squares(fsq):
            length = 0
            for rs in ray:
                if piece_cache.get(rs):
                    break
                length += 1
            if length > max_reach:
                max_reach = length

        strength = ("strong"   if king_castled and max_reach >= 4 else
                    "moderate" if max_reach >= 3 else "weak")

        fianchettos.append({
            "square": chess.square_name(fsq),
            "king_castled_to_side": king_castled,
            "on_long_diagonal": on_long_diag,
            "max_diagonal_reach": max_reach,
            "strength": strength,
        })

    return {
        "fianchetto_count": len(fianchettos),
        "fianchettos": fianchettos,
    }


def analyze_opposite_color_bishops(board: chess.Board,
                                    phase: Optional[float] = None) -> Dict:
    white_bishops = list(board.pieces(chess.BISHOP, chess.WHITE))
    black_bishops = list(board.pieces(chess.BISHOP, chess.BLACK))

    if len(white_bishops) != 1 or len(black_bishops) != 1:
        return {"opposite_color": False, "implication": "N/A"}

    wb_dark = hp.is_dark(white_bishops[0])
    bb_dark = hp.is_dark(black_bishops[0])
    opposite = wb_dark != bb_dark

    return {
        "opposite_color": opposite,
    }


def analyze_bishop_battery(board: chess.Board, color: bool,
                            piece_cache: Optional[Dict] = None) -> Dict:
    """Battery detection with O(1) queen-in-ray lookup via a set."""
    bishops = list(board.pieces(chess.BISHOP, color))
    queens  = list(board.pieces(chess.QUEEN,  color))
    queen_set   = set(queens)
    enemy_king  = board.king(not color)
    batteries   = []

    if piece_cache is None:
        piece_cache = _piece_cache(board)

    for bsq in bishops:
        for ray in hp.get_diagonal_squares(bsq):
            for i, rs in enumerate(ray):
                # Stop scanning this ray on the first blocker.
                occupant = piece_cache.get(rs)
                if occupant:
                    if rs in queen_set:
                        # Path from bishop to queen is clear (no piece at indices < i).
                        clear = not any(piece_cache.get(ray[j]) for j in range(i))
                        points_to_king = enemy_king is not None and enemy_king in ray
                        batteries.append({
                            "bishop": chess.square_name(bsq),
                            "queen":  chess.square_name(rs),
                            "clear_path": clear,
                            "points_to_king": points_to_king,
                        })
                    break   # ray blocked regardless

    return {
        "battery_count": len(batteries),
        "batteries": batteries,
        "king_targeting_battery": any(b["points_to_king"] for b in batteries),
    }



def full_bishop_evaluation(board: chess.Board) -> Dict:
    phase        = hp.compute_phase(board)
    piece_cache  = _piece_cache(board)

    result = {
        "opposite_color_bishops": analyze_opposite_color_bishops(board, phase=phase),
        #  NEED TO LOOK INTO
    }

    for color, name in [(chess.WHITE, "white"), (chess.BLACK, "black")]:
        bishops = board.pieces(chess.BISHOP, color)

        bishop_squares = [chess.square_name(sq) for sq in bishops]
        mobility = analyze_bishop_mobility(board, color, piece_cache)
        # need to look into mobility 
        pair = len(list(bishops)) > 2
        open_diag = analyze_open_diagonals(board, color, piece_cache)
        fianchetto = analyze_fianchetto(board, color, piece_cache)
        battery = analyze_bishop_battery(board, color, piece_cache)

        result[name] = {
            "bishop_count":mobility["count"],
            "bishop_locations": bishop_squares, 
            "bishop_pair": pair,
            "bishop_mobility": mobility["total_mobility"],
            "per_bishop_mobility": [
                {"sq": d["name"], "color": d["color"],
                 "mobility": d["mobility"], "center_attacks": d["attacks_center"]}
                for d in mobility["per_bishop"]
            ],
            "open_diagonals":open_diag["open_diagonals"],
            "semi_open_diagonals": open_diag["semi_open_diagonals"],
            "blocked_diagonals": open_diag["blocked_diagonals"],
            "fianchetto_count": fianchetto["fianchetto_count"],
            "fianchettos":fianchetto["fianchettos"],
            "battery_count": battery["battery_count"],
            "king_targeting_battery": battery["king_targeting_battery"],
        }

    return result