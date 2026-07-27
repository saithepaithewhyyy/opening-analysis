import chess
from typing import Dict, List, Tuple, Optional
from piece_analysis import bishop, knight, queen, pawn, rook, helpers

def position_evaluate(board: chess.Board) -> List:
    return [
                helpers.compute_phase(board),
                bishop.full_bishop_evaluation(board),
                knight.full_knight_evaluation(board),
                queen.full_queen_evaluation(board),
                pawn.full_pawn_evaluation(board),
                rook.full_rook_evaluation(board),
            ]