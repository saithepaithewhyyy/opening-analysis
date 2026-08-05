import chess
from typing import Dict, List, Tuple, Optional
from . import bishop, knight, queen, pawn, rook, helpers

def position_evaluate(fen: str) -> List:
    board = chess.Board(fen)
    evaluations = [
                bishop.full_bishop_evaluation(board),
                knight.full_knight_evaluation(board),
                queen.full_queen_evaluation(board),
                pawn.full_pawn_evaluation(board),
                rook.full_rook_evaluation(board),
            ]

    result = {
            "white": {},
            "black": {},
            }

    for evaluation in evaluations:
        for key, value in evaluation.items():
            if key == "white":
                result["white"].update(value)
            elif key == "black":
                result["black"].update(value)
            else:
                result[key] = value

    return result
