#pragma once
#include "board.hpp"
#include <vector>

vector<pair<Move, double>> generate_legal_scored_moves(const Board& b, const int& depth);
Board apply_move(const Board& b, const Move& m);
double move_score_heuristics(const Move& m, const int& depth=5);