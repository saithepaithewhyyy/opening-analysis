#pragma once
#include "board.hpp"
#include <vector>

vector<pair<Move, double>> generate_legal_scored_moves(const Board& b, const int& topk=5);
Board apply_move(const Board& b, const Move& m);