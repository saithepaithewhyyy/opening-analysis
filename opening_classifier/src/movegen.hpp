#pragma once
#include "board.hpp"
#include <vector>

vector<Move> generate_legal_moves(const Board& b);
Board apply_move(const Board& b, const Move& m);