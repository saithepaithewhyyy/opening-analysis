#pragma once
#include "board.hpp"
#include "reader.hpp"
#include <vector>

vector<pair<Move, double>> generate_legal_scored_moves(const Board& b, const int& depth, const Reader::Book& book);
Board apply_move(const Board& b, const Move& m);
double move_scoring(const Move& m, const Board& before, const Board& after, const int& depth=1);