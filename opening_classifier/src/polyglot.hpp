#pragma once
#include "board.hpp"
#include <cstdint>
#include <vector>
#include <string>

using namespace std;

struct PolyGlotEntry{
    uint64_t key;
    uint16_t move;
    uint16_t weight;
    uint32_t learn;
};

vector<PolyGlotEntry> load_polyglot(const string& path);
Move decode_PolyGlotMove(uint16_t poly_move);
