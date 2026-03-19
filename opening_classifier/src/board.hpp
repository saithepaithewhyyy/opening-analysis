#pragma once
#include <cstdint>
#include <string>
#include <array>
#include <vector>
#include <random>

using namespace std;

enum Piece  { PAWN=0, KNIGHT, BISHOP, ROOK, QUEEN, KING, NO_PIECE=6 };
enum Color  { WHITE=0, BLACK=1 };

struct Move {
    uint8_t from;
    uint8_t to;
    uint8_t promotion;   
    bool is_castle;
    bool is_ep;
};

struct Board {
    uint64_t bb[2][6] = {};   // bb[color][piece]
    uint64_t occupied = 0;
    Color turn = WHITE;
    uint8_t castling = 0xF;  // WK WQ BK BQ
    uint8_t ep_sq = 255;  // 255 = none // used throughout
    uint64_t zobrist = 0;

    uint64_t us() const 
    { 
        return bb[turn][0]|bb[turn][1]|bb[turn][2]|bb[turn][3]|bb[turn][4]|bb[turn][5]; 
    }
    
    uint64_t them() const 
    { 
        Color t=(Color)(1-turn);
        return bb[t][0]|bb[t][1]|bb[t][2]|bb[t][3]|bb[t][4]|bb[t][5]; 
    }
};

struct ZobristTable {
    uint64_t piece[2][6][64];
    uint64_t black_to_move;
    uint64_t castling[16];
    uint64_t ep_file[8];

    ZobristTable() {
        mt19937_64 rng(0xDEADBEEFCAFE1234ULL);
        for (auto& c : piece)
            for (auto& p : c)
                for (auto& s : p) s = rng();
        black_to_move = rng();
        for (auto& c : castling) c = rng();
        for (auto& f : ep_file)  f = rng();
    }
};

inline const ZobristTable ZT;  
Board board_from_fen(const string& fen);
uint64_t compute_zobrist(const Board& b);