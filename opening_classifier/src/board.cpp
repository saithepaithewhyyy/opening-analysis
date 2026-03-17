#include "board.hpp"
#include <sstream>
#include <stdexcept>

using namespace std;

static const char PIECE_CHARS[] = "pnbrqk";

Board board_from_fen(const string& fen) {
    Board b = {};
    istringstream ss(fen);
    string placement, turn_str, castle_str, ep_str;
    ss >> placement >> turn_str >> castle_str >> ep_str;

    int sq = 56;
    for (char c : placement) {
        if (c == '/') { sq -= 16; continue; }
        if (c >= '1' && c <= '8') { 
            sq += (c - '0'); 
            continue; 
        }

        Color col = (islower(c)) ? BLACK : WHITE;
        char  lc  = (islower(c)) ? c : tolower(c);
        int   pc  = 0;
        while (PIECE_CHARS[pc] != lc) pc++;
        b.bb[col][pc] |= (1ULL << sq);
        sq++;
    }

    for (int c = 0; c < 2; c++)
        for (int p = 0; p < 6; p++)
            b.occupied |= b.bb[c][p];

    b.turn = (turn_str == "b") ? BLACK : WHITE;

    b.castling = 0;
    for (char c : castle_str) {
        if (c == 'K') b.castling |= 0x8;
        if (c == 'Q') b.castling |= 0x4;
        if (c == 'k') b.castling |= 0x2;
        if (c == 'q') b.castling |= 0x1;
    }

    if (ep_str != "-") {
        int file = ep_str[0] - 'a';
        int rank = ep_str[1] - '1';
        b.ep_sq  = rank * 8 + file;
    } else {
        b.ep_sq = 255;
    }

    b.zobrist = compute_zobrist(b);
    return b;
}

uint64_t compute_zobrist(const Board& b) {
    uint64_t z = 0;
    for (int c = 0; c < 2; c++)
        for (int p = 0; p < 6; p++) {
            uint64_t bb = b.bb[c][p];
            while (bb) {
                int sq = __builtin_ctzll(bb);
                z ^= ZT.piece[c][p][sq];
                bb &= bb - 1;
            }
        }
    if (b.turn == BLACK) z ^= ZT.black_to_move;
    z ^= ZT.castling[b.castling & 0xF];
    if (b.ep_sq != 255) z ^= ZT.ep_file[b.ep_sq & 7];
    return z;
}