#include "movegen.hpp"
#include "definitions.hpp"
#include "reader.hpp"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <set>
#include <numeric>
#include <mutex>

using namespace std;
using namespace Reader;

const int promos[] = {QUEEN, ROOK, BISHOP, KNIGHT};
static const double PROB_THRESHOLD = 0.02;

static inline int lsb(uint64_t b) { return ctzll(b); }
static inline int msb(uint64_t b) { return clzll(b); }

static inline uint64_t pop_lsb(uint64_t& b) {
    uint64_t t = b & -b; 
    b &= b-1; 
    return t;
}

static uint64_t KNIGHT_ATK[64];
static uint64_t KING_ATK[64];
static uint64_t PAWN_ATK[2][64];
static uint64_t RAY[64][8];       

static const int DIR[8] = {8, -8, 1, -1, 9, 7, -9, -7};
static const double PIECE_VAL[7] = {100, 300, 325, 500, 900, 0, 0};

static const double KNIGHT_PST[64] = {
    10, 15, 10, 10, 10, 10, 15, 10,
    15, 20, 30, 35, 35, 30, 20, 15,
    25, 40, 75, 65, 65, 75, 40, 25,
    30, 50, 75, 90, 90, 75, 50, 30,
    25, 50, 75, 95, 95, 75, 50, 25,
    20, 40, 60, 65, 65, 60, 40, 20,
    10, 20, 30, 35, 35, 30, 20, 10,
     5, 10, 10, 10, 10, 10, 10,  5,
};

static const double BISHOP_PST[64] = {
    30, 20, 20, 20, 20, 20, 20, 30,
    20, 70, 45, 50, 50, 45, 70, 20,
    25, 45, 55, 55, 55, 55, 45, 25,
    25, 45, 55, 65, 65, 55, 45, 25,
    25, 45, 55, 65, 65, 55, 45, 25,
    25, 40, 50, 55, 55, 50, 40, 25,
    20, 65, 40, 45, 45, 40, 65, 20,
    30, 20, 20, 20, 20, 20, 20, 30,
};

static const double PAWN_PST[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,
    40, 40, 42, 45, 45, 42, 40, 40,
    42, 40, 44, 50, 50, 44, 40, 42,
    44, 44, 47, 70, 70, 47, 44, 44,
    48, 48, 50, 60, 60, 50, 48, 48,
    52, 52, 55, 60, 60, 55, 52, 52,
    70, 70, 70, 70, 70, 70, 70, 70,
     0,  0,  0,  0,  0,  0,  0,  0,
};

static const double ROOK_PST[64] = {
    40, 40, 42, 45, 45, 42, 40, 40,
    40, 40, 42, 45, 45, 42, 40, 40,
    40, 40, 42, 45, 45, 42, 40, 40,
    40, 40, 42, 45, 45, 42, 40, 40,
    40, 40, 42, 45, 45, 42, 40, 40,
    40, 40, 42, 45, 45, 42, 40, 40,
    60, 60, 62, 65, 65, 62, 60, 60,
    45, 45, 47, 50, 50, 47, 45, 45,
};

static const double QUEEN_PST[64] = {
    20, 25, 30, 30, 30, 30, 25, 20,
    25, 30, 35, 40, 40, 35, 30, 25,
    30, 35, 40, 45, 45, 40, 35, 30,
    35, 40, 45, 55, 55, 45, 40, 35,
    35, 40, 45, 55, 55, 45, 40, 35,
    30, 35, 40, 45, 45, 40, 35, 30,
    25, 30, 35, 40, 40, 35, 30, 25,
    20, 25, 30, 30, 30, 30, 25, 20,
};

static const double KING_PST[64] = {
    35, 45, 65, 20, 35, 20, 70, 40,
    20, 25, 25, 10, 10, 25, 25, 20,
    10, 15, 15,  5,  5, 15, 15, 10,
     5, 10, 10,  0,  0, 10, 10,  5,
     5, 10, 10,  0,  0, 10, 10,  5,
    10, 15, 15,  5,  5, 15, 15, 10,
    20, 25, 25, 10, 10, 25, 25, 20,
    35, 45, 65, 20, 35, 20, 70, 40,
};

static const double* PST[6] = {
    PAWN_PST, KNIGHT_PST, BISHOP_PST, ROOK_PST, QUEEN_PST, KING_PST
};


static bool tables_ready = false;
static void init_tables() {
    static once_flag flag;
    // if (tables_ready) return;
    call_once(flag, []() {
        for (int sq = 0; sq < 64; sq++) {
            
            uint64_t b = 1ULL << sq;

            // knight
            KNIGHT_ATK[sq] =
                ((b<<17)&~0x0101010101010101ULL) | ((b<<15)&~0x8080808080808080ULL) |
                ((b<<10)&~0x0303030303030303ULL) | ((b<< 6)&~0xC0C0C0C0C0C0C0C0ULL) |
                ((b>>17)&~0x8080808080808080ULL) | ((b>>15)&~0x0101010101010101ULL) |
                ((b>>10)&~0xC0C0C0C0C0C0C0C0ULL) | ((b>> 6)&~0x0303030303030303ULL);

            // king
            uint64_t kb = b;
            kb |= (b<<8)|(b>>8);
            kb |= ((kb & ~0x0101010101010101ULL) >> 1) |
                ((kb & ~0x8080808080808080ULL) << 1);
            KING_ATK[sq] = kb & ~b;

            // pawns
            PAWN_ATK[WHITE][sq] =
                ((b & ~0x0101010101010101ULL) << 7) |
                ((b & ~0x8080808080808080ULL) << 9);
            PAWN_ATK[BLACK][sq] =
                ((b & ~0x8080808080808080ULL) >> 7) |
                ((b & ~0x0101010101010101ULL) >> 9);

            // rays
            for (int d = 0; d < 8; d++) {
                RAY[sq][d] = 0;
                int cur = sq;
                while (true) {
                    int file = cur & 7, rank = cur >> 3;
                    int nf = file + (d==2||d==4||d==5 ? 1 : d==3||d==6||d==7 ? -1 : 0);
                    int nr = rank + (d==0||d==4||d==6 ? 1 : d==1||d==5||d==7 ? -1 : 0);
                    if (nf < 0||nf > 7||nr < 0||nr > 7) break;
                    cur = nr*8 + nf;
                    RAY[sq][d] |= (1ULL << cur);
                }
            }
        }
    });
    // tables_ready = true;
}

static uint64_t attack_lines(int sq, uint64_t occ, bool diagonal) {
    uint64_t atk = 0;
    int dirs[4];
    
    if (diagonal) { 
        dirs[0]=4; dirs[1]=5; dirs[2]=6; dirs[3]=7; 
    }
    else {
        dirs[0]=0; dirs[1]=1; dirs[2]=2; dirs[3]=3;
    }


    for (int i = 0; i < 4; i++) {
        uint64_t ray = RAY[sq][dirs[i]];
        uint64_t blk = ray & occ;


        if (blk) {
            int blocker;
            if (dirs[i] == 0 || dirs[i] == 2 || dirs[i] == 4 || dirs[i] == 5)
                blocker = lsb(blk);
            else
                blocker = msb(blk);
            ray ^= RAY[blocker][dirs[i]];
        }
        
        atk |= ray;
    
    }
    return atk;
}


static bool is_attacked(const Board& b, int sq, Color enemy) {
    if (KNIGHT_ATK[sq] & b.bb[enemy][KNIGHT]) return true;
    if (KING_ATK[sq] & b.bb[enemy][KING])   return true;
    if (PAWN_ATK[1-enemy][sq] & b.bb[enemy][PAWN])   return true;
    
    uint64_t diag  = attack_lines(sq, b.occupied, true);
    uint64_t cross = attack_lines(sq, b.occupied, false);
    
    if (diag  & (b.bb[enemy][BISHOP] | b.bb[enemy][QUEEN])) return true;
    if (cross & (b.bb[enemy][ROOK]   | b.bb[enemy][QUEEN])) return true;
    return false;
}

Board apply_move(const Board& b, const Move& m) {
    Board n = b;
    Color enemy = (Color)(1 - b.turn);

    Piece pc = NO_PIECE;
    for (int p = 0; p < 6; p++)
        if (n.bb[b.turn][p] & (1ULL << m.from)) { 
            pc = (Piece)p; 
            break;
        }

    n.bb[b.turn][pc] &= ~(1ULL << m.from);
    n.occupied &= ~(1ULL << m.from);
    
    if (m.is_ep) {
        int cap = m.to + (b.turn == WHITE ? -8 : 8);
        n.bb[enemy][PAWN] &= ~(1ULL << cap);
        n.occupied &= ~(1ULL << cap);
    }

    for (int p = 0; p < 6; p++)
        n.bb[enemy][p] &= ~(1ULL << m.to);

    Piece placed = (m.promotion != NO_PIECE) ? (Piece)m.promotion : pc;
    n.bb[b.turn][placed] |= (1ULL << m.to);
    n.occupied |= (1ULL << m.to);
    

    if (m.is_castle) {
        bool ks = (m.to > m.from);
        int rfrom = ks ? m.from + 3 : m.from - 4;
        int rto = ks ? m.from + 1 : m.from - 1;
        
        n.bb[b.turn][ROOK] &= ~(1ULL << rfrom);
        n.occupied &= ~(1ULL << rfrom);

        n.bb[b.turn][ROOK] |=  (1ULL << rto);
        n.occupied |=  (1ULL << rto);
    }

    // Update castling rights
    auto clear = [&](int s) {
        if (s == 0)  n.castling &= ~0x4;
        if (s == 7)  n.castling &= ~0x8;
        if (s == 56) n.castling &= ~0x1;
        if (s == 63) n.castling &= ~0x2;
        if (s == 4)  n.castling &= ~0xC;  // white king
        if (s == 60) n.castling &= ~0x3;  // black king
    };
    
    clear(m.from); 
    clear(m.to);

    n.ep_sq = 255;
    if (pc == PAWN && abs((int)m.to - (int)m.from) == 16)
        n.ep_sq = (m.from + m.to) / 2;
    else
        n.ep_sq = 255;

    n.turn    = enemy;
    n.zobrist = compute_zobrist(n);
    return n;
}

double move_scoring(const Move& m, const Board& before, const Board& after, const int& depth) {
    Color us = before.turn;
    Color enemy = (Color)(1 - us);
    uint8_t from = m.from, to = m.to;

    Piece p = NO_PIECE;
    for (int i = 0; i < 6; i++)
        if (before.bb[us][i] & (1ULL << from)) { p = (Piece)i; break; }

    int to_sq = (us == WHITE) ? to : (to ^ 56);
    int from_sq = (us == WHITE) ? from : (from ^ 56);

    double score = 0.0;

    Piece victim = NO_PIECE;
    if (m.is_ep) {
        victim = PAWN;
    } else {
        for (int i = 0; i < 6; i++)
            if (before.bb[enemy][i] & (1ULL << to)) { victim = (Piece)i; break; }
    }
    if (victim != NO_PIECE) {
        double loss = max(0.0, PIECE_VAL[p] - PIECE_VAL[victim]);
        score += (PIECE_VAL[victim] - 0.1 * loss) / 10.0;
    }

    if (victim == NO_PIECE && is_attacked(before, (int)from, enemy)
                           && !is_attacked(after, (int)to, enemy))
        score += PIECE_VAL[p] / 10.0;

    if (is_attacked(after, lsb(after.bb[enemy][KING]), us))
        score += 25.0;

    if (m.is_castle)
        return score + 70.0;

    if (p != NO_PIECE)
        score += PST[p][to_sq] - PST[p][from_sq];

    if ((p == KNIGHT || p == BISHOP) && from_sq < 8)
        score += 35.0;

    if (p == PAWN && (to_sq == 27 || to_sq == 28 || to_sq == 35 || to_sq == 36))
        score += 15.0;

    if (p == QUEEN && depth < 3)
        score -= 50.0;
    if (p == KING)
        score -= 35.0;  

    return score;
}

vector<pair<Move, double>> generate_legal_scored_moves(const Board& b, const int& depth, const Book& book) {
    init_tables();
    vector <pair<Move, double>> legal;
    legal.reserve(64);

    Color us  = b.turn;
    Color enemy = (Color)(1 - us);

    uint64_t our_pieces = b.us();
    uint64_t enemy_pieces = b.them();

    map<Move, uint32_t> book_moves = book.GetBookMoves(b.zobrist);

    auto try_add = [&](Move m) {
        Board after = apply_move(b, m);
        int ksq = lsb(after.bb[us][KING]);
        if (!is_attacked(after, ksq, enemy)){
            double score = move_scoring(m, b, after, depth);
            
            Move dup = m;
            dup.is_ep = false;   
            dup.is_castle = false;
            dup.promotion = (m.promotion == NO_PIECE) ? 0 : m.promotion;

            auto it = book_moves.find(dup);
            if (it != book_moves.end()) {
                cout << "found!" << endl;
                score += log1p((double)it->second) * 50.0;
            }

            legal.push_back({m, score});
        }
    };

    // pawnies
    {
        uint64_t pawns = b.bb[us][PAWN];
        int push_dir = (us == WHITE) ? 8 : -8;
        int start_rank = (us == WHITE) ? 1 : 6;
        int promo_rank = (us == WHITE) ? 7 : 0;

        while (pawns) {
            int sq = lsb(pawns); 
            pawns &= pawns-1;
            int rank = sq >> 3;

            // Single push
            int to = sq + push_dir;
            if (to >= 0 && to < 64 && !(b.occupied & (1ULL << to))) {
                if ((to >> 3) == promo_rank) {
                    for (int pr : promos)
                        try_add({(uint8_t)sq,(uint8_t)to,(uint8_t)pr,false,false});
                } else {
                    try_add({(uint8_t)sq,(uint8_t)to,(uint8_t)NO_PIECE,false,false});
                    // Double push
                    if (rank == start_rank) {
                        int to2 = to + push_dir;
                        if (!(b.occupied & (1ULL << to2)))
                            try_add({(uint8_t)sq,(uint8_t)to2,(uint8_t)NO_PIECE,false,false});
                    }
                }
            }

            // Captures
            uint64_t atk = PAWN_ATK[us][sq] & enemy_pieces;
            while (atk) {
                int t = lsb(atk); atk &= atk-1;
                if ((t >> 3) == promo_rank) {
                    for (int pr : promos)
                        try_add({(uint8_t)sq,(uint8_t)t,(uint8_t)pr,false,false});
                } else {
                    try_add({(uint8_t)sq,(uint8_t)t,(uint8_t)NO_PIECE,false,false});
                }
            }

            // En passant
            if (b.ep_sq != 255 && (PAWN_ATK[us][sq] & (1ULL << b.ep_sq)))
                try_add({(uint8_t)sq,(uint8_t)b.ep_sq,(uint8_t)NO_PIECE,false,true});
        }
    }

    // nights
    {
        uint64_t knights = b.bb[us][KNIGHT];
        while (knights) {
            int sq = lsb(knights); 
            knights &= knights-1;
            uint64_t atk = KNIGHT_ATK[sq] & ~our_pieces;

            while (atk) {
                int t = lsb(atk); 
                atk &= atk-1;
                try_add({(uint8_t)sq,(uint8_t)t,(uint8_t)NO_PIECE,false,false});
            }
        }
    }

    // bishops
    {
        uint64_t bishops = b.bb[us][BISHOP];
        while (bishops) {
            int sq = lsb(bishops); 
            bishops &= bishops-1;
            uint64_t atk = attack_lines(sq, b.occupied, true) & ~our_pieces;
            while (atk) {
                int t = lsb(atk); 
                atk &= atk-1;
                try_add({(uint8_t)sq,(uint8_t)t,(uint8_t)NO_PIECE,false,false});
            }
        }
    }

    // rooks
    {
        uint64_t rooks = b.bb[us][ROOK];
        while (rooks) {
            int sq = lsb(rooks); 
            rooks &= rooks-1;
            uint64_t atk = attack_lines(sq, b.occupied, false) & ~our_pieces;
            while (atk) {
                int t = lsb(atk); 
                atk &= atk-1;
                try_add({(uint8_t)sq,(uint8_t)t,(uint8_t)NO_PIECE,false,false});
            }
        }
    }

    // queens
    {
        uint64_t queens = b.bb[us][QUEEN];
        while (queens) {
            int sq = lsb(queens); 
            queens &= queens-1;
            uint64_t atk = (attack_lines(sq, b.occupied, true) |
                            attack_lines(sq, b.occupied, false)) & ~our_pieces;
            while (atk) {
                int t = lsb(atk); atk &= atk-1;
                try_add({(uint8_t)sq,(uint8_t)t,(uint8_t)NO_PIECE,false,false});
            }
        }
    }

    // king
    {
        int sq = lsb(b.bb[us][KING]);
        uint64_t atk = KING_ATK[sq] & ~our_pieces;
        while (atk) {
            int t = lsb(atk); atk &= atk-1;
            try_add({(uint8_t)sq,(uint8_t)t,(uint8_t)NO_PIECE,false,false});
        }

        // castling
        if (us == WHITE) {
            if ((b.castling & 0x8) &&
                !(b.occupied & 0x60ULL) &&
                !is_attacked(b,4,enemy) && !is_attacked(b,5,enemy) && !is_attacked(b,6,enemy))
                try_add({4,6,(uint8_t)NO_PIECE,true,false});
            if ((b.castling & 0x4) &&
                !(b.occupied & 0xEULL) &&
                !is_attacked(b,4,enemy) && !is_attacked(b,3,enemy) && !is_attacked(b,2,enemy))
                try_add({4,2,(uint8_t)NO_PIECE,true,false});
        } else {
            if ((b.castling & 0x2) &&
                !(b.occupied & 0x6000000000000000ULL) &&
                !is_attacked(b,60,enemy) && !is_attacked(b,61,enemy) && !is_attacked(b,62,enemy))
                try_add({60,62,(uint8_t)NO_PIECE,true,false});
            if ((b.castling & 0x1) &&
                !(b.occupied & 0xE000000000000000ULL) &&
                !is_attacked(b,60,enemy) && !is_attacked(b,59,enemy) && !is_attacked(b,58,enemy))
                try_add({60,58,(uint8_t)NO_PIECE,true,false});
        }
    }

    sort(legal.begin(), legal.end(), [](auto &a, auto &b){ return a.second > b.second; });

    static const double TEMP = 45.0;
    double max_score = legal.empty() ? 0.0 : legal.front().second;
    double sum = 0.0;
    for (auto& p : legal) { p.second = exp((p.second - max_score) / TEMP); sum += p.second; }
    for (auto& p : legal) p.second /= sum;

    legal.erase(
        remove_if(legal.begin(), legal.end(),
            [](const pair<Move,double>& p) { return p.second < PROB_THRESHOLD; }),
        legal.end()
    );

    return legal;
}