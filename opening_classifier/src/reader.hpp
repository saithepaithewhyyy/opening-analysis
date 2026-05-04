#ifndef READER_HPP
#define READER_HPP

#include <iostream>
#include "board.hpp"
#include <cstdio>
#include <fstream>
#include <string>
#include <cstdint>
#include <vector>
#include <map>
#include <random>
#include <algorithm>

using namespace std;

// Taken from https://github.com/deoshreyas/Chess-Polyglot-Reader
// Changed according to use case

namespace Reader {
    class underlying {
    public:
        uint16_t endian_swap_u16(uint16_t x) {
            return (x>>8) | (x<<8);
        }

        uint32_t endian_swap_u32(uint32_t x) {
            return (x>>24) | 
            ((x<<8) & 0x00FF0000) | 
            ((x>>8) & 0x0000FF00) | 
            (x<<24);
        }

        uint64_t endian_swap_u64(uint64_t x) {
            return (x>>56) | 
            ((x<<40) & 0x00FF000000000000) | 
            ((x<<24) & 0x0000FF0000000000) | 
            ((x<<8)  & 0x000000FF00000000) | 
            ((x>>8)  & 0x00000000FF000000) | 
            ((x>>24) & 0x0000000000FF0000) | 
            ((x>>40) & 0x000000000000FF00) | 
            (x<<56);
        }
    };
}

namespace Reader {
    struct EntryStruct {
        uint64_t key;
        uint16_t move;
        uint16_t weight;
        uint32_t learn;
    };
    
    static string Files[8] = {"a", "b", "c", "d", "e", "f", "g", "h"};
    static string Rows[8] = {"1", "2", "3", "4", "5", "6", "7", "8"};

    struct BookEntry {
        Move move;
        uint32_t weight;
        uint32_t learn;
    };

    typedef vector<BookEntry> BookEntries;
}

namespace Reader {
    class Book {
    private:
        long int num_entries = 0;
        EntryStruct* entries = nullptr;
    public:
        void Load(const vector<const char*>& paths) {
            underlying u;
            vector<EntryStruct> all_entries;

            for (auto& path : paths) {
                FILE *file = fopen(path, "rb");
                if (file == NULL) { 
                    cerr << "<Error> Please use valid book: " << path << endl; 
                    continue; 
                }

                fseek(file, 0, SEEK_END);
                long position = ftell(file);
                if (position < (long)sizeof(EntryStruct)) { 
                    cerr << "<Error> No entries found in: " << path << endl; fclose(file); 
                    continue; 
                }

                long n = position / sizeof(EntryStruct);
                vector<EntryStruct> tmp(n);
                rewind(file);
                fread(tmp.data(), sizeof(EntryStruct), n, file);
                fclose(file);

                for (auto& e : tmp) {
                    e.key = u.endian_swap_u64(e.key);
                    e.move = u.endian_swap_u16(e.move);
                    e.weight = u.endian_swap_u16(e.weight);
                    e.learn = u.endian_swap_u32(e.learn);
                }

                all_entries.insert(all_entries.end(), tmp.begin(), tmp.end());
            }

            sort(all_entries.begin(), all_entries.end(), [](const EntryStruct& a, const EntryStruct& b) {
                if (a.key != b.key) return a.key < b.key;
                return a.move < b.move;
            });

            vector<EntryStruct> merged;
            for (auto& e : all_entries) {
                if (!merged.empty() &&
                    merged.back().key  == e.key &&
                    merged.back().move == e.move) {
                    uint32_t combined = merged.back().weight + e.weight;
                    merged.back().weight = (uint16_t)min(combined, (uint32_t)UINT16_MAX);
                } else {
                    merged.push_back(e);
                }
            }

            num_entries = (long)merged.size();
            entries = (EntryStruct*)malloc(num_entries * sizeof(EntryStruct));
            memcpy(entries, merged.data(), num_entries * sizeof(EntryStruct));
        }

        static Move decode_move(uint16_t m) {
            Move mv;
            mv.from = ((m >> 9) & 7) * 8 + ((m >> 6) & 7);
            mv.to = ((m >> 3) & 7) * 8 + ((m >> 0) & 7);
            mv.promotion = ((m >> 12) & 7);
            mv.is_castle = false;
            mv.is_ep = false;
            if (mv.from == 4 && mv.to == 7) { 
                mv.is_castle = true; 
                mv.to = 6;  
            }  // e1→g1

            if (mv.from == 4 && mv.to == 0) { 
                mv.is_castle = true; 
                mv.to = 2;
            }  // e1→c1

            if (mv.from == 60 && mv.to == 63) { 
                mv.is_castle = true; 
                mv.to = 62; 
            }  // e8→g8
            if (mv.from == 60 && mv.to == 56) { 
                mv.is_castle = true; 
                mv.to = 58; 
            }  // e8→c8
            
            return mv;
        }

        const EntryStruct* lower_bound_key(uint64_t key) const {
            long lo = 0, hi = num_entries;
            while (lo < hi) {
                long mid = lo + (hi - lo) / 2;
                if (entries[mid].key < key) lo = mid + 1;
                else                        hi = mid;
            }
            return entries + lo;
        }

        // @brief Get moves from book for a given Zobrist key.
        // Now const (safe for concurrent OMP reads) and O(log N + k).
        map<Move, uint32_t> GetBookMoves(uint64_t key, uint16_t minimum_weight=0) const {
            map<Move, uint32_t> bookMoves;
            for (const EntryStruct* e = lower_bound_key(key);
                 e < entries + num_entries && e->key == key; ++e) {
                if (e->weight >= minimum_weight)
                    bookMoves[decode_move(e->move)] = e->weight;
            }
            return bookMoves;
        }

        BookEntries SearchBook(uint64_t key, uint16_t minimum_weight=0) const {
            BookEntries bookEntries;
            for (const EntryStruct* e = lower_bound_key(key);
                 e < entries + num_entries && e->key == key; ++e) {
                if (e->weight >= minimum_weight) {
                    BookEntry be;
                    be.move   = decode_move(e->move);
                    be.weight = e->weight;
                    be.learn  = e->learn;
                    bookEntries.push_back(be);
                }
            }
            return bookEntries;
        }
        
        void Clear() {
            free(entries);
            num_entries = 0;
            entries = nullptr;
        } 
    };

    typedef vector<Book> Books;
}

#endif