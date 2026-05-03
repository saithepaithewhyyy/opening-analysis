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
        uint32_t weight;
        uint32_t learn;
    };
    
    typedef vector<BookEntry> BookEntries;
    typedef vector<Book> Books;

    long int num_entries = 0;
    EntryStruct* entries;

    static string Files[8] = {"a", "b", "c", "d", "e", "f", "g", "h"};
    static string Rows[8] = {"1", "2", "3", "4", "5", "6", "7", "8"};

    struct BookEntry {
        Move move;
        uint32_t weight;
        uint32_t learn;
    };

}

namespace Reader {
    class Book {
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

                all_entries.insert(all_entries.end(), tmp.begin(), tmp.end());
            }

            sort(all_entries.begin(), all_entries.end(), [&](const EntryStruct& a, const EntryStruct& b) {
                if (u.endian_swap_u64(a.key) != u.endian_swap_u64(b.key))
                    return u.endian_swap_u64(a.key) < u.endian_swap_u64(b.key);
                return u.endian_swap_u16(a.move) < u.endian_swap_u16(b.move);
            });


            vector<EntryStruct> merged;
            for (auto& e : all_entries) {
                if (!merged.empty() &&
                    u.endian_swap_u64(merged.back().key)  == u.endian_swap_u64(e.key) &&
                    u.endian_swap_u16(merged.back().move) == u.endian_swap_u16(e.move)) {
                    uint32_t combined = (uint32_t)u.endian_swap_u16(merged.back().weight)
                                    + (uint32_t)u.endian_swap_u16(e.weight);
                    merged.back().weight = u.endian_swap_u16((uint16_t)min(combined, (uint32_t)UINT16_MAX));
                } else {
                    merged.push_back(e);
                }
            }

            num_entries = (long)merged.size();
            entries = (EntryStruct*)malloc(num_entries * sizeof(EntryStruct));
            memcpy(entries, merged.data(), num_entries * sizeof(EntryStruct));
        } 

        // @brief Get move from book
        // @param key Zobrist key
        // @param minimum_weight Minimum weight of book moves to be returned (default 0)
        // @return Vector with the book moves (Move)
        map<Move, uint32_t> GetBookMoves(uint64_t key, uint16_t minimum_weight=0) {
            EntryStruct *entry;
            uint16_t move;
            Move bookMove;
            underlying u;
            map<Move, uint32_t> bookMoves;
            for (entry=entries; entry<entries+num_entries; entry++) {
                if (u.endian_swap_u64(entry->key) == key && u.endian_swap_u16(entry->weight) >= minimum_weight) {
                    move = u.endian_swap_u16(entry->move);
                    bookMove.from = ((move>>6) & 7) * 8 + ((move>>9) & 7); 
                    bookMove.to = ((move>>0) & 7) * 8 + ((move>>3) & 7);  
                    bookMove.promotion = ((move>>12) & 7); 
                    // White
                    if (bookMove.from == 4  && bookMove.to == 7)  { bookMove.is_castle = true; bookMove.to = 6; }  // e1→h1, remap to e1→g1
                    if (bookMove.from == 4  && bookMove.to == 0)  { bookMove.is_castle = true; bookMove.to = 2; }  // e1→a1, remap to e1→c1
                    // Black
                    if (bookMove.from == 60 && bookMove.to == 63) { bookMove.is_castle = true; bookMove.to = 62; } // e8→h8
                    if (bookMove.from == 60 && bookMove.to == 56) { bookMove.is_castle = true; bookMove.to = 58; } // e8→a8
                    bookMoves[bookMove] = u.endian_swap_u16(entry->weight);
                }
            }
            return bookMoves;
        }

        // @brief Search book for key. Unlike GetBookMoves(), which only returns the book moves, SearchBook()
        // also returns the weight and learn value of the book moves. 
        // @param key Zobrist key
        // @param minimum_weight Minimum weight of book moves to be returned (default 0)
        // @return Vector with the book entries (move, weight, learn)
        BookEntries SearchBook(uint64_t key, uint16_t minimum_weight=0) {
            EntryStruct *entry;
            uint16_t move;
            Move bookMove;
            underlying u;
            BookEntry book_entry;
            BookEntries bookEntries;
            for (entry=entries; entry<entries+num_entries; entry++) {
                if (u.endian_swap_u64(entry->key) == key && u.endian_swap_u16(entry->weight) >= minimum_weight) {
                    move = u.endian_swap_u16(entry->move); 
                    bookMove.from = ((move>>6) & 7) * 8 + ((move>>9) & 7); 
                    bookMove.to = ((move>>0) & 7) * 8 + ((move>>3) & 7);  
                    bookMove.promotion = ((move>>12) & 7); 
                    
                    if (bookMove.from == 4  && bookMove.to == 7)  { bookMove.is_castle = true; bookMove.to = 6; }  
                    if (bookMove.from == 4  && bookMove.to == 0)  { bookMove.is_castle = true; bookMove.to = 2; }  
                    if (bookMove.from == 60 && bookMove.to == 63) { bookMove.is_castle = true; bookMove.to = 62; } 
                    if (bookMove.from == 60 && bookMove.to == 56) { bookMove.is_castle = true; bookMove.to = 58; } 
                    book_entry.move = bookMove;

                    book_entry.weight = u.endian_swap_u16(entry->weight);
                    book_entry.learn = u.endian_swap_u32(entry->learn);
                    bookEntries.push_back(book_entry);
                }
            }
            return bookEntries;
        }  
        
        void Clear() {
            free(entries);
        } 
    };
}

#endif