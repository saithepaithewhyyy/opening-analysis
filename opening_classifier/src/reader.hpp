#ifndef READER_HPP
#define READER_HPP

#include <iostream>
#include "board.hpp"
#include <cstdio>
#include <fstream>
#include <string>
#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>

using namespace std;

// Taken from https://github.com/deoshreyas/Chess-Polyglot-Reader
// Much more easier than making my own polyglot reader

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

    static long int num_entries = 0;

    static EntryStruct* entries;

    static std::string Files[8] = {"a", "b", "c", "d", "e", "f", "g", "h"};
    static std::string Rows[8] = {"1", "2", "3", "4", "5", "6", "7", "8"};

    struct BookEntry {
        Move move;
        uint16_t weight;
        uint32_t learn;
    };

    typedef vector<BookEntry> BookEntries;
}

namespace Reader {
    class Book {
    public:
        void Load(const char *path) {
            FILE *file = std::fopen(path, "rb");
            if (file==NULL) {
                cerr << "<Error> Please use valid book" << std::endl;
                return;
            } else {
                fseek(file, 0, SEEK_END);
                long position = ftell(file);
                if (position < sizeof(EntryStruct)) {
                    cerr << "<Error> No entries found" << std::endl;
                    return;
                }
                num_entries = position / sizeof(EntryStruct);
                entries = (EntryStruct*)malloc(num_entries * sizeof(EntryStruct));
                rewind(file);
                size_t returnValue = fread(entries, sizeof(EntryStruct), num_entries, file);
                fclose(file);
            }
        }

        // @brief Get move from book
        // @param key Zobrist key
        // @param minimum_weight Minimum weight of book moves to be returned (default 0)
        // @return Vector with the book moves (toFile, toRow, fromFile, fromRow, promotion, weight)
        Moves GetBookMoves(uint64_t key, uint16_t minimum_weight=0) {
            EntryStruct *entry;
            uint16_t move;
            Move bookMove;
            underlying u;
            Moves bookMoves;
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
                    
                    bookMoves.push_back(bookMove);
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
            std::free(entries);
        } 
    };
}

#endif