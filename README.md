# Opening Analysis
 
A fast bayesian chess opening classifier exposed as a Python extension via pybind11. Given a FEN position, it returns a ranked list of ECO openings with posterior probabilities. Move generation is heuristically scored and optionally guided by Polyglot opening books. 
 
## Architecture
 
```
board.hpp / board.cpp       — Board representation, FEN parsing, Zobrist hashing
zobrist_key.hpp             — Official Polyglot Random64 table (781 entries)
definitions.hpp             — Platform-specific bit intrinsics (MSVC / GCC)
movegen.hpp / movegen.cpp   — Legal move generation, move scoring, softmax normalisation
reader.hpp                  — Polyglot .bin book reader with multi-book merge support
classifier.hpp / classifier.cpp — Bayesian BFS index builder and classifier engine
bindings.cpp                — pybind11 Python bindings
```
 
### How it works
 
`build_index` runs a BFS from each ECO root position up to `max_depth` plies, using `generate_legal_scored_moves` to compute transition probabilities. These are accumulated into a hash map keyed by Zobrist hash. At query time, `classify` looks up the position hash, and uses a bayesian methodology to return the top-N openings by posterior.
 
Opening book weights (`2*wins + draws`) are blended into move scoring proportionally, so book-supported moves are favoured during BFS without overriding the heuristics entirely.
 
## Requirements
-

## Building

Can be built via `pip install .` in chess_classifier. `inference.py` gives a rundown on how things work via Python bindings.

# Note!

This is still in early stages; there's still quite a bit to come.
