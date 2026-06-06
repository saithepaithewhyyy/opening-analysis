# Opening Analysis
Identifying a chess opening from an arbitrary and static board position is non-trivial, unlike dynamic lines, which can be directly mapped to known openings. Several factors affect this, some such as transpositions, move-order variations, and overlapping ECO families, all of which complicate a naive lookup. This project approaches the problem in two ways: a Bayesian classifier built on a probabilistic BFS index, and a dual-stream transformer that learns the structural characteristics of openings from the same data.

Why is identifying a static position important? Each opening position, regardless of where it came from, has its own heatmap; in other words, certain positional and structural characteristics that define it, as well as the game to come. Identifying such characteristics in an opening position that does not stem from popular study is crucial in understanding how to proceed.

**Note:-** This is still largely in dev, there's a lot that I plan to do with this!
 
## Architecture

This project consists of two parts: the opening classifier and the board opening positional learning model. 

### Opening Classifier

The opening classifier is a fast Bayesian search model exposed as a Python extension via pybind11. The indexing code runs a massive multithreaded BFS search (aided by move scoring heuristics and polyglot opening books) to index all positions to a given depth. 

Given a FEN position, the Bayesian classifier then returns a ranked list of ECO openings with posterior probabilities. Why is it Bayesian? Well, each position is probabilistically scored at indexing time, and on inference time, the posterior is calculated by the simple Bayes rule. The priors for each opening are calculated using the frequency-of-play statistics from the Lichess API

```
board.hpp / board.cpp           — Board representation, FEN parsing, Zobrist hashing
zobrist_key.hpp                 — Official Polyglot Random64 table hashes
definitions.hpp                 — Platform-specific bit intrinsics (MSVC / GCC)
movegen.hpp / movegen.cpp       — Legal move generation and scoring
reader.hpp                      — Polyglot.bin book reader
classifier.hpp / classifier.cpp — Bayesian BFS index builder and classifier engine
bindings.cpp                    — pybind11 Python bindings

demo.py                         — Demo of indexing and classifying
opening_priors.py               — Fetches the priors for each opening from Lichess' API
```

### Board Learn

A simple dual stream transformer model that learns positional and piece (self and cross) relationships that are central characteristics of openings. The data used to train is directly taken from `index.bin`. Following is the architecture of the transformer model used:- 

<img width="800" height="679" alt="image" src="https://github.com/user-attachments/assets/3a7c3f0a-a1b2-4c4f-a21a-d6ff9471bc21" />


### How it works
 
`build_index` runs a BFS from each ECO root position up to `max_depth` plies, using `generate_legal_scored_moves` to compute transition probabilities. These are accumulated into a hash map keyed by the polyglot Zobrist hash. At query time, `classify` looks up the position hash and uses a Bayesian posterior calculation to return the top-N openings by posterior.
 
Opening book weights (`2*wins + draws`) from the polyglot books are blended into move scoring proportionally, so book-supported moves are favoured during BFS without overriding the heuristics entirely.

## Requirements and Building

Works on both Windows and Mac. (I have not checked on Linux, but I'm sure it'll work)

### Requirements:- 
- C++ 17
- Python 3+
- A couple of python libraries included in the requirements.txt 

### Build and Run:-

It's quite simple, but here are the steps to build and run both components:-
For your safety and sanity, consider using a venv; it keeps things clean even though the code is a dumpster pile.

The opening classifier library can be built via `pip install .` in opening_classifier. `demo.py` gives a rundown on how things work. The demo file will run the indexing at a default depth of 3 plies and classify some randomly chosen opening positions. It also creates an `index.bin` file at root, which can be used for further runs.

For `board_learn`, the `train.py` file takes care of data loading (from the `index.bin` file) as well as the training. `inference.py` runs the model against a random opening position for a sanity check.

One last note, the opening classifier is also powered by close to 50 polyglot books. The indexing works without the books, but it might give slightly worse results, especially at larger depths. For now I havent added the books to the repo, I have to think how to that.

## Note!
This is still in dev; there's still quite a bit to come.
