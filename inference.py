import csv, io, requests
import chess_classifier as cc
import chess
from typing import Optional, Tuple
import opening_priors as op

def pgn_process(pgn_string: str) -> Optional[str]:
    board = chess.Board()

    tokens = pgn_string.strip().split()
    for token in tokens:
        if token.endswith("."):
            continue
        try:
            move = board.parse_san(token)
        except Exception:
            print("Somehthing went wrong in PGN processing, skipping!")
            return None  # invalid PGN

        board.push(move)

    return board.fen()

def load_data() -> tuple[list[tuple[str,str,str]], dict[str, Optional[float]]]:
    rows = []
    eco_fen = {}
    for letter in "abcde":
        count = 0
        url  = f"https://raw.githubusercontent.com/lichess-org/chess-openings/master/{letter}.tsv"
        response = requests.get(url)
        text = response.text

        if response.status_code == 200:
            print(f"Loaded {letter} openings successfully")  
        else:
            print("Failed to load")

        for r in csv.DictReader(io.StringIO(text), delimiter="\t"):
            pgn = (r.get("pgn") or "").strip()
            eco = (r.get("eco") or "").strip()
            name = (r.get("name") or "").strip()

            if pgn:
                fen = pgn_process(pgn)
                if fen is not None: 
                    eco += str(count)  
                    rows.append((eco, name, fen)) 
                    eco_fen[eco] = fen
                    count = count + 1

    print("done")            
    return rows, op.get_priors(eco_fen)

def classify(fen: str, top_n: int = 3, verbose: bool = True) -> list[tuple[str, str, float, float, int]]:
    results = engine.classify(fen, top_n=top_n)
    prob = 1.0
    EPS = 1e-4
    filtered = []

    for r in results:
        if prob > EPS:
            filtered.append(r)
            prob -= r.posterior
        else:
            break

    if verbose: 
        for r in filtered:
            print(f"{r.eco:4s}  {r.name:45s}  "
                f"post={r.posterior:.4f}  "
                f"likelihood={r.likelihood:.4f} "
                f"path={r.path_length} ")

    return filtered


if __name__ == "__main__":
    rows, priors = load_data()
    engine = cc.ClassifierEngine()
    engine.load_eco(rows)
    engine.load_priors(priors)
    engine.load_book(["C:/praneeth/progs/py/opening-analysis/opening_books/Human-polyglot/Human.bin"])
    engine.build_index(max_depth=4)
    save_path = "index.bin"
    engine.save_index(save_path)

    # rows, priors = load_data()
    # engine = cc.ClassifierEngine()
    # save_path = "index.bin"
    # engine.load_index(save_path)
    # engine.load_priors(priors)

    # Sicialian Dragon (Accelerated) B34
    classify("r1bqkbnr/pp1ppp1p/2n3p1/8/3NP3/2N5/PPP2PPP/R1BQKB1R b KQkq - 1 5")
    print("\n")
    # # Nimzo Larsen Attack: Modern Variation
    classify("rnbqkbnr/pppp1ppp/8/4p3/8/1P6/P1PPPPPP/RNBQKBNR w KQkq - 0 2")
    print("\n")
    # Scandivanian Defence, Modern Variation B02
    classify("rnbqkb1r/ppp1pppp/5n2/3P4/8/2N5/PPPP1PPP/R1BQKBNR b KQkq - 2 3")
    print("\n")
    # Nimzowitsch Defense: El Columpio Defense, Pin Variation (+1 plies)
    classify("r1bqkb1r/1pp1pppp/p1np3n/1B2P3/3P4/5N1P/PPP2PP1/RNBQK2R w KQkq - 0 7")
    print("\n")
    # Nimzowitsch Defense: El Columpio Defense, Pin Variation (+2 plies)
    classify("r1bqkb1r/1pp1pppp/p1Bp3n/4P3/3P4/5N1P/PPP2PP1/RNBQK2R b KQkq - 0 7")
    print("\n")
    # Nimzowitsch Defense: El Columpio Defense, Pin Variation (+3 plies)
    classify("r1bqkb1r/2p1pppp/p1pp3n/4P3/3P4/5N1P/PPP2PP1/RNBQK2R w KQkq - 0 8")
    print("\n")
    # depth testing (king pawn/morphy variaton branches)
    classify("r1bqkbnr/1ppp1ppp/p1n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 1 4")
    print("\n")
    classify("r1bqkb1r/1ppp1ppp/p1n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 5")
    print("\n")
    classify("r1bqkb1r/1ppp1ppp/p1n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 0 5")
    print("\n")
    # RANDOM
    classify("rnbqkbnr/1ppp1ppp/8/p3p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3")
    print("\n")