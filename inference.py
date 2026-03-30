import csv, io, requests
import chess_classifier as cc
import chess
from typing import Optional

def pgn_process(pgn_string: str) -> Optional[str]:
    board = chess.Board()

    tokens = pgn_string.strip().split()
    for token in tokens:
        if token.endswith("."):
            continue
        try:
            move = board.parse_san(token)
        except Exception:
            return None  # invalid PGN

        board.push(move)

    return board.fen()


def load_eco_rows() -> list[tuple[str,str,str]]:
    rows = []
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
                    rows.append((eco+str(count), name, fen)) 
                    
            count = count + 1
                
    return rows

# rows = load_eco_rows()
# engine = cc.ClassifierEngine()
# engine.load_eco(rows)
# engine.load_priors({})
# engine.build_index(max_depth=2)
save_path = "index.bin"
# engine.save_index(save_path)

engine = cc.ClassifierEngine()
engine.load_index(save_path)

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

# Sicialian Dragon (Accelerated) B34
classify("r1bqkbnr/pp1ppp1p/2n3p1/8/3NP3/2N5/PPP2PPP/R1BQKB1R b KQkq - 1 5")
# # Nimzo Larsen Attack: Modern Variation
classify("rnbqkbnr/pppp1ppp/8/4p3/8/1P6/P1PPPPPP/RNBQKBNR w KQkq - 0 2")
# Scandivanian Defence, Modern Variation B02
classify("rnbqkb1r/ppp1pppp/5n2/3P4/8/2N5/PPPP1PPP/R1BQKBNR b KQkq - 2 3")