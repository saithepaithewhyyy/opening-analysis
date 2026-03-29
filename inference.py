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
    for letter in "b":
        count = 0
        # url  = f"https://raw.githubusercontent.com/lichess-org/chess-openings/master/{letter}.tsv"
        url  = f"https://raw.githubusercontent.com/saithepaithewhyyy/Infinitely-Wide-Neural-Networks-for-Small-Data-Tasks/refs/heads/main/test.tsv"
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


engine = cc.ClassifierEngine()
engine.load_eco(load_eco_rows())
engine.load_priors({})
engine.build_index(max_depth=2)
# save_path = "index.bin"
# engine.save_index()

# ── Classify ─────────────────────────────────────────────────────────────────

def classify(fen: str, top_n: int = 35):
    results = engine.classify(fen, top_n=top_n)
    for r in results:
        print(f"{r.eco:4s}  {r.name:45s}  "
              f"post={r.posterior:.4f}  "
              f"path={r.path_length}")

# Sicialian Dragon (Accelerated) B34
classify("r1bqk1nr/pp1pppbp/2n3p1/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 1 6")