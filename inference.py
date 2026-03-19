import csv, io, requests
import chess_classifier as cc


def load_eco_rows() -> list[tuple[str,str,str]]:
    rows = []
    for letter in "abcde":
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
                rows.append((eco, name, pgn)) 
                
    return rows

engine = cc.ClassifierEngine()
engine.load_eco(load_eco_rows())

# ── Classify ─────────────────────────────────────────────────────────────────

def classify(fen: str, top_n: int = 5):
    results = engine.classify(fen, top_n=top_n)
    for r in results:
        print(f"{r.eco:4s}  {r.name:45s}  "
              f"post={r.posterior:.4f}  "
              f"path={r.path_length}")

classify("rnbqkbnr/pp1ppppp/2p5/3P4/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2")