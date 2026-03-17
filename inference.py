# inference.py
import csv, io, requests
import opening_classifier as cc   # the compiled .so

# ── Load ECO db once at startup ──────────────────────────────────────────────

def load_eco_rows() -> list[tuple[str,str,str]]:
    rows = []
    for letter in "abcde":
        url  = f"https://raw.githubusercontent.com/lichess-org/chess-openings/master/{letter}.tsv"
        text = requests.get(url).text
        for r in csv.DictReader(io.StringIO(text), delimiter="\t"):
            epd = r.get("epd","").strip()
            if epd:
                rows.append((r["eco"], r["name"], epd))
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

classify("rnbqkb1r/pp2pppp/3p1n2/2p5/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 0 4")