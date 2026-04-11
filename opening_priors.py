import requests
import os
from dotenv import load_dotenv
from typing import Dict, Optional
import chess
import json
import inference
from tqdm import tqdm

load_dotenv()
TOKEN = os.getenv("LICHESS_TOKEN")
API = "https://explorer.lichess.org/lichess"
BASE_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
PRIOR_FILE = "priors.json"

def call_api(fen:Optional[str] = BASE_FEN, variant:str = "standard", speeds:str = "blitz,rapid,classical"):
    r = requests.get(
        API,
        params={"variant": variant, "speeds": speeds, "fen": fen},
        headers={"Authorization": f"Bearer {TOKEN}"}
    )

    if r.status_code != 200:
        print(f"ERROR")  

    data = r.json()
    tot_games: int = data["white"] + data["black"] + data["draws"]
    moves = data["moves"]    

    return tot_games, moves

def add_move(san: str, fen: Optional[str]) -> str:
    board = chess.Board()
    if fen is not None: board.set_fen(fen=fen)
    board.push_san(san=san)
    return board.fen()

def get_priors(eco_fen: Dict[str, Optional[str]]) -> Dict[str, Optional[float]]:

    priors: Dict[str, Optional[float]] = {}

    if os.path.exists(PRIOR_FILE):
        with open(PRIOR_FILE, 'r') as file:
            priors = json.load(file)
            return priors

    N, _ = call_api()
    fen_eco = {v: k for k, v in eco_fen.items()}
    for eco, fen in tqdm(eco_fen.items(), desc="loading priors"):
        if eco not in priors:
            count, moves = call_api(fen)
            priors[eco] = float((count + 0.0) / N)

            for move in moves:
                new_fen = add_move(move["san"], fen)
                if new_fen is not None and new_fen in eco_fen.values():
                    priors[fen_eco[fen]] = float((move['white'] + move['draws'] + move['black'] + 0.0)/N)

    with open(PRIOR_FILE, 'w', encoding='utf-8') as f:
        json.dump(priors, f)

    return priors

if __name__ == "__main__":
    inference.load_data()