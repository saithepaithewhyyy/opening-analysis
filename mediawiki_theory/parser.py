from tabnanny import verbose


from board_learn.inference import inference
from board_learn.train import CHECKPOINTS_DIR
import board_learn.utils as utils
import board_learn.visualisation as viz
import chess
import chess_classifier as cc
import json
import mwparserfromhell as mw
import os
import pickle
import re

STOP_SECTIONS = {
    "Theory table",
    "References",
    "See also",
    "External links",
    "Further reading",
    "Bibliography",
    "Notes",
    "Sources",
}

def board_from_page(title: str):
    board = chess.Board()

    parts = title.split("/")

    for part in parts:
        if part == "Chess Opening Theory" or part == "REDIRECT Chess Opening Theory":
            continue

        if "..." in part:
            san = part.split("...", 1)[1]
        elif "._" in part:
            san = part.split("._", 1)[1]
        elif "." in part:
            san = part.split(".", 1)[1]
        else:
            continue

        san = san.replace("_", "").strip()

        if san:
            try:
                board.push_san(san)
            except Exception:
                return None

    return board

def get_data(engine, fen: str =""):
    if fen == "":
        print("fen string missing, need fen!")
        return ""
    
    bayesian_result = utils.classify(fen=fen, engine=engine, verbose=False)
    topk_probs, topk_idx, grad_data, pos_encoding = inference(pos=fen, form="fen", topk=3, visual_flag=True, verbose=False)
    pos_encoding = pos_encoding.squeeze(0).detach().cpu().tolist()
#    topk_probs, topk_idx, grad_data = [], [], []
 
    with open(os.path.join(CHECKPOINTS_DIR, "eco_classes.pkl"), "rb") as f:
        eco_classes = pickle.load(f)
    
    topk_classes =[[eco_classes[idx] for idx in indices.tolist()] for indices in topk_idx]  
    return bayesian_result, topk_probs, topk_classes, grad_data, pos_encoding

def parse_theory(wikitext: str) -> str:
    code = mw.parse(wikitext)
    text = code.strip_code(
        normalize=True,
        collapse=True,
    )

    output = []
    skipping = False

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue
        m = re.match(r"^(=+)\s*(.*?)\s*=+$", line)
        if m:
            heading = m.group(2).strip()
            level = max(1, len(m.group(1)) - 1)
            output.append("#" * level + " " + heading)
            continue

        if skipping:
            continue

        line = re.sub(r"<[^>]+>", "", line)
        line = re.sub(r"\s+", " ", line)

        if line:
            output.append(line)

    text = "\n\n".join(output)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    
    if len(text) >= 1 and text.split(maxsplit=1)[0] == "REDIRECT":
        text = "REDIRECT from position: " + str(board_from_page(text).fen())

    return text
