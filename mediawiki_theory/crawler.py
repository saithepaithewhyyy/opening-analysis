import bz2
import json
import xml.etree.ElementTree as ET

import demo as demo
from board_learn.train import CHECKPOINTS_DIR
import board_learn.utils as utils
import board_learn.inference as inference
import chess
import chess_classifier as cc
from tqdm import tqdm
import os
import pickle

from mediawiki_theory.piece_analysis.complete import position_evaluate
from . import parser
from typing import Dict

class WikibooksCrawler:
    def __init__(self, dump_file: str, max_pages: int = None):
        self.dump_file: str = dump_file
        self.theory_db: Dict = {}
        self.max_pages: int = max_pages 
        # get latest wikimedia dump from here, I am too lazy to automatically retrieve the dump
        self.dump_base: str = "https://dumps.wikimedia.org/enwikibooks/latest"
        
        engine = cc.ClassifierEngine()
        save_path = "index.bin"
        _, priors = demo.load_data() 
        engine.load_index(save_path)
        engine.load_priors(priors) 
        
        self.engine = engine

    def crawl(self):
        count = 0

        context = ET.iterparse(
            bz2.open(self.dump_file, "rb"),
            events=("end",)
        )

        pbar = tqdm(desc="crawling wikimedia dump", unit="page")
        for _, elem in context:

            if not elem.tag.endswith("page"):
                continue

            title = elem.findtext("{*}title")

            if (title is None or not title.startswith("Chess Opening Theory")):
                elem.clear()
                continue

            revision = elem.find("{*}revision")
            if revision is None:
                elem.clear()
                continue

            text = revision.findtext("{*}text") or ""

            board = parser.board_from_page(title=title)
            if board is None:
                elem.clear()
                continue

            fen = board.fen()

            bayesian_result, topk_probs, topk_classes, grad_data = parser.get_data(fen=fen, engine=self.engine)
            piece_analysis = position_evaluate(fen)

            entry = self.theory_db.setdefault(
                cc.fen_to_hash(fen),
                {
                    "fen": fen,
                    "bayesian_result": [
                    {
                        "eco": f"{r.eco:4s}",
                        "name": f"{r.name}",
                        "posterior": f"{r.posterior:.4f}",
                        "probability": f"{r.probability:.4f}",
                    }
                        for r in bayesian_result
                    ],
#                    "grad_data": grad_data,
                    "piece_analysis": piece_analysis,
                    "entries": [],
                },
            )

            entry["entries"].append(
                {
                    "title": title,
                    "theory": parser.parse_theory(text),
                }
            )

            elem.clear()

            count += 1
            pbar.update(1)

            if self.max_pages is not None and count >= self.max_pages:
                break

            elem.clear()

        pbar.close()

        print(f"Crawled {count} pages.")
        print(f"Stored {len(self.theory_db)} positions.")

    def get_theory(self, fen):
        return self.theory_db.get(fen)

    def save(self, filename="wikibooks_theory.json"):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(
                self.theory_db,
                f,
                ensure_ascii=False,
                indent=4,
            )


if __name__ == "__main__":
    # need to change to dynamically pick the latest version if file doesnt exit on local
    # nah im not doing allat, just download it lol its surpirisingly only like 200 mb
    crawler = WikibooksCrawler(
        "enwikibooks-latest-pages-articles.xml.bz2"
    )
    crawler.crawl()
    crawler.save()

