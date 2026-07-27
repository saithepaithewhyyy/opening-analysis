import numpy as np
import pandas as pd
from . import inference
from . import utils

def visualize(attn_maps, bb_grad):
    bb_grad = bb_grad[0]
    bb_grad_np = bb_grad.cpu().numpy()

    grad_data = []

    grad_data.append(utils.get_grad_values(bb_grad_np[0], title="occ"))

    for piece, index in utils.PIECE_INDEX.items():
        grad_data.append(utils.get_grad_values(bb_grad_np[index + 1], title=piece))

    grad_data.append(
        utils.get_grad_values(
            np.mean(bb_grad_np[1:], axis=0),
            title="mean"
        )
    )

    return grad_data

if __name__ == "__main__":
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    inference.inference(pos=fen, form="fen", topk=10, visual_flag=True, verbose=True)
