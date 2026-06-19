import numpy as np
import pandas as pd
import inference as inf
import utils

def visualize(attn_maps, bb_grad):
    bb_grad = bb_grad[0]
    bb_grad_np = bb_grad.cpu().numpy()
    grad_plots = [None] * 14

    grad_plots[0] = utils.plot_grads(bb_grad_np[0], title='occ')

    for piece, index in utils.PIECE_INDEX.items():
        grad_plots[index + 1] = utils.plot_grads(bb_grad_np[index + 1], title=piece)

    grad_plots[13] = utils.plot_grads(np.mean(bb_grad_np[1:], axis=0), title='mean')

    return grad_plots

if __name__ == "__main__":
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    inf.inference(pos=fen, form="fen", topk=10, visual_flag=True)
