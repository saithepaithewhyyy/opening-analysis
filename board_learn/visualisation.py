import numpy as np
import pandas as pd
import inference as inf
import utils

def visualize(attn_maps, bb_grad):
    # Bitboard gradient features and visualisations
    # bb_grad (1, 13, 64) -> bb_grad (13, 64)
    bb_grad = bb_grad[0]
    bb_grad_np = bb_grad.cpu().numpy()
    
    grad_plots = []
    grad_plots[0] = utils.plot_grads(bb_grad_np[0])
    
    for _, index in utils.PIECE_INDEX.items():
        grad_plots[index+1] = utils.plot_grads(bb_grad_np[index+1])

    grad_plots[13] = utils.plot_grads(np.mean(bb_grad_np[1:], axis=0, keepdims=True))

    return grad_plots

if __name__ == "__main__":
    fen = "rnbqkbnr/pp1ppp1p/6p1/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3"
    inf.inference(pos=fen, form="fen", topk=10, visual_flag=True)
