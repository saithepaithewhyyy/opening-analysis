import utils
import os
import torch
import pickle

import opening_model as om
import visualisation as viz

CHECKPOINTS_DIR = "checkpoints"
FINAL_MODEL_PATH = "final_model.pt"

def inference(pos, form="fen", topk=5, visual_flag=True, checkpoint_dir=CHECKPOINTS_DIR):
    device = torch.device('cuda' if torch.cuda.is_available()
                        else 'mps' if torch.backends.mps.is_available()
                        else 'cpu')
    
    with open(os.path.join(checkpoint_dir, "eco_classes.pkl"), "rb") as f:
        eco_classes = pickle.load(f)

    model = om.OpeningModel(n_classes=len(eco_classes)).to(device)
    ckpt = torch.load(os.path.join(checkpoint_dir, FINAL_MODEL_PATH))
    model = torch.compile(model)
    model.load_state_dict(ckpt["model_state_dict"])
    bb, sc = utils.inference_features(pos, form)

    bb = torch.as_tensor(bb, dtype=torch.float32, device=device).unsqueeze(0).requires_grad_(True)
    sc = torch.as_tensor(sc, dtype=torch.float32, device=device).unsqueeze(0)
    attn_maps = {}
    
    if visual_flag:
        out, attn_maps = model.forward_with_attn_maps(bb, sc)
    else:
        with torch.no_grad():
            out = model(bb, sc)
    
    probs = torch.exp(out)
    topk_probs, topk_idx = probs.topk(topk, dim=-1)
   
    for b in range(len(topk_idx)):
        print("---------------------------------")
        print(f"Top {topk} possible openings")
        print("---------------------------------")
        max_len = max(len(name) for name in eco_classes)
        for idx, prob in zip(topk_idx[b].tolist(), topk_probs[b].tolist()):
            print(f"{eco_classes[idx]:<{max_len}} | {prob}")

    if visual_flag:
        viz.visualize(attn_maps, bb.grad[0])
    
    return topk_probs, topk_idx

if __name__ == "__main__":
    fen = "rnbqkbnr/pp1ppp1p/6p1/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3"
    inference(pos=fen, form="fen", topk=10, visual_flag=False)
