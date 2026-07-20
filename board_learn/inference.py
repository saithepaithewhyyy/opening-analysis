from . import utils
from .utils import sparse_kl_loss
import os
import torch
import pickle

from . import opening_model as om
from . import visualisation as viz

CHECKPOINTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
FINAL_MODEL_PATH = "final_model.pt"

def inference(pos, form="fen", topk=5, visual_flag=True, verbose=True, checkpoint_dir=CHECKPOINTS_DIR):
    device = torch.device('cuda' if torch.cuda.is_available()
                        else 'mps' if torch.backends.mps.is_available()
                        else 'cpu')
    
    with open(os.path.join(checkpoint_dir, "eco_classes.pkl"), "rb") as f:
        eco_classes = pickle.load(f)

    model = om.OpeningModel(n_classes=len(eco_classes)).to(device)
    ckpt = torch.load(os.path.join(checkpoint_dir, FINAL_MODEL_PATH))
    model.load_state_dict(ckpt["model_state_dict"])
    bb, sc = utils.inference_features(pos, form)
    torch.compile(model)
    model.eval()

    bb = torch.as_tensor(bb, dtype=torch.float32, device=device).unsqueeze(0)
    sc = torch.as_tensor(sc, dtype=torch.float32, device=device).unsqueeze(0)
    attn_maps = {}
    
    grad_plots = []
    if visual_flag:
        bb = bb.requires_grad_(True)
        out, attn_maps = model.forward_with_attn_maps(bb, sc)
        # this is literally just a hotfix, i need to think of a better etric that I can actually use
        target = out[:, out.argmax(dim=-1)].sum()
        target.backward()
        grad_plots = viz.visualize(attn_maps, bb.grad)
    else:
        with torch.no_grad():
            out = model(bb, sc)
    
    probs = torch.exp(out)
    topk_probs, topk_idx = probs.topk(topk, dim=-1)
   
    if verbose:
        for b in range(len(topk_idx)):
            print("---------------------------------")
            print(f"top {topk} possible openings")
            print("---------------------------------")
            max_len = max(len(name) for name in eco_classes)
            for idx, prob in zip(topk_idx[b].tolist(), topk_probs[b].tolist()):
                print(f"{eco_classes[idx]:<{max_len}} | {prob}")
     
    return topk_probs, topk_idx, grad_plots

if __name__ == "__main__":
    fen = "rnbqkb1r/pppp1ppp/5n2/4p3/2b1p3/3p4/ppp2ppp/rnbqk1nr b kqkq - 0 3"
    inference(pos=fen, form="fen", topk=10, visual_flag=False, verbose=True)
