import utils
import os
import torch

import opening_model as om

def inference(pos, form, topk=5, checkpoint_dir="checkpoints"):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # how do i import num_classes eco_classes?
    eco_classes = []
    model = om.OpeningModel(n_classes=4000).to(device)
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "final_model.pt")))
    model = torch.compile(model)
    bb, sc = utils.inference_features(pos, form)
    
    with torch.no_grad():
        out = model(bb, sc)
    probs = torch.exp(out)
    topk_probs, topk_idx = probs.topk(topk, dim=-1)
   
    for b in range(len(topk_idx)):
        print("Top {topk} predicted openings with probabilities \n")
        print("------------------------------------------------------------")
        for idx, prob in zip(topk_idx[b].tolist(), topk_probs[b].tolist()):
            print(f"{eco_classes[idx]} | {prob} \n")
    
    return topk_probs, topk_idx