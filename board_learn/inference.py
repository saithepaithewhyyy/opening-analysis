import utils
import os
import torch

import opening_model as om

def inference(pos, form, checkpoint_dir="checkpoints"):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # how do i import num_classes?
    model = om.OpeningModel(n_classes=4000).to(device)
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "final_model.pt")))
    model = torch.compile(model)
    
    bb, sc = utils.inference_features(pos, form)
    out = model(bb, sc)
    return out