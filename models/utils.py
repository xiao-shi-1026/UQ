import torch
from collections import OrderedDict

def load_model(model_class, model_path):
    model = model_class()
    checkpoint = torch.load(model_path, map_location="cpu")
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    return model