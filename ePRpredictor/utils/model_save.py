######################################
# model save and load
######################################
import os
import pickle

import torch


def save_checkpoint(dir, model, optimizer, xgb_model=None):
    state_dict = {
        "model": model.state_dict() if model else {},
        "optimizer": optimizer.state_dict() if optimizer else {},
    }
    torch.save(state_dict, f"{dir}/bert.pth.tar")
    if xgb_model is not None:
        pickle.dump(xgb_model, open(f"{dir}/xgb_pickle.dat", "wb"))


def load_checkpoint(dir, model, optimizer=None, xgb_model=None):
    if not os.path.exists(dir):
        assert 1 == 2, "Sorry, don't have checkpoint.pth file, continue training!"
        return
    checkpoint = torch.load(f"{dir}/bert.pth.tar")
    res = {}
    if model:
        model.load_state_dict(checkpoint["model"])
        res["model"] = model
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])
        res["optimizer"] = optimizer
    if xgb_model:
        xgb_model = pickle.load(open(f"{dir}/xgb_pickle.dat", "rb"))
        res["xgb_model"] = xgb_model
    return res
