import torch 
import torch.nn as nn
from pdb import set_trace as st
import os

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

def save_network(network, exp_dir, iter_step):
    save_filename = 'model_{}.pkl'.format(iter_step)
    save_path = os.path.join(exp_dir, save_filename)
    torch.save(network.cpu(), save_path)