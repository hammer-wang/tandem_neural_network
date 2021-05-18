import torch
import os
import numpy as np

def set_torch_seed(seed):
    """
    Sets the pytorch seeds for current experiment run
    :param seed: The seed (int)
    :return: A random number generator to use
    """
    rng = np.random.RandomState(seed=seed)
    torch_seed = rng.randint(0, 999999)
    torch.manual_seed(seed=torch_seed)

    return rng

def load_model(net, name):
    state_dict = torch.load(os.path.join('./trained_models', name), map_location=lambda storage, loc: storage)
    net.load_state_dict(state_dict)
    del state_dict
    torch.cuda.empty_cache()
    return net

def serialize_model(log_path, net, name):
    print('serializing model to {}'.format(log_path))
    torch.save(net.state_dict(), os.path.join(log_path, name))
