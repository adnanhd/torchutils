import os
import numpy as np
import torch
import random


def set_seed(seed: int):
    assert np.iinfo(np.uint32).min <= seed
    assert np.iinfo(np.uint32).max >= seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed) 
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
