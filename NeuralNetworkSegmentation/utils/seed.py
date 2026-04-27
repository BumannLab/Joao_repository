import random
import numpy as np
import torch

def seed_everything(seed):
    """ Seeds all the packages into one seed
    

    Args:
        seed (_type_): _description_
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)