import torch
import numpy as np
import random


def seed_all(seed=69):
    random.seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)
