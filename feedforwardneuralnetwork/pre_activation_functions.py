# Pre Avtivation function

import numpy as np

def pre_activation(W, h, b):
    return np.add(np.matmul(W,h) , b)