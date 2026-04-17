import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    x = np.asarray(x)
    return np.maximum(0, x)