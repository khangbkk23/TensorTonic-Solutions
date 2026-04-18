import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    # Write code here
    x = np.asarray(x)
    mean_x = np.mean(x)
    n = len(x)
    s_square = np.sum((x - mean_x)**2)
    return float(s_square/(n-1)),float(np.sqrt(s_square / (n-1)))