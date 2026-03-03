import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    """
    # Write code here
    x = np.asarray(x, dtype=float)
    mean = float(np.mean(x))
    median = float(np.median(x))

    values, counts = np.unique(x, return_counts=True)
    max_count = np.max(counts)
    modes = values[counts == max_count]

    mode = float(np.min(modes))


    return mean, median, mode