import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    # Write code here
    X = np.asarray(x)
    if X.ndim == 3:
        return X.mean(axis=(1, 2)).tolist()
    elif X.ndim == 4:
        return X.mean(axis=(2,3)).tolist()
    else:
        raise ValueError()