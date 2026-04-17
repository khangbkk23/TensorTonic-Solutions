import numpy as np

def dropout(x, p=0.5, rng=None):
    x = np.asarray(x)

    if not (0 <= p < 1):
        raise ValueError("p must be in [0, 1)")

    if rng is not None:
        r = rng.random(x.shape)
    else:
        r = np.random.random(x.shape)

    keep = (r < (1 - p))

    pattern = keep.astype(x.dtype) / (1 - p)

    out = x * pattern

    return out, pattern