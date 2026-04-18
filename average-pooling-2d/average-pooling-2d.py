import numpy as np
def average_pooling_2d(X, pool_size):
    """
    Apply 2D average pooling with non-overlapping windows.
    """
    # Write code here
    X = np.asarray(X)
    H, W = X.shape
    H_out = H//pool_size
    W_out = W//pool_size
    out = np.zeros((H_out, W_out))
    stride = pool_size
    for i in range(H_out):
        for j in range(W_out):
            h_start = i*stride
            h_end = h_start+pool_size
            w_start = j*stride
            w_end = w_start+pool_size
            window = X[h_start:h_end,w_start:w_end]
            out[i,j] = np.mean(window)

    return out.tolist()